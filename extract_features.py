"""
extract_features.py
-------------------
Feature extraction pipeline for the MVG-VQA system.

Loads each video, extracts spatial/temporal/MVG features, caches results to
disk, and saves the final feature matrix + MOS labels to ``data/features.pkl``.

Usage
-----
    python extract_features.py

Outputs
-------
    data/features.pkl  — joblib dump of (DataFrame, Series)
                         DataFrame: [N_videos x N_features]
                         Series:    MOS labels aligned by video ID
"""

import os
import cv2
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from src.spatial_features import sharpness_stats, niqe_stats, canny_edge_stats, lbp_stats
from src.temporal_features import (
    chroma_instability, optical_flow_pairs, flow_stats,
    fb_consistency, warp_error, ssim_stats,
)
from src.advanced_temporal_features import (
    optical_flow_variance, zero_flow_stats, frame_difference_stats,
    fft_flicker, blockiness_stats,
)
from src.mvg_features import build_per_frame_vectors, mvg_stats

# Optional deep features — comment out if not available
try:
    from src.nima_features import get_deep_feature_stats
    _DEEP_FEATURES = True
except ImportError:
    _DEEP_FEATURES = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VIDEO_DIR  = "data/train_videos"
LABEL_FILE = "data/labels.csv"
CACHE_DIR  = "cache/features"
OUT_FILE   = "data/features.pkl"
MAX_FRAMES = None          # Set to an int to cap frames per video (faster dev)
N_JOBS     = -1            # Parallel workers; -1 = all CPUs


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------

def load_video_ycbcr(path: str, max_frames: int | None = None) -> list:
    """Load a video as a list of (Y, Cb, Cr) float32 frame tuples.

    Args:
        path:       Path to video file.
        max_frames: Optional cap on the number of frames loaded.

    Returns:
        List of (Y, Cb, Cr) tuples, each channel float32 in [0, 1].
        Returns an empty list if the file cannot be opened.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)

        frames.append((
            Y.astype(np.float32)  / 255.0,
            Cb.astype(np.float32) / 255.0,
            Cr.astype(np.float32) / 255.0,
        ))

        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Per-video feature extraction
# ---------------------------------------------------------------------------

def extract_features_from_video(path: str, fps: float = 30.0) -> dict | None:
    """Extract all features from a single video file.

    Args:
        path: Path to the video file.
        fps:  Frame rate (used for FFT-flicker frequency axis).

    Returns:
        Dict of feature name → float, or None if the video has fewer than
        2 frames or cannot be loaded.
    """
    frames = load_video_ycbcr(path, max_frames=MAX_FRAMES)
    if len(frames) < 2:
        return None

    feats: dict = {}

    # Spatial features
    feats.update(sharpness_stats(frames))
    feats.update(niqe_stats(frames))
    feats.update(canny_edge_stats(frames))
    feats.update(lbp_stats(frames))

    # Temporal features
    flows_fw = optical_flow_pairs(frames)
    feats.update(chroma_instability(frames))
    feats.update(flow_stats(flows_fw))
    feats.update(fb_consistency(frames, flows_fw))
    feats.update(warp_error(frames, flows_fw))
    feats.update(ssim_stats(frames))

    # Advanced temporal features
    feats.update(optical_flow_variance(flows_fw))
    feats.update(zero_flow_stats(flows_fw))
    feats.update(frame_difference_stats(frames))
    feats.update(fft_flicker(frames, fps=fps))
    feats.update(blockiness_stats(frames))

    # MVG statistical features
    Fs = build_per_frame_vectors(frames, flows_fw)
    mvg_feats, _, _ = mvg_stats(Fs)
    feats.update(mvg_feats)

    # Optional deep features
    if _DEEP_FEATURES:
        feats.update(get_deep_feature_stats(frames))

    # Sanitise NaN / Inf
    for k, v in feats.items():
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                feats[k] = 0.0
        elif isinstance(v, np.ndarray):
            feats[k] = np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)

    return feats


# ---------------------------------------------------------------------------
# Cached worker (used by Parallel)
# ---------------------------------------------------------------------------

def _process_video(video_dir: str, vid: str, cache_dir: str):
    """Extract features for one video, using a per-video cache file.

    Returns:
        (video_id, feats) on success, or None if extraction fails.
    """
    video_id   = os.path.splitext(vid)[0]
    cache_file = os.path.join(cache_dir, f"{video_id}.pkl")

    if os.path.exists(cache_file):
        return video_id, joblib.load(cache_file)

    vid_path = os.path.join(video_dir, vid)
    feats    = extract_features_from_video(vid_path)

    if feats is not None:
        joblib.dump(feats, cache_file)
        return video_id, feats

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # --- Collect video files ---
    ext = (".mp4", ".avi", ".mov", ".mkv")
    videos = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(ext)]
    print(f"Found {len(videos)} videos in {VIDEO_DIR}")

    # --- Parallel extraction ---
    Parallel(n_jobs=N_JOBS)(
        delayed(_process_video)(VIDEO_DIR, vid, CACHE_DIR)
        for vid in tqdm(videos, desc="Extracting features")
    )

    # --- Load all cached results ---
    ids, cached = [], []
    for f in sorted(os.listdir(CACHE_DIR)):
        if f.endswith(".pkl"):
            vid_id = os.path.splitext(f)[0]
            feats  = joblib.load(os.path.join(CACHE_DIR, f))
            ids.append(vid_id)
            cached.append(feats)

    df = pd.DataFrame(cached, index=ids)
    print(f"Loaded {len(df)} cached feature vectors with {df.shape[1]} features.")

    # --- Align labels ---
    labels = pd.read_csv(LABEL_FILE, index_col=0)["mos"]
    labels.index = (
        labels.index.astype(str)
               .str.replace(r"\.(mp4|avi|mov|mkv)$", "", regex=True)
    )
    df.index = df.index.astype(str)
    labels   = labels.reindex(df.index)

    # --- Drop videos with missing labels ---
    missing = labels.isna()
    if missing.any():
        print(f"\n⚠️  {missing.sum()} videos have no labels and will be dropped:")
        for vid in df.index[missing]:
            print(f"   {vid}")
        df     = df.loc[~missing]
        labels = labels.dropna()

    # --- Save ---
    joblib.dump((df, labels), OUT_FILE)
    print(f"\n✅  Saved {df.shape[0]} samples × {df.shape[1]} features → {OUT_FILE}")


if __name__ == "__main__":
    main()
