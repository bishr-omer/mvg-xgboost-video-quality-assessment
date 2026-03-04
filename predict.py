"""
predict.py
----------
Run the trained MVG-VQA model on one or more new video files.

Usage
-----
    # Single video
    python predict.py path/to/video.mp4

    # Directory of videos
    python predict.py path/to/videos/

    # Custom model path
    python predict.py video.mp4 --model models/xgb_mvg_final.pkl

Outputs
-------
    Prints predicted MOS to stdout. If a directory is given, prints a table
    sorted by predicted quality (best first).
"""

import argparse
import os
import sys
import joblib
import numpy as np
import xgboost as xgb

from extract_features import extract_features_from_video


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_single(video_path: str, bundle: dict) -> float | None:
    """Predict MOS for a single video file.

    Args:
        video_path: Path to the video.
        bundle:     Dict with keys ``model``, ``scaler``, ``columns``.

    Returns:
        Predicted MOS as float, or None if feature extraction fails.
    """
    feats = extract_features_from_video(video_path)
    if feats is None:
        return None

    import pandas as pd
    df = pd.DataFrame([feats])[bundle["columns"]]
    df = df.fillna(0.0)

    X_scaled = bundle["scaler"].transform(df.values)
    dmat     = xgb.DMatrix(X_scaled)
    pred     = bundle["model"].predict(dmat)
    return float(pred[0])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MVG-VQA inference")
    parser.add_argument("input",  help="Video file or directory of videos")
    parser.add_argument("--model", default="models/xgb_mvg_final.pkl",
                        help="Path to saved model bundle (default: models/xgb_mvg_final.pkl)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"❌ Model not found: {args.model}\nRun train.py first.")

    bundle = joblib.load(args.model)

    # Collect video paths
    ext = (".mp4", ".avi", ".mov", ".mkv")
    if os.path.isdir(args.input):
        paths = [
            os.path.join(args.input, f)
            for f in sorted(os.listdir(args.input))
            if f.lower().endswith(ext)
        ]
        if not paths:
            sys.exit(f"No video files found in {args.input}")
    elif os.path.isfile(args.input):
        paths = [args.input]
    else:
        sys.exit(f"❌ Input not found: {args.input}")

    # Run predictions
    results = []
    for path in paths:
        mos = predict_single(path, bundle)
        name = os.path.basename(path)
        if mos is None:
            print(f"  ⚠️  {name:40s}  — extraction failed (skipped)")
        else:
            results.append((name, mos))

    if not results:
        return

    # Print sorted table
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'Video':<45} {'Predicted MOS':>14}")
    print("-" * 61)
    for name, mos in results:
        print(f"  {name:<43} {mos:>14.4f}")
    print("-" * 61)
    avg = np.mean([r[1] for r in results])
    print(f"  {'Average':<43} {avg:>14.4f}\n")


if __name__ == "__main__":
    main()
