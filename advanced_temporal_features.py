"""
src/advanced_temporal_features.py
----------------------------------
Additional temporal quality features for video quality assessment.

Features
--------
- optical_flow_variance : Spatial variance of flow magnitude per frame pair
- zero_flow_stats       : Fraction of near-zero flow vectors (static regions)
- frame_difference_stats: Mean squared luminance difference between frames
- fft_flicker           : Spectral energy in a temporal flicker frequency band
- blockiness_stats      : Blocking artifact strength per frame
"""

import numpy as np

from src.utils import agg_basic


# ---------------------------------------------------------------------------
# Optical flow variance
# ---------------------------------------------------------------------------

def optical_flow_variance(flows: list) -> dict:
    """Spatial variance of flow magnitude for each frame pair.

    High variance indicates non-uniform motion, which can expose compression
    or temporal inconsistency artefacts.

    Args:
        flows: List of flow arrays [H x W x 2] from
               :func:`~src.temporal_features.optical_flow_pairs`.

    Returns:
        Dict with keys ``flow_var_{mean,std,p05,p50,p95,max}``.
    """
    vals = np.array([np.linalg.norm(F, axis=2).var() for F in flows])
    return agg_basic(vals, "flow_var")


# ---------------------------------------------------------------------------
# Zero-flow (static region) statistics
# ---------------------------------------------------------------------------

def zero_flow_stats(flows: list, threshold: float = 0.5) -> dict:
    """Fraction of flow vectors below a magnitude threshold (static pixels).

    Args:
        flows:     List of flow arrays [H x W x 2].
        threshold: Magnitude below which a vector is considered static.
                   Default: 0.5 px.

    Returns:
        Dict with keys ``zero_flow_{mean,std,p05,p50,p95,max}``.
    """
    vals = []
    for F in flows:
        mag = np.linalg.norm(F, axis=2)
        vals.append(float((mag < threshold).mean()))
    return agg_basic(np.array(vals), "zero_flow")


# ---------------------------------------------------------------------------
# Frame difference
# ---------------------------------------------------------------------------

def frame_difference_stats(frames: list) -> dict:
    """Mean squared luminance difference between consecutive frames (flicker).

    Args:
        frames: List of (Y, Cb, Cr) tuples, Y float32 in [0, 1].

    Returns:
        Dict with keys ``flicker_{mean,std,p05,p50,p95,max}``.
    """
    Ys   = [f[0] for f in frames]
    vals = np.array([
        float(np.mean((Ys[t] - Ys[t - 1]) ** 2))
        for t in range(1, len(Ys))
    ])
    return agg_basic(vals, "flicker")


# ---------------------------------------------------------------------------
# FFT flicker
# ---------------------------------------------------------------------------

def fft_flicker(frames: list, fps: float = 30.0, band: tuple = (1, 3)) -> dict:
    """Spectral energy ratio in a temporal frequency band (flicker detection).

    Computes the 1-D FFT of the mean luminance signal over time, and returns
    the fraction of power in ``band`` Hz relative to total power.

    Args:
        frames: List of (Y, Cb, Cr) tuples.
        fps:    Frame rate of the video. Default: 30.
        band:   (low_hz, high_hz) frequency band of interest. Default: (1, 3).

    Returns:
        Dict with single key ``fft_flicker_ratio``.
    """
    ys    = np.array([f[0].mean() for f in frames])
    ys   -= ys.mean()
    P     = np.abs(np.fft.rfft(ys)) ** 2
    freqs = np.fft.rfftfreq(len(ys), 1.0 / max(fps, 1.0))

    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    ratio     = P[band_mask].sum() / (P.sum() + 1e-8)

    return {"fft_flicker_ratio": float(ratio)}


# ---------------------------------------------------------------------------
# Blockiness
# ---------------------------------------------------------------------------

def _blockiness(Y: np.ndarray, block: int = 8) -> float:
    """Blocking artifact strength for a single luminance frame.

    Measures the difference between block-boundary gradients and
    interior gradients — positive values indicate blocking.

    Args:
        Y:     Luminance frame, float32 in [0, 1].
        block: DCT block size. Default: 8.

    Returns:
        Scalar blockiness score.
    """
    H, W = Y.shape

    # Vertical block boundaries vs interior
    a_v = Y[:, block::block]
    b_v = Y[:, block - 1::block]
    m   = min(a_v.shape[1], b_v.shape[1])
    vb  = np.mean(np.abs(a_v[:, :m] - b_v[:, :m]))

    ai  = Y[:, 1:W - 1]
    bi  = Y[:, 0:W - 2]
    m   = min(ai.shape[1], bi.shape[1])
    vi  = np.mean(np.abs(ai[:, :m] - bi[:, :m]))

    # Horizontal block boundaries vs interior
    a_h = Y[block::block, :]
    b_h = Y[block - 1::block, :]
    m   = min(a_h.shape[0], b_h.shape[0])
    hb  = np.mean(np.abs(a_h[:m, :] - b_h[:m, :]))

    ai  = Y[1:H - 1, :]
    bi  = Y[0:H - 2, :]
    m   = min(ai.shape[0], bi.shape[0])
    hi  = np.mean(np.abs(ai[:m, :] - bi[:m, :]))

    return float((vb - vi + hb - hi) / 2.0)


def blockiness_stats(frames: list) -> dict:
    """Aggregate blockiness artifact score across all frames.

    Args:
        frames: List of (Y, Cb, Cr) tuples.

    Returns:
        Dict with keys ``block_{mean,std,p05,p50,p95,max}``.
    """
    vals = np.array([_blockiness(f[0]) for f in frames])
    return agg_basic(vals, "block")
