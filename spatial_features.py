"""
src/spatial_features.py
-----------------------
Per-frame spatial quality features for video quality assessment.

Features
--------
- sharpness_stats  : Laplacian variance (blur detection)
- niqe_stats       : NIQE perceptual quality score
- canny_edge_stats : Canny edge density
- lbp_stats        : Local Binary Pattern texture variance
"""

import cv2
import numpy as np
from skimage.feature import canny, local_binary_pattern

from niqe_master.niqe import niqe
from src.utils import agg_basic


# ---------------------------------------------------------------------------
# Individual feature extractors
# ---------------------------------------------------------------------------

def _lap_var(Y: np.ndarray) -> float:
    """Laplacian variance of a luminance frame (sharpness proxy).

    Args:
        Y: Luminance frame in [0, 1], float32.

    Returns:
        Variance of the Laplacian response.
    """
    Y_uint8 = (Y * 255).astype(np.uint8)
    return float(cv2.Laplacian(Y_uint8, cv2.CV_32F).var())


def sharpness_stats(frames: list) -> dict:
    """Aggregate Laplacian-variance sharpness across all frames.

    Args:
        frames: List of (Y, Cb, Cr) tuples, each channel float32 in [0, 1].

    Returns:
        Dict with keys ``sharp_{mean,std,p05,p50,p95,max}``.
    """
    vals = np.array([_lap_var(f[0]) for f in frames])
    return agg_basic(vals, "sharp")


def niqe_stats(frames: list) -> dict:
    """Aggregate NIQE perceptual quality score across all frames.

    Args:
        frames: List of (Y, Cb, Cr) tuples.

    Returns:
        Dict with keys ``niqe_{mean,std,p05,p50,p95,max}``.
    """
    vals = np.array([niqe((f[0] * 255).astype(np.uint8)) for f in frames])
    return agg_basic(vals, "niqe")


def canny_edge_stats(frames: list) -> dict:
    """Aggregate Canny edge density (fraction of edge pixels) across frames.

    Args:
        frames: List of (Y, Cb, Cr) tuples.

    Returns:
        Dict with keys ``canny_edges_{mean,std,p05,p50,p95,max}``.
    """
    vals = []
    for f in frames:
        Y = (f[0] * 255).astype(np.uint8)
        edges = canny(Y, sigma=1.5)
        vals.append(float(edges.mean()))
    return agg_basic(np.array(vals), "canny_edges")


def lbp_stats(frames: list) -> dict:
    """Aggregate LBP texture variance across all frames.

    Uses uniform LBP with P=8 neighbours at radius 1.

    Args:
        frames: List of (Y, Cb, Cr) tuples.

    Returns:
        Dict with keys ``lbp_{mean,std,p05,p50,p95,max}``.
    """
    vals = []
    for f in frames:
        Y = (f[0] * 255).astype(np.uint8)
        lbp_img = local_binary_pattern(Y, P=8, R=1, method="uniform")
        vals.append(float(lbp_img.var()))
    return agg_basic(np.array(vals), "lbp")
