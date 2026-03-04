"""
src/utils.py
------------
Shared utility functions used across all feature modules.
"""

import numpy as np


def agg_basic(arr: np.ndarray, prefix: str) -> dict:
    """Aggregate a 1-D array into summary statistics.

    Returns mean, std, 5th percentile, median, 95th percentile, and max,
    all keyed as ``{prefix}_{stat}``.

    If the array is empty, returns ``{prefix}_nan: 1.0`` as a sentinel.

    Args:
        arr:    1-D numpy array of values.
        prefix: String prefix for the output keys.

    Returns:
        Dict of six float-valued statistics (or one sentinel key).

    Example:
        >>> agg_basic(np.array([1.0, 2.0, 3.0]), "sharp")
        {'sharp_mean': 2.0, 'sharp_std': 0.816..., ...}
    """
    if len(arr) == 0:
        return {f"{prefix}_nan": 1.0}

    arr = np.asarray(arr, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std":  float(np.std(arr)),
        f"{prefix}_p05":  float(np.percentile(arr, 5)),
        f"{prefix}_p50":  float(np.percentile(arr, 50)),
        f"{prefix}_p95":  float(np.percentile(arr, 95)),
        f"{prefix}_max":  float(np.max(arr)),
    }
