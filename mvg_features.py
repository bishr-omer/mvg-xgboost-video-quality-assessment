"""
src/mvg_features.py
-------------------
Multivariate Gaussian (MVG) quality features for video quality assessment.

Builds a per-frame feature matrix combining spatial and temporal cues, then
fits an MVG model and extracts:
  - Mahalanobis distance statistics (deviation from mean frame quality)
  - Covariance eigenvalue statistics (feature spread / correlation structure)

Functions
---------
- build_per_frame_vectors : Assemble [T x D] feature matrix from raw frames + flows
- mvg_stats               : Fit MVG model and return distortion statistics
"""

import numpy as np
from skimage.feature import canny, local_binary_pattern

from niqe_master.niqe import niqe
from src.utils import agg_basic


# ---------------------------------------------------------------------------
# Per-frame feature vector construction
# ---------------------------------------------------------------------------

def build_per_frame_vectors(frames: list, flows: list) -> np.ndarray:
    """Build a [T x 4] per-frame feature matrix for MVG fitting.

    For each frame, extracts:
      - NIQE score            (perceptual quality)
      - Canny edge density    (sharpness / detail)
      - LBP variance          (texture richness)
      - Optical flow magnitude (temporal motion)

    Flow magnitudes are padded with 0.0 for the first frame.

    Args:
        frames: List of (Y, Cb, Cr) tuples, Y float32 in [0, 1].
        flows:  Forward flow list from
                :func:`~src.temporal_features.optical_flow_pairs`,
                length = len(frames) - 1.

    Returns:
        numpy array of shape [len(frames) x 4].
    """
    # Pad flow magnitudes so index aligns with frames
    flow_mags = [0.0] + [float(np.linalg.norm(F, axis=2).mean()) for F in flows]

    rows = []
    for i, frame in enumerate(frames):
        Y       = frame[0]
        Y_uint8 = (Y * 255).astype(np.uint8)

        niqe_val   = float(niqe(Y_uint8))
        canny_val  = float(canny(Y_uint8, sigma=1.5).mean())
        lbp_val    = float(local_binary_pattern(Y_uint8, P=8, R=1, method="uniform").var())
        flow_val   = flow_mags[i]

        rows.append([niqe_val, canny_val, lbp_val, flow_val])

    return np.array(rows, dtype=np.float64)   # [T x 4]


# ---------------------------------------------------------------------------
# MVG model fitting and feature extraction
# ---------------------------------------------------------------------------

def mvg_stats(Fs: np.ndarray, lam: float = 0.1) -> tuple:
    """Fit a Multivariate Gaussian model to per-frame feature vectors.

    Computes:
      - Mahalanobis distance of each frame from the mean (aggregated to stats)
      - Eigenvalue statistics of the regularised covariance matrix

    Args:
        Fs:  Per-frame feature matrix [T x D] from
             :func:`build_per_frame_vectors`.
        lam: Tikhonov regularisation coefficient added to the diagonal of the
             covariance matrix to prevent singularity. Default: 0.1.

    Returns:
        Tuple of:
          - feats (dict): MVG feature dict with keys
              ``mahala_{mean,std,p05,p50,p95,max}``,
              ``mvg_eigval_{mean,std,max}``.
          - mu (np.ndarray or None): Mean vector [D].
          - C  (np.ndarray or None): Regularised covariance [D x D].

    Notes:
        Returns ``{"mvg_nan": 1.0}`` and (None, None) if fewer than 2 frames
        are available.
    """
    if len(Fs) < 2:
        return {"mvg_nan": 1.0}, None, None

    # Sanitise input
    Fs = np.nan_to_num(Fs, nan=0.0, posinf=1e6, neginf=-1e6)

    mu = np.mean(Fs, axis=0)                        # [D]
    C  = np.cov(Fs.T) + lam * np.eye(Fs.shape[1])  # [D x D], regularised

    feats = {}

    # --- Mahalanobis distances ---
    try:
        inv_C  = np.linalg.inv(C)
        delta  = Fs - mu                            # [T x D]
        mahala = np.sum(delta @ inv_C * delta, axis=1)  # [T]
        mahala = np.sqrt(np.maximum(mahala, 0.0))   # non-negative sqrt
        feats.update(agg_basic(mahala, "mahala"))
    except np.linalg.LinAlgError:
        # Singular matrix — skip Mahalanobis, keep eigenvalue features
        feats["mahala_nan"] = 1.0

    # --- Covariance eigenvalue statistics ---
    eigvals = np.linalg.eigvalsh(C)
    feats["mvg_eigval_mean"] = float(np.mean(eigvals))
    feats["mvg_eigval_std"]  = float(np.std(eigvals))
    feats["mvg_eigval_max"]  = float(np.max(eigvals))

    return feats, mu, C
