"""
train.py
--------
XGBoost model training with randomised hyperparameter search for MVG-VQA.

Pipeline
--------
1. Load pre-extracted features from ``data/features.pkl``
2. Apply feature selection from ``data/selected_features.pkl``
3. Standardise features with StandardScaler
4. Randomised search over XGBoost hyperparameters using stratified K-Fold CV
5. Retrain the best model on the full dataset
6. Save model + scaler + feature columns to ``models/xgb_mvg_final.pkl``

Usage
-----
    python train.py

Outputs
-------
    models/xgb_mvg_final.pkl  — dict with keys: model, scaler, columns
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
from tqdm import trange


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEAT_FILE     = "data/features.pkl"
SEL_FEAT_FILE = "data/selected_features.pkl"
MODEL_OUT     = "models/xgb_mvg_final.pkl"

N_SPLITS      = 5      # K-Fold splits
N_BINS        = 10     # Bins for stratified-like split
N_SEARCH      = 200    # Randomised search iterations
RANDOM_STATE  = 42

PARAM_DIST = {
    "n_estimators":    [100, 300, 500],
    "max_depth":       [3, 5, 7],
    "learning_rate":   [0.01, 0.05, 0.1],
    "subsample":       [0.6, 0.8, 1.0],
    "colsample_bytree":[0.6, 0.8, 1.0],
    "reg_lambda":      [0.5, 1.0, 2.0],
    "gamma":           [0, 0.1, 0.3],
}

XGB_BASE_PARAMS = {
    "objective":   "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
}


# ---------------------------------------------------------------------------
# Stratified K-Fold for regression
# ---------------------------------------------------------------------------

def stratified_kfold_regression(
    y: pd.Series,
    n_splits: int = N_SPLITS,
    n_bins: int = N_BINS,
) -> list:
    """Generate stratified K-Fold splits for a continuous target.

    Bins the target into quantiles, then uses KFold on the bin labels to
    ensure each fold has a representative MOS distribution.

    Args:
        y:        Target MOS values.
        n_splits: Number of folds.
        n_bins:   Number of quantile bins.

    Returns:
        List of (train_idx, val_idx) index arrays.
    """
    y_binned = pd.qcut(y, q=n_bins, duplicates="drop").cat.codes
    kf       = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(kf.split(np.zeros(len(y)), y_binned))


# ---------------------------------------------------------------------------
# Cross-validated evaluation
# ---------------------------------------------------------------------------

def fit_with_metrics(
    params: dict,
    X: np.ndarray,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    n_bins:   int = N_BINS,
) -> tuple[float, float, float, float]:
    """Train and evaluate XGBoost with K-Fold CV, returning mean metrics.

    Args:
        params:   XGBoost hyperparameters (including ``n_estimators``).
        X:        Feature matrix [N x D], already scaled.
        y:        MOS target values.
        n_splits: Number of CV folds.
        n_bins:   Quantile bins for stratified splitting.

    Returns:
        Tuple of (mean_SRCC, mean_PLCC, mean_RMSE, mean_MAE) across folds.
    """
    splits = stratified_kfold_regression(y, n_splits=n_splits, n_bins=n_bins)
    srccs, plccs, rmses, maes = [], [], [], []

    for tr_idx, va_idx in splits:
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval   = xgb.DMatrix(X_va, label=y_va)

        xgb_params     = XGB_BASE_PARAMS.copy()
        fold_params     = params.copy()
        n_estimators    = int(fold_params.pop("n_estimators", 200))
        xgb_params.update(fold_params)

        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, "validation")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        pred = model.predict(dval)

        srccs.append(spearmanr(y_va, pred).statistic)
        plccs.append(float(np.corrcoef(y_va, pred)[0, 1]))
        rmses.append(float(np.sqrt(mean_squared_error(y_va, pred))))
        maes.append(float(mean_absolute_error(y_va, pred)))

    return (
        float(np.mean(srccs)),
        float(np.mean(plccs)),
        float(np.mean(rmses)),
        float(np.mean(maes)),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Load data ---
    print(f"Loading features from {FEAT_FILE} ...")
    X_full, y = joblib.load(FEAT_FILE)

    selected = joblib.load(SEL_FEAT_FILE)
    X = X_full[selected]
    print(f"  {X.shape[0]} samples, {X.shape[1]} selected features.")

    # --- Scale ---
    scaler   = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # --- Hyperparameter search ---
    rng        = np.random.default_rng(RANDOM_STATE)
    best_srcc  = -np.inf
    best_params = None

    print(f"\nStarting randomised search ({N_SEARCH} iterations, {N_SPLITS}-fold CV) ...")
    with trange(N_SEARCH, desc="Hyperparam search") as t:
        for _ in t:
            params = {k: rng.choice(v) for k, v in PARAM_DIST.items()}
            srcc, plcc, rmse, mae = fit_with_metrics(params, X_scaled, y)

            t.set_postfix(SRCC=f"{srcc:.3f}", PLCC=f"{plcc:.3f}",
                          RMSE=f"{rmse:.3f}", MAE=f"{mae:.3f}")

            if srcc > best_srcc:
                best_srcc   = srcc
                best_params = params.copy()

    print(f"\nBest CV SRCC : {best_srcc:.4f}")
    print(f"Best params  : {best_params}")

    # --- Final model on full dataset ---
    print("\nTraining final model on full dataset ...")
    final_params = best_params.copy()
    n_estimators = int(final_params.pop("n_estimators", 200))

    xgb_params = {**XGB_BASE_PARAMS, **final_params}
    dall       = xgb.DMatrix(X_scaled, label=y)

    final_model = xgb.train(xgb_params, dall, num_boost_round=n_estimators)

    # --- Save ---
    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"model": final_model, "scaler": scaler, "columns": X.columns.tolist()},
        MODEL_OUT,
    )
    print(f"✅  Saved → {MODEL_OUT}")

    # --- Full-dataset evaluation (optimistic, for sanity check) ---
    y_pred = final_model.predict(dall)
    srcc   = spearmanr(y, y_pred).correlation
    plcc   = pearsonr(y, y_pred).statistic
    rmse   = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae    = float(mean_absolute_error(y, y_pred))

    print("\nFull-dataset performance (train set — not a held-out estimate):")
    print(f"  SRCC : {srcc:.4f}")
    print(f"  PLCC : {plcc:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")


if __name__ == "__main__":
    main()
