"""
cross_validation.py
-------------------
K-fold cross-validation for the earthquake PINN.

CV is performed on the training pool (train + val combined = 85% of total
data).  The held-out test set (15%) is never touched inside CV.

Per-fold procedure:
    1. KFold splits the pool into CV-train and CV-val.
    2. Feature scaler fitted on CV-train only (no leakage).
    3. Target scaler fitted on CV-train only.
    4. Fresh model trained and evaluated on CV-val.

Summary: mean ± std of R², MAE, RMSE across all folds.
"""

import copy
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional

from src.pinn_model import OptimizedPINN
from src.train import train_optimized
from src.evaluate import compute_metrics


def run_kfold_cv(
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    raw_energy_pool: np.ndarray,
    raw_frequency_pool: np.ndarray,
    model_config: Dict,
    train_config: Dict,
    groups_pool: Optional[pd.Series] = None,
    split_strategy: str = "event",
    k: int = 5,
    scale_target: bool = True,
    device: Optional[torch.device] = None,
    random_state: int = 42,
) -> Dict:
    """Run cross-validation on the dataset pool.

    Supports:
        - "event": LeaveOneGroupOut (12 folds) so each fold holds out one entire
          earthquake event. Completely eliminates cross-station event leakage.
        - "random": Standard random K-fold (LEAKY comparator).

    Args:
        X_pool:               Features (DataFrame).
        y_pool:               Targets (Series).
        raw_energy_pool:      Raw energy values aligned with X_pool.
        raw_frequency_pool:   Raw spectral centroid values aligned with X_pool.
        model_config:         Model architecture config dict.
        train_config:         Training hyperparameter config dict.
        groups_pool:          Event IDs per row (required for "event" strategy).
        split_strategy:       "event" (default, LeaveOneGroupOut) | "random" (leaky).
        k:                    Number of folds (used only if split_strategy == "random").
        scale_target:         Apply StandardScaler to target (recommended).
        device:               Torch device.
        random_state:         Seed for KFold shuffle.

    Returns:
        Dict with per-fold metrics and cross-fold summary statistics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_arr   = X_pool.values
    y_arr   = y_pool.values.flatten()
    e_arr   = np.asarray(raw_energy_pool).flatten()
    f_arr   = np.asarray(raw_frequency_pool).flatten()

    fold_results: List[Dict] = []

    if split_strategy == "event":
        if groups_pool is None:
            raise ValueError("[CV] split_strategy='event' requires groups_pool (event IDs).")
        groups_arr = groups_pool.values.flatten()
        logo = LeaveOneGroupOut()
        splits = list(logo.split(X_arr, y_arr, groups=groups_arr))
        n_folds = len(splits)
        print(f"\n[CV] Starting Leave-One-Group-Out ({n_folds} folds) Cross-Validation [EVENT-BASED]")
        print("[CV] Each fold evaluates generalization to a completely unseen earthquake event.")
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        splits = list(kf.split(X_arr))
        n_folds = k
        print(f"\n[CV] Starting {k}-fold Cross-Validation [STRATEGY: RANDOM - ⚠️ LEAKY]")
        print("[CV] WARNING: Random fold splitting leaks recordings of the same earthquake into train and val!")

    for fold_idx, (tr_idx, va_idx) in enumerate(splits, 1):
        fold_desc = f"Fold {fold_idx}/{n_folds} (train={len(tr_idx)}, val={len(va_idx)})"
        if split_strategy == "event":
            val_ev = groups_arr[va_idx][0]
            val_mag = y_arr[va_idx][0]
            fold_desc += f" | held-out event={val_ev} (M={val_mag})"
        print(f"\n[CV] ── {fold_desc} ──────────────")

        X_cv_tr, X_cv_va = X_arr[tr_idx], X_arr[va_idx]
        y_cv_tr, y_cv_va = y_arr[tr_idx], y_arr[va_idx]
        e_cv_tr = e_arr[tr_idx]
        f_cv_tr = f_arr[tr_idx]

        # Feature scaler — fit on CV-train only
        feat_scaler = StandardScaler()
        X_cv_tr_s = feat_scaler.fit_transform(X_cv_tr)
        X_cv_va_s = feat_scaler.transform(X_cv_va)

        # Target scaler — fit on CV-train only
        if scale_target:
            tgt_scaler = StandardScaler()
            y_cv_tr_s = tgt_scaler.fit_transform(
                y_cv_tr.reshape(-1, 1)).flatten()
            y_cv_va_s = tgt_scaler.transform(
                y_cv_va.reshape(-1, 1)).flatten()
        else:
            tgt_scaler = None
            y_cv_tr_s = y_cv_tr
            y_cv_va_s = y_cv_va

        # Convert to tensors
        Xt = torch.tensor(X_cv_tr_s, dtype=torch.float32, device=device)
        Xv = torch.tensor(X_cv_va_s, dtype=torch.float32, device=device)
        yt = torch.tensor(y_cv_tr_s, dtype=torch.float32, device=device).view(-1, 1)
        yv = torch.tensor(y_cv_va_s, dtype=torch.float32, device=device).view(-1, 1)
        et = torch.tensor(e_cv_tr,   dtype=torch.float32, device=device)
        ft = torch.tensor(f_cv_tr,   dtype=torch.float32, device=device)

        # Fresh model per fold
        mcfg = copy.deepcopy(model_config)
        mcfg["input_dim"] = X_cv_tr_s.shape[1]
        model = OptimizedPINN(mcfg).to(device)

        model, history = train_optimized(
            model=model,
            X_train=Xt, y_train=yt,
            X_val=Xv,   y_val=yv,
            raw_energy_train=et,
            raw_frequency_train=ft,
            config=train_config,
            model_name=f"CV_Fold{fold_idx}",
            device=device,
        )

        # Evaluate on CV-val
        model.eval()
        with torch.no_grad():
            pred_s = model(Xv).cpu().numpy().flatten()

        if tgt_scaler is not None:
            y_pred = tgt_scaler.inverse_transform(
                pred_s.reshape(-1, 1)).flatten()
            y_true = y_cv_va          # original magnitude scale
        else:
            y_pred, y_true = pred_s, y_cv_va

        m = compute_metrics(y_true, y_pred)
        m["fold"] = fold_idx
        fold_results.append(m)
        print(f"[CV] Fold {fold_idx}: R²={m['R2']:.4f}  "
              f"MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}")

    # Summary statistics
    r2s   = [r["R2"]   for r in fold_results]
    maes  = [r["MAE"]  for r in fold_results]
    rmses = [r["RMSE"] for r in fold_results]

    summary = {
        "fold_results": fold_results,
        "R2_mean":   round(float(np.mean(r2s)),   4),
        "R2_std":    round(float(np.std(r2s)),    4),
        "MAE_mean":  round(float(np.mean(maes)),  4),
        "MAE_std":   round(float(np.std(maes)),   4),
        "RMSE_mean": round(float(np.mean(rmses)), 4),
        "RMSE_std":  round(float(np.std(rmses)),  4),
    }

    print("\n[CV] ── Cross-Validation Summary ─────────────────────────────")
    print(f"  R²   : {summary['R2_mean']:.4f} ± {summary['R2_std']:.4f}")
    print(f"  MAE  : {summary['MAE_mean']:.4f} ± {summary['MAE_std']:.4f}")
    print(f"  RMSE : {summary['RMSE_mean']:.4f} ± {summary['RMSE_std']:.4f}")

    return summary
