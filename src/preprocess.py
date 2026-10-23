"""
preprocess.py
-------------
Data preprocessing for the earthquake PINN pipeline.

Design guarantees:
    - 3-way split: train (70%) / val (15%) / test (15%).
      The test set is held out until final evaluation only.
    - Feature scaler fitted ONLY on training data (no leakage).
    - Target scaler fitted ONLY on training data (no leakage).
    - All reported metrics use inverse-transformed predictions so that
      MAE / RMSE are in the original magnitude scale.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Missing-value utilities  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Check and report missing values per column."""
    missing = df.isnull().sum()
    total   = missing.sum()
    if total > 0:
        print(f"[Preprocess] WARNING: {total} missing values found:")
        print(missing[missing > 0])
    else:
        print("[Preprocess] No missing values detected.")
    return missing


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows containing any NaN value."""
    before = len(df)
    df_out = df.dropna()
    after  = len(df_out)
    if before != after:
        print(f"[Preprocess] Dropped {before - after} rows with missing values.")
    else:
        print("[Preprocess] No rows dropped (no missing values).")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# 3-way train / val / test split
# ─────────────────────────────────────────────────────────────────────────────

def split_data_three_way(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series] = None,
    val_size: float  = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    split_strategy: str = "event",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,    pd.Series,    pd.Series]:
    """Split data into three independent sets.

    Supports:
        - "event": GroupShuffleSplit by event_id so recordings from the same
          earthquake event NEVER appear in multiple splits (prevents data leakage).
        - "random": Unstratified random row split (LEAKY: recordings of the same
          earthquake event land in both train and test).

    Args:
        X:              Feature DataFrame.
        y:              Target Series.
        groups:         Event identifiers (required for "event" strategy).
        val_size:       Fraction of data/groups allocated to validation.
        test_size:      Fraction of data/groups allocated to test.
        random_state:   Reproducibility seed.
        split_strategy: "event" (default, valid) | "random" (leaky comparator).

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    total = len(X)

    if split_strategy == "event":
        if groups is None:
            raise ValueError(
                "[Preprocess] split_strategy='event' requires 'groups' (event IDs)."
            )
        print(f"\n[Preprocess] 3-Way Split [STRATEGY: EVENT-BASED (no station/event leakage)] (N = {total}):")

        # Step 1: Carve out test events using GroupShuffleSplit
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_val_idx, test_idx = next(gss_test.split(X, y, groups))

        # Step 2: Carve out val events from train_val pool
        val_frac = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state + 1)
        tr_sub, val_sub = next(gss_val.split(
            X.iloc[train_val_idx], y.iloc[train_val_idx], groups.iloc[train_val_idx]
        ))
        train_idx = train_val_idx[tr_sub]
        val_idx   = train_val_idx[val_sub]

        X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
        y_train, y_val, y_test = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]

        train_ev = sorted(groups.iloc[train_idx].unique())
        val_ev   = sorted(groups.iloc[val_idx].unique())
        test_ev  = sorted(groups.iloc[test_idx].unique())

        print(f"  Train : {len(X_train):>5} rows ({len(X_train)/total*100:.1f}%) | {len(train_ev)} events: {train_ev}")
        print(f"  Val   : {len(X_val):>5} rows ({len(X_val)/total*100:.1f}%) | {len(val_ev)} events: {val_ev}")
        print(f"  Test  : {len(X_test):>5} rows ({len(X_test)/total*100:.1f}%) | {len(test_ev)} events: {test_ev}  ← held out")
    else:
        print(f"\n[Preprocess] 3-Way Split [STRATEGY: RANDOM - ⚠️ LEAKY (recordings of same event in train and test)] (N = {total}):")
        # Step 1: carve out test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        # Step 2: split val from remainder
        val_frac = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_frac,
            random_state=random_state + 1,
            shuffle=True,
        )

        print(f"  Train  : {len(X_train):>5}  ({len(X_train)/total*100:.1f}%) [LEAKY]")
        print(f"  Val    : {len(X_val):>5}  ({len(X_val)/total*100:.1f}%) [LEAKY]")
        print(f"  Test   : {len(X_test):>5}  ({len(X_test)/total*100:.1f}%) [LEAKY] ← held out")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Feature scaling (fit on train only)
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(
    X_train,
    X_val,
    X_test,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Standardise features (scaler fitted ONLY on training data).

    Accepts DataFrames or ndarrays for X_train / X_val / X_test.

    Returns:
        (X_train_scaled, X_val_scaled, X_test_scaled, fitted_scaler)
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    print("[Preprocess] Feature scaler fitted on train → applied to val + test (no leakage).")
    return X_train_s, X_val_s, X_test_s, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Target scaling (fit on train only)
# ─────────────────────────────────────────────────────────────────────────────

def scale_target(
    y_train,
    y_val,
    y_test,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Standardise the magnitude target (scaler fitted ONLY on training data).

    IMPORTANT: always inverse-transform predictions before computing MAE /
    RMSE / R² so that metrics are reported in the original magnitude scale.

    Returns:
        (y_train_scaled, y_val_scaled, y_test_scaled, target_scaler)
    """
    def _arr(y):
        if isinstance(y, pd.Series):
            return y.values
        return np.asarray(y)

    tgt_scaler = StandardScaler()
    y_tr_s = tgt_scaler.fit_transform(_arr(y_train).reshape(-1, 1)).flatten()
    y_va_s = tgt_scaler.transform(_arr(y_val).reshape(-1, 1)).flatten()
    y_te_s = tgt_scaler.transform(_arr(y_test).reshape(-1, 1)).flatten()

    print(
        f"[Preprocess] Target scaler fitted on train. "
        f"Scaled range [{y_tr_s.min():.3f}, {y_tr_s.max():.3f}]  "
        f"(inverse-transform applied before metric reporting)."
    )
    return y_tr_s, y_va_s, y_te_s, tgt_scaler


# ─────────────────────────────────────────────────────────────────────────────
# Tensor conversion
# ─────────────────────────────────────────────────────────────────────────────

def to_tensors(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
    y_train,
    y_val,
    y_test,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert numpy arrays to float32 tensors and move to device.

    Args:
        X_train, X_val, X_test: Scaled feature arrays.
        y_train, y_val, y_test: Target arrays (scaled or raw).
        device:                 Target torch device.

    Returns:
        Six tensors: X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t
    """
    def _arr(y):
        if isinstance(y, pd.Series):
            return y.values
        return np.asarray(y)

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_va = torch.tensor(X_val,   dtype=torch.float32, device=device)
    X_te = torch.tensor(X_test,  dtype=torch.float32, device=device)
    y_tr = torch.tensor(_arr(y_train).flatten(), dtype=torch.float32, device=device).view(-1, 1)
    y_va = torch.tensor(_arr(y_val).flatten(),   dtype=torch.float32, device=device).view(-1, 1)
    y_te = torch.tensor(_arr(y_test).flatten(),  dtype=torch.float32, device=device).view(-1, 1)

    print(
        f"[Preprocess] Tensors on {device}: "
        f"train={X_tr.shape}, val={X_va.shape}, test={X_te.shape}"
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te
