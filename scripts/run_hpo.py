"""
run_hpo.py
----------
Hyperparameter Optimization using Optuna with Event-Based Splits.

Fixes the issues in archive/old_scripts/run_optimization.py:
1. Evaluates trials on VALIDATION set (held-out test set evaluated only once at the end).
2. Uses event-based splitting (GroupShuffleSplit) to ensure no event leakage.
3. Fixes tensor/device conversion bugs (using .cpu().numpy() before scikit-learn).
4. Accurately reports metrics and avoids returning -Infinity unless a trial actually diverges.
"""

import os
import sys
import io
import copy
import time
import numpy as np
import pandas as pd
import torch
import optuna

# Force UTF-8 stdout
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.data_loader import load_dataset
from src.preprocess import (
    handle_missing_values,
    split_data_three_way,
    scale_features,
    scale_target,
    to_tensors,
)
from src.feature_audit import detect_degenerate_features, analyze_correlations
from src.pinn_model import OptimizedPINN
from src.train import train_optimized
from src.evaluate import compute_metrics, compute_mean_baseline


def run_event_hpo(
    data_path: str = "data/earthquake_29_features.csv",
    n_trials: int = 15,
    epochs_per_trial: int = 80,
    output_dir: str = "outputs/reports",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HPO] Starting Optuna HPO ({n_trials} trials, {epochs_per_trial} epochs/trial) on {device}")

    # 1. Load data with event groups
    X_raw, y_raw, groups_raw = load_dataset(
        csv_path=data_path,
        target_column="magnitude",
        drop_columns=["filename"],
        return_groups=True,
    )
    X_raw = handle_missing_values(X_raw)
    y_raw = y_raw.loc[X_raw.index]
    groups_raw = groups_raw.loc[X_raw.index]

    # 2. Audit
    degen_removed, _ = detect_degenerate_features(X_raw)
    X_audit = X_raw.drop(columns=degen_removed)
    corr_removed, _ = analyze_correlations(X_audit)
    X_clean = X_audit.drop(columns=corr_removed)
    feature_names = X_clean.columns.tolist()

    # 3. Event-based split
    X_tr_df, X_va_df, X_te_df, y_tr_s, y_va_s, y_te_s = split_data_three_way(
        X_clean, y_raw,
        groups=groups_raw.loc[X_clean.index],
        val_size=0.15,
        test_size=0.15,
        random_state=42,
        split_strategy="event",
    )

    # 4. Scale features & target on train only
    X_tr_s, X_va_s, X_te_s, _ = scale_features(X_tr_df, X_va_df, X_te_df)
    y_tr_sc, y_va_sc, y_te_sc, tgt_scaler = scale_target(y_tr_s, y_va_s, y_te_s)

    # Convert to tensors
    X_tr_t, X_va_t, X_te_t, y_tr_t, y_va_t, y_te_t = to_tensors(
        X_tr_s, X_va_s, X_te_s, y_tr_sc, y_va_sc, y_te_sc, device=device
    )

    energy_col = "energy"
    freq_col = "spectral_centroid"
    raw_e_train = torch.tensor(
        X_tr_df[energy_col].values if energy_col in X_tr_df.columns else np.zeros(len(X_tr_df)),
        dtype=torch.float32, device=device,
    )
    raw_f_train = torch.tensor(
        X_tr_df[freq_col].values if freq_col in X_tr_df.columns else np.zeros(len(X_tr_df)),
        dtype=torch.float32, device=device,
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 5e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        dropout = trial.suggest_float("dropout", 0.05, 0.25)
        hidden_choice = trial.suggest_categorical(
            "hidden_layers",
            ["[64, 32, 16]", "[128, 64, 32]", "[64, 64]"]
        )
        hidden_layers = eval(hidden_choice)
        lambda_energy = trial.suggest_float("lambda_energy", 0.01, 0.2, log=True)

        m_cfg = {
            "input_dim": len(feature_names),
            "hidden_layers": hidden_layers,
            "activation": "gelu",
            "dropout": dropout,
            "use_layer_norm": True,
            "use_batch_norm": False,
            "use_residual": True,
            "use_attention": True,
            "use_uncertainty": False,
        }

        t_cfg = {
            "epochs": epochs_per_trial,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "grad_clip": 1.0,
            "patience": 20,
            "print_every": 9999,
            "scheduler": "plateau",
            "use_physics": True,
            "physics_experiment": "energy",
            "energy_mode": "correlation",
            "lambda_energy": lambda_energy,
            "lambda_frequency": 0.0,
        }

        try:
            model = OptimizedPINN(m_cfg).to(device)
            model, _ = train_optimized(
                model=model,
                X_train=X_tr_t, y_train=y_tr_t,
                X_val=X_va_t, y_val=y_va_t,
                raw_energy_train=raw_e_train,
                raw_frequency_train=raw_f_train,
                config=t_cfg,
                model_name=f"trial_{trial.number}",
                device=device,
            )

            model.eval()
            with torch.no_grad():
                val_pred_s = model(X_va_t).cpu().numpy().flatten()

            y_va_pred = tgt_scaler.inverse_transform(val_pred_s.reshape(-1, 1)).flatten()
            m_val = compute_metrics(y_va_s.values, y_va_pred)
            r2 = m_val["R2"]
            if np.isnan(r2):
                return -10.0
            return float(r2)

        except Exception as e:
            print(f"[HPO] Trial {trial.number} failed with error: {e}")
            return -10.0

    study = optuna.create_study(direction="maximize", study_name="pinn_event_hpo")
    study.optimize(objective, n_trials=n_trials)

    print("\n" + "=" * 60)
    print("  HPO COMPLETED")
    print("=" * 60)
    print(f"  Best Trial: #{study.best_trial.number}")
    print(f"  Best Val R²: {study.best_value:.4f}")
    print("  Best Parameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Retrain best config and evaluate on test set once
    best = study.best_params
    best_m_cfg = {
        "input_dim": len(feature_names),
        "hidden_layers": eval(best["hidden_layers"]),
        "activation": "gelu",
        "dropout": best["dropout"],
        "use_layer_norm": True,
        "use_batch_norm": False,
        "use_residual": True,
        "use_attention": True,
        "use_uncertainty": False,
    }
    best_t_cfg = {
        "epochs": epochs_per_trial,
        "learning_rate": best["lr"],
        "weight_decay": best["weight_decay"],
        "batch_size": best["batch_size"],
        "grad_clip": 1.0,
        "patience": 30,
        "print_every": 9999,
        "scheduler": "plateau",
        "use_physics": True,
        "physics_experiment": "energy",
        "energy_mode": "correlation",
        "lambda_energy": best["lambda_energy"],
        "lambda_frequency": 0.0,
    }

    final_model = OptimizedPINN(best_m_cfg).to(device)
    final_model, _ = train_optimized(
        model=final_model,
        X_train=X_tr_t, y_train=y_tr_t,
        X_val=X_va_t, y_val=y_va_t,
        raw_energy_train=raw_e_train,
        raw_frequency_train=raw_f_train,
        config=best_t_cfg,
        model_name="best_hpo_model",
        device=device,
    )
    final_model.eval()
    with torch.no_grad():
        test_pred_s = final_model(X_te_t).cpu().numpy().flatten()
    y_te_pred = tgt_scaler.inverse_transform(test_pred_s.reshape(-1, 1)).flatten()
    test_metrics = compute_metrics(y_te_s.values, y_te_pred)
    print(f"  Final Test Metrics for Best Trial: R² = {test_metrics['R2']:.4f}, MAE = {test_metrics['MAE']:.4f}, RMSE = {test_metrics['RMSE']:.4f}")
    print("=" * 60 + "\n")

    return {
        "best_trial": study.best_trial.number,
        "best_val_r2": study.best_value,
        "best_params": study.best_params,
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    run_event_hpo()
