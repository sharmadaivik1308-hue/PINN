"""
lambda_sweep.py
---------------
Sensitivity analysis of lambda_energy in the Physics-Informed Neural Network.
Evaluates lambda_energy in {0, 0.001, 0.01, 0.05, 0.1, 0.3} using event-based
splits across 5 random seeds (0 to 4).

Outputs results to outputs/reports/lambda_sweep.csv.
"""

import os
import sys
import io
import copy
import time
import numpy as np
import pandas as pd
import torch

# Ensure UTF-8 stdout on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.data_loader import load_dataset
from src.preprocess import (
    check_missing_values,
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


def run_lambda_sweep(
    data_path: str = "data/earthquake_29_features.csv",
    output_csv: str = "outputs/reports/lambda_sweep.csv",
    lambdas: list = None,
    seeds: list = None,
    epochs: int = 150,
    patience: int = 30,
) -> pd.DataFrame:
    if lambdas is None:
        lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.3]
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LambdaSweep] Running on device: {device}")
    print(f"[LambdaSweep] Lambdas: {lambdas}")
    print(f"[LambdaSweep] Seeds: {seeds}")

    # 1. Load data and extract event groups
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

    # Physics tensors
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

    # Base configs
    base_model_cfg = {
        "input_dim": len(feature_names),
        "hidden_layers": [64, 32, 16],
        "activation": "gelu",
        "dropout": 0.1,
        "use_layer_norm": True,
        "use_batch_norm": False,
        "use_residual": True,
        "use_attention": True,
        "use_uncertainty": False,
    }

    base_train_cfg = {
        "epochs": epochs,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "grad_clip": 1.0,
        "patience": patience,
        "print_every": 9999,  # Quiet
        "scheduler": "plateau",
        "use_physics": True,
        "physics_experiment": "energy",
        "energy_mode": "correlation",
        "lambda_frequency": 0.0,
    }

    baseline = compute_mean_baseline(y_tr_s.values, y_te_s.values)
    print(f"[LambdaSweep] Test Mean Baseline: R² = {baseline['R2']:.4f}, MAE = {baseline['MAE']:.4f}")

    results = []

    for lam in lambdas:
        print(f"\n[LambdaSweep] Evaluating lambda_energy = {lam} across {len(seeds)} seeds...")
        val_r2s, val_maes, val_rmses = [], [], []
        test_r2s, test_maes, test_rmses = [], [], []
        t0 = time.time()

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = OptimizedPINN(base_model_cfg).to(device)
            t_cfg = copy.deepcopy(base_train_cfg)
            t_cfg["lambda_energy"] = lam
            if lam == 0.0:
                t_cfg["use_physics"] = False
                t_cfg["physics_experiment"] = "none"

            model, _ = train_optimized(
                model=model,
                X_train=X_tr_t, y_train=y_tr_t,
                X_val=X_va_t, y_val=y_va_t,
                raw_energy_train=raw_e_train,
                raw_frequency_train=raw_f_train,
                config=t_cfg,
                model_name=f"sweep_lam{lam}_s{seed}",
                device=device,
            )

            # Evaluate on Val
            model.eval()
            with torch.no_grad():
                val_pred_s = model(X_va_t).cpu().numpy().flatten()
                test_pred_s = model(X_te_t).cpu().numpy().flatten()

            y_va_pred = tgt_scaler.inverse_transform(val_pred_s.reshape(-1, 1)).flatten()
            y_te_pred = tgt_scaler.inverse_transform(test_pred_s.reshape(-1, 1)).flatten()

            m_val = compute_metrics(y_va_s.values, y_va_pred)
            m_test = compute_metrics(y_te_s.values, y_te_pred)

            val_r2s.append(m_val["R2"])
            val_maes.append(m_val["MAE"])
            val_rmses.append(m_val["RMSE"])
            test_r2s.append(m_test["R2"])
            test_maes.append(m_test["MAE"])
            test_rmses.append(m_test["RMSE"])

        elapsed = time.time() - t0
        res = {
            "lambda_energy": lam,
            "val_R2_mean": float(np.mean(val_r2s)),
            "val_R2_std": float(np.std(val_r2s)),
            "val_MAE_mean": float(np.mean(val_maes)),
            "val_MAE_std": float(np.std(val_maes)),
            "val_RMSE_mean": float(np.mean(val_rmses)),
            "val_RMSE_std": float(np.std(val_rmses)),
            "test_R2_mean": float(np.mean(test_r2s)),
            "test_R2_std": float(np.std(test_r2s)),
            "test_MAE_mean": float(np.mean(test_maes)),
            "test_MAE_std": float(np.std(test_maes)),
            "test_RMSE_mean": float(np.mean(test_rmses)),
            "test_RMSE_std": float(np.std(test_rmses)),
            "time_seconds": round(elapsed, 1),
        }
        results.append(res)
        print(f"  lam={lam:<5}: Val R² = {res['val_R2_mean']:.4f} ± {res['val_R2_std']:.4f} | Test R² = {res['test_R2_mean']:.4f} ± {res['test_R2_std']:.4f}")

    df_out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"\n[LambdaSweep] Completed. Results saved -> {output_csv}")
    print("\nSummary Table:")
    print(df_out.to_string(index=False))
    return df_out


if __name__ == "__main__":
    run_lambda_sweep()
