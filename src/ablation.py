"""
ablation.py
-----------
Controlled ablation study for the earthquake PINN pipeline.

Four experiments (in order of increasing complexity):

    A. Baseline      — data loss only, no dropout, no weight decay,
                       no early stopping.
    B. Regularized   — dropout + weight decay + early stopping, no physics.
    C. Energy PINN   — same as B, plus energy–magnitude physics constraint.
    D. Full PINN     — same as B, plus energy + frequency constraints.

Each experiment trains a freshly initialised model on the same 3-way split
so results are directly comparable.  The test set is evaluated after all
experiments are complete — never used for model selection.
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from src.pinn_model import OptimizedPINN
from src.train import train_optimized
from src.evaluate import compute_metrics


# ---------------------------------------------------------------------------

def _fresh_model(model_config: Dict, device: torch.device) -> OptimizedPINN:
    """Instantiate and move a new OptimizedPINN to the target device."""
    m = OptimizedPINN(model_config)
    m.to(device)
    return m


# ---------------------------------------------------------------------------

def run_ablation_study(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    raw_energy_train: torch.Tensor,
    raw_frequency_train: torch.Tensor,
    base_model_config: Dict,
    base_train_config: Dict,
    target_scaler=None,
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """Run experiments A–D and return a list of result dictionaries.

    Args:
        X_train … y_test:      Pre-split, feature-scaled, on-device tensors.
        raw_energy_train:      Raw (un-normalised) energy (on device).
        raw_frequency_train:   Raw spectral centroid values (on device).
        base_model_config:     Model architecture config dict.
        base_train_config:     Training hyperparameter config dict.
        target_scaler:         Fitted StandardScaler for y (or None).
        device:                Torch device.

    Returns:
        List of dicts, each containing metrics and history for one experiment.
    """
    if device is None:
        device = X_train.device

    # Experiment definitions ------------------------------------------------
    experiments = [
        {
            "name":              "A_Baseline",
            "label":             "A: Baseline",
            "dropout":           0.0,
            "weight_decay":      0.0,
            # Very large patience ≈ no early stopping for the baseline
            "patience":          base_train_config.get("epochs", 500) + 9999,
            "use_physics":       False,
            "physics_experiment":"none",
            "lambda_energy":     0.0,
            "lambda_frequency":  0.0,
            "scheduler":         "none",
        },
        {
            "name":              "B_Regularized",
            "label":             "B: Regularized",
            "dropout":           base_model_config.get("dropout", 0.1),
            "weight_decay":      base_train_config.get("weight_decay", 1e-4),
            "patience":          base_train_config.get("patience", 50),
            "use_physics":       False,
            "physics_experiment":"none",
            "lambda_energy":     0.0,
            "lambda_frequency":  0.0,
            "scheduler":         base_train_config.get("scheduler", "plateau"),
        },
        {
            "name":              "C_EnergyPINN",
            "label":             "C: Energy PINN",
            "dropout":           base_model_config.get("dropout", 0.1),
            "weight_decay":      base_train_config.get("weight_decay", 1e-4),
            "patience":          base_train_config.get("patience", 50),
            "use_physics":       True,
            "physics_experiment":"energy",
            "lambda_energy":     base_train_config.get("lambda_energy", 0.1),
            "lambda_frequency":  0.0,
            "scheduler":         base_train_config.get("scheduler", "plateau"),
        },
        {
            "name":              "D_FullPINN",
            "label":             "D: Full PINN",
            "dropout":           base_model_config.get("dropout", 0.1),
            "weight_decay":      base_train_config.get("weight_decay", 1e-4),
            "patience":          base_train_config.get("patience", 50),
            "use_physics":       True,
            "physics_experiment":"full",
            "lambda_energy":     base_train_config.get("lambda_energy", 0.1),
            "lambda_frequency":  base_train_config.get("lambda_frequency", 0.05),
            "scheduler":         base_train_config.get("scheduler", "plateau"),
        },
    ]

    results: List[Dict] = []
    seeds = [0, 1, 2, 3, 4]

    for exp in experiments:
        print(f"\n{'='*72}")
        print(f"  ABLATION: {exp['label']} (Running across 5 seeds: {seeds})")
        print(f"{'='*72}")

        val_r2_list, val_mae_list, val_rmse_list = [], [], []
        test_r2_list, test_mae_list, test_rmse_list = [], [], []
        epoch_list = []
        elapsed_total = 0.0
        last_history = None

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Fresh model with experiment-specific dropout
            model_cfg = copy.deepcopy(base_model_config)
            model_cfg["dropout"] = exp["dropout"]
            model = _fresh_model(model_cfg, device)

            # Training config overrides
            train_cfg = copy.deepcopy(base_train_config)
            train_cfg.update({
                "use_physics":        exp["use_physics"],
                "physics_experiment": exp["physics_experiment"],
                "lambda_energy":      exp["lambda_energy"],
                "lambda_frequency":   exp["lambda_frequency"],
                "weight_decay":       exp["weight_decay"],
                "patience":           exp["patience"],
                "scheduler":          exp["scheduler"],
            })

            t0 = time.time()
            model, history = train_optimized(
                model=model,
                X_train=X_train, y_train=y_train,
                X_val=X_val,     y_val=y_val,
                raw_energy_train=raw_energy_train,
                raw_frequency_train=raw_frequency_train,
                config=train_cfg,
                model_name=f"{exp['name']}_s{seed}",
                device=device,
            )
            elapsed_total += time.time() - t0
            last_history = history
            epoch_list.append(history["actual_epochs"][0])

            # Evaluate on Validation set (Used for selection/ranking)
            model.eval()
            with torch.no_grad():
                val_out  = model(X_val)
                test_out = model(X_test)

            use_unc = model_cfg.get("use_uncertainty", False)
            y_val_pred_s = (
                val_out[:, 0].cpu().numpy().flatten()
                if use_unc else val_out.cpu().numpy().flatten()
            )
            y_val_true_s = y_val.cpu().numpy().flatten()

            y_test_pred_s = (
                test_out[:, 0].cpu().numpy().flatten()
                if use_unc else test_out.cpu().numpy().flatten()
            )
            y_test_true_s = y_test.cpu().numpy().flatten()

            if target_scaler is not None:
                y_val_pred = target_scaler.inverse_transform(y_val_pred_s.reshape(-1, 1)).flatten()
                y_val_true = target_scaler.inverse_transform(y_val_true_s.reshape(-1, 1)).flatten()
                y_test_pred = target_scaler.inverse_transform(y_test_pred_s.reshape(-1, 1)).flatten()
                y_test_true = target_scaler.inverse_transform(y_test_true_s.reshape(-1, 1)).flatten()
            else:
                y_val_pred, y_val_true = y_val_pred_s, y_val_true_s
                y_test_pred, y_test_true = y_test_pred_s, y_test_true_s

            m_val  = compute_metrics(y_val_true, y_val_pred)
            m_test = compute_metrics(y_test_true, y_test_pred)

            val_r2_list.append(m_val["R2"])
            val_mae_list.append(m_val["MAE"])
            val_rmse_list.append(m_val["RMSE"])

            test_r2_list.append(m_test["R2"])
            test_mae_list.append(m_test["MAE"])
            test_rmse_list.append(m_test["RMSE"])

        results.append({
            "name":              exp["name"],
            "label":             exp["label"],
            "physics":           exp["physics_experiment"],
            "dropout":           exp["dropout"],
            "weight_decay":      exp["weight_decay"],
            "early_stop":        exp["patience"] <= base_train_config.get("epochs", 500),
            "val_R2_mean":       float(np.mean(val_r2_list)),
            "val_R2_std":        float(np.std(val_r2_list)),
            "val_MAE_mean":      float(np.mean(val_mae_list)),
            "val_MAE_std":       float(np.std(val_mae_list)),
            "val_RMSE_mean":     float(np.mean(val_rmse_list)),
            "val_RMSE_std":      float(np.std(val_rmse_list)),
            "test_R2_mean":      float(np.mean(test_r2_list)),
            "test_R2_std":       float(np.std(test_r2_list)),
            "test_MAE_mean":     float(np.mean(test_mae_list)),
            "test_MAE_std":      float(np.std(test_mae_list)),
            "test_RMSE_mean":    float(np.mean(test_rmse_list)),
            "test_RMSE_std":     float(np.std(test_rmse_list)),
            "training_time":     round(elapsed_total, 1),
            "avg_epochs":        float(np.mean(epoch_list)),
            "history":           last_history,
        })

    # Rank and choose best model strictly based on validation R²
    results.sort(key=lambda r: r["val_R2_mean"], reverse=True)
    for rank, r in enumerate(results, 1):
        r["val_rank"] = rank

    print(f"\n[Ablation] Model Selection Ranking (based on 5-seed Validation R²):")
    for r in results:
        print(f"  Rank {r['val_rank']}: {r['label']} -> Val R² = {r['val_R2_mean']:.4f} ± {r['val_R2_std']:.4f} | Test R² = {r['test_R2_mean']:.4f} ± {r['test_R2_std']:.4f}")

    return results


# ---------------------------------------------------------------------------

def format_ablation_table(results: List[Dict]) -> str:
    """Return ablation results as a Markdown table string ranked by Validation R²."""
    header = (
        "| Rank | Experiment | Physics | Val R² (mean±std) | Val MAE | Val RMSE "
        "| Test R² (mean±std) | Test MAE | Test RMSE | Time (s) |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]

    for r in results:
        rows.append(
            f"| {r.get('val_rank', '-')} "
            f"| {r['label']} "
            f"| {r['physics']} "
            f"| {r['val_R2_mean']:.4f} ± {r['val_R2_std']:.4f} "
            f"| {r['val_MAE_mean']:.4f} "
            f"| {r['val_RMSE_mean']:.4f} "
            f"| {r['test_R2_mean']:.4f} ± {r['test_R2_std']:.4f} "
            f"| {r['test_MAE_mean']:.4f} "
            f"| {r['test_RMSE_mean']:.4f} "
            f"| {r['training_time']} |"
        )

    return "\n".join(rows)
