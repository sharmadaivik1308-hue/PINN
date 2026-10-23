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

    for exp in experiments:
        print(f"\n{'='*72}")
        print(f"  ABLATION  {exp['label']}")
        print(f"{'='*72}")

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
            model_name=exp["name"],
            device=device,
        )
        elapsed = time.time() - t0

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_out = model(X_test)

        use_unc = model_cfg.get("use_uncertainty", False)
        y_pred_s = (
            test_out[:, 0].cpu().numpy().flatten()
            if use_unc else test_out.cpu().numpy().flatten()
        )
        y_true_s = y_test.cpu().numpy().flatten()

        # Inverse-transform if target was scaled
        if target_scaler is not None:
            y_pred = target_scaler.inverse_transform(
                y_pred_s.reshape(-1, 1)).flatten()
            y_true = target_scaler.inverse_transform(
                y_true_s.reshape(-1, 1)).flatten()
        else:
            y_pred, y_true = y_pred_s, y_true_s

        metrics = compute_metrics(y_true, y_pred)

        results.append({
            "name":           exp["name"],
            "label":          exp["label"],
            "physics":        exp["physics_experiment"],
            "dropout":        exp["dropout"],
            "weight_decay":   exp["weight_decay"],
            "early_stop":     exp["patience"] <= base_train_config.get("epochs", 500),
            "R2":             metrics["R2"],
            "MAE":            metrics["MAE"],
            "RMSE":           metrics["RMSE"],
            "training_time":  round(elapsed, 1),
            "actual_epochs":  history["actual_epochs"][0],
            "history":        history,
        })

    return results


# ---------------------------------------------------------------------------

def format_ablation_table(results: List[Dict]) -> str:
    """Return ablation results as a Markdown table string."""
    header = (
        "| Experiment | Physics | Dropout | Weight Decay | Early Stop "
        "| R² | MAE | RMSE | Time (s) |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]

    for r in results:
        rows.append(
            f"| {r['label']} "
            f"| {r['physics']} "
            f"| {r['dropout']} "
            f"| {r['weight_decay']:.0e} "
            f"| {'Yes' if r['early_stop'] else 'No'} "
            f"| {r['R2']:.4f} "
            f"| {r['MAE']:.4f} "
            f"| {r['RMSE']:.4f} "
            f"| {r['training_time']} |"
        )

    return "\n".join(rows)
