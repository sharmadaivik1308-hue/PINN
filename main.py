"""
main.py
-------
Single entry point for the corrected Physics-Informed Neural Network pipeline.

Orchestrates:
    1.  Configuration loading  (config/config.yaml + CLI overrides)
    2.  Device detection  (CUDA or CPU)
    3.  Dataset loading
    4.  Missing-value handling
    5.  Automatic feature audit  (degenerate detection + correlation analysis)
    6.  Target distribution analysis
    7.  Proper 3-way split  (train 70% / val 15% / test 15%)
    8.  Feature scaling  (scaler fitted on train only — no leakage)
    9.  Target scaling   (scaler fitted on train only — no leakage)
    10. Tensor creation  (all tensors on device)
    11. Model initialisation
    12. Mean-prediction baseline (trivial comparator)
    13. Main training run  (corrected joint optimisation)
    14. Test-set evaluation  (inverse-transformed predictions)
    15. Plot generation
    16. Output saving
    17. Feature analysis report  (FEATURE_ANALYSIS_REPORT.md)
    18. Ablation study  (experiments A / B / C / D)
    19. 5-fold cross-validation
    20. MODEL_TRAINING_AUDIT.md  (full results)

Usage:
    python main.py
    python main.py --lr 0.001 --epochs 500
    python main.py --config path/to/config.yaml
"""

import os
import sys
import io
# Force UTF-8 stdout so Unicode chars (arrows, Greek letters) don't crash on Windows CP1252
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import time
import argparse
import json
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader  import load_dataset
from src.preprocess   import (
    check_missing_values,
    handle_missing_values,
    split_data_three_way,
    scale_features,
    scale_target,
    to_tensors,
)
from src.feature_audit import (
    detect_degenerate_features,
    analyze_correlations,
    analyze_target,
    generate_feature_report,
)
from src.pinn_model   import OptimizedPINN
from src.train        import train_optimized, get_magnitude_prediction
from src.evaluate     import (
    compute_metrics,
    compute_mean_baseline,
    plot_loss_curves,
    plot_predicted_vs_actual,
    plot_error_distribution,
)
from src.utils        import (
    setup_directories,
    save_model,
    save_scaler,
    save_feature_names,
    save_predictions,
    save_metrics,
    save_loss_history,
)
from src.ablation        import run_ablation_study, format_ablation_table
from src.cross_validation import run_kfold_cv


# ============================================================================
# Configuration helpers
# ============================================================================

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.lr is not None:
        print(f"[Config] CLI override: learning_rate = {args.lr}")
        config["training"]["learning_rate"] = args.lr
    if args.epochs is not None:
        print(f"[Config] CLI override: epochs = {args.epochs}")
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        print(f"[Config] CLI override: batch_size = {args.batch_size}")
        config["training"]["batch_size"] = args.batch_size
    return config


# ============================================================================
# Plot helpers
# ============================================================================

def plot_physics_losses(history: dict, save_path: str) -> None:
    """Plot individual physics loss components over training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_energy"]) + 1)
    axes[0].plot(epochs, history["train_energy"],    label="Energy loss",    color="#FF5722", lw=1.8)
    axes[0].plot(epochs, history["train_frequency"], label="Frequency loss", color="#9C27B0", lw=1.8, ls="--")
    axes[0].set_title("Physics Loss Components", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Weighted Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_data"],    label="Data loss",    color="#2196F3", lw=1.8)
    axes[1].plot(epochs, history["train_physics"], label="Physics loss", color="#FF5722", lw=1.8, ls="--")
    axes[1].set_title("Data vs Physics Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Physics loss plot saved -> {save_path}")


def plot_attention_weights(
    model: torch.nn.Module,
    X:     torch.Tensor,
    feature_names: list,
    save_path: str,
) -> None:
    weights = model.get_attention_weights(X)
    if weights is None:
        return

    mean_weights   = weights.mean(dim=0).cpu().numpy()
    sorted_indices = np.argsort(mean_weights)
    sorted_names   = [feature_names[i] for i in sorted_indices]
    sorted_weights = mean_weights[sorted_indices]

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.35)))
    colors  = plt.cm.RdYlGn(sorted_weights / (sorted_weights.max() + 1e-8))
    ax.barh(sorted_names, sorted_weights, color=colors, edgecolor="white", lw=0.5)
    ax.set_xlabel("Attention Weight", fontsize=13)
    ax.set_title("Feature Importance (Attention)", fontsize=15, fontweight="bold")
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Attention weights plot saved -> {save_path}")


def plot_learning_rate(history: dict, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["learning_rate"], color="#4CAF50", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] LR schedule plot saved -> {save_path}")


# ============================================================================
# Report writer
# ============================================================================

def write_audit_report(
    path: str,
    problems_found: list,
    fixes_applied: list,
    dataset_info: dict,
    feature_info: dict,
    split_info: dict,
    preprocessing_info: dict,
    regularisation_info: dict,
    physics_audit: dict,
    timing_info: dict,
    main_metrics: dict,
    baseline_metrics: dict,
    ablation_table: str,
    cv_summary: dict,
    limitations: list,
) -> None:
    """Write MODEL_TRAINING_AUDIT.md."""
    L = []

    L += ["# Model Training Audit Report", "",
          "Auto-generated by `main.py`.", "", "---", ""]

    L += ["## 1. Original Problems Found", ""]
    for p in problems_found:
        L.append(f"- {p}")
    L.append("")

    L += ["## 2. Fixes Applied", ""]
    for f in fixes_applied:
        L.append(f"- {f}")
    L.append("")

    L += ["## 3. Dataset", ""]
    for k, v in dataset_info.items():
        L.append(f"- **{k}**: {v}")
    L.append("")

    L += ["## 4. Features Removed", ""]
    for k, v in feature_info.items():
        L.append(f"- **{k}**: {v}")
    L.append("")

    L += ["## 5. Train / Val / Test Split", ""]
    for k, v in split_info.items():
        L.append(f"- **{k}**: {v}")
    L.append("")

    L += ["## 6. Preprocessing", ""]
    for k, v in preprocessing_info.items():
        L.append(f"- **{k}**: {v}")
    L.append("")

    L += ["## 7. Regularisation", ""]
    for k, v in regularisation_info.items():
        L.append(f"- **{k}**: {v}")
    L.append("")

    L += ["## 8. Physics Loss Audit", ""]
    for k, v in physics_audit.items():
        L.append(f"- **{k}**: {v}")
    L.append("")

    L += ["## 9. Training Time", ""]
    for k, v in timing_info.items():
        L.append(f"- **{k}**: {v}")
    L.append("")

    L += ["## 10. Final Metrics (Test Set)", ""]
    L += ["| Metric | PINN | Mean Baseline |",
          "|---|---|---|",
          f"| R²   | {main_metrics.get('R2',  'N/A')} | {baseline_metrics.get('R2',  'N/A')} |",
          f"| MAE  | {main_metrics.get('MAE', 'N/A')} | {baseline_metrics.get('MAE', 'N/A')} |",
          f"| RMSE | {main_metrics.get('RMSE','N/A')} | {baseline_metrics.get('RMSE','N/A')} |",
          ""]

    L += ["## 11. Ablation Results", ""]
    if ablation_table:
        L += [ablation_table, ""]
    else:
        L += ["_Ablation not run._", ""]

    L += ["## 12. Cross-Validation Results (5-Fold)", ""]
    if cv_summary:
        L += [
            f"| Metric | Mean | Std |",
            "|---|---|---|",
            f"| R²   | {cv_summary.get('R2_mean',   'N/A')} | {cv_summary.get('R2_std',   'N/A')} |",
            f"| MAE  | {cv_summary.get('MAE_mean',  'N/A')} | {cv_summary.get('MAE_std',  'N/A')} |",
            f"| RMSE | {cv_summary.get('RMSE_mean', 'N/A')} | {cv_summary.get('RMSE_std', 'N/A')} |",
            "",
        ]
    else:
        L += ["_Cross-validation not run._", ""]

    L += ["## 13. Remaining Limitations", ""]
    for lim in limitations:
        L.append(f"- {lim}")
    L.append("")

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"\n[Audit] MODEL_TRAINING_AUDIT.md saved -> {path}")


# ============================================================================
# Pre-split pipeline  (newdata(claude) mode)
# ============================================================================

def _run_presplit_pipeline(config: dict) -> dict:
    """
    Pipeline variant for pre-split, pre-scaled datasets.

    Differences vs run_pipeline:
      - Loads train/val/test.csv directly (already split by event + feature-scaled).
      - Skips split_data_three_way and scale_features steps.
      - Uses log10_energy_raw column (converted to linear) for the GR physics loss.
      - Ablation and CV are disabled (event-based split breaks random k-fold).
      - All outputs go to the output paths defined in the config (outputs_newdata/ by default).
    """
    dataset_cfg  = config["dataset"]
    model_cfg    = config["model"]
    training_cfg = config["training"]
    physics_cfg  = config["physics"]
    output_cfg   = config["output"]

    model_dir   = output_cfg["model_dir"]
    metrics_dir = output_cfg["metrics_dir"]
    plots_dir   = output_cfg["plots_dir"]
    reports_dir = output_cfg.get("reports_dir", "outputs_newdata/reports")

    pipeline_start = time.time()

    print("\n" + "=" * 65)
    print("  PINN — EARTHQUAKE MAGNITUDE PREDICTION  [pre-split mode]")
    print("=" * 65 + "\n")

    # ── Device ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")
    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    # ── Directories ────────────────────────────────────────────────────────────
    for d in [model_dir, metrics_dir, plots_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Load pre-split CSVs ────────────────────────────────────────────────────
    target_col = dataset_cfg.get("target_column", "magnitude")
    phys_raw_col = dataset_cfg.get("physics_raw_energy_col", "log10_energy_raw")

    train_df = pd.read_csv(dataset_cfg["train_csv"])
    val_df   = pd.read_csv(dataset_cfg["val_csv"])
    test_df  = pd.read_csv(dataset_cfg["test_csv"])

    # Auto-detect numeric feature columns (exclude metadata + physics raw col)
    _meta_cols = {target_col, "filename", "station", "event_id", "channel", phys_raw_col}
    feature_names = [
        c for c in train_df.columns
        if c not in _meta_cols and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    original_features = feature_names  # no removal in pre-split mode

    print(f"[PreSplit] Train: {len(train_df)} rows | Val: {len(val_df)} rows "
          f"| Test: {len(test_df)} rows")
    print(f"[PreSplit] Features auto-detected: {len(feature_names)}")
    print(f"[PreSplit] Features already scaled — StandardScaler step skipped")

    X_train_df = train_df[feature_names]
    X_val_df   = val_df[feature_names]
    X_test_df  = test_df[feature_names]
    y_train_s  = train_df[target_col]
    y_val_s    = val_df[target_col]
    y_test_s   = test_df[target_col]

    target_stats = analyze_target(y_train_s)

    # ── Physics raw energy (log10_energy_raw → linear so energy_constraint
    #    can apply log10 internally, exactly as in the standard pipeline) ─────
    if phys_raw_col in train_df.columns:
        raw_e_train_np = 10.0 ** train_df[phys_raw_col].values
        raw_e_pool_np  = np.concatenate([
            10.0 ** train_df[phys_raw_col].values,
            10.0 ** val_df[phys_raw_col].values,
        ])
    else:
        print(f"[Physics] WARNING: '{phys_raw_col}' not in CSV — physics disabled.")
        physics_cfg["enabled"] = False
        raw_e_train_np = np.zeros(len(train_df))
        raw_e_pool_np  = np.zeros(len(train_df) + len(val_df))

    raw_energy_train = torch.tensor(raw_e_train_np, dtype=torch.float32, device=device)
    raw_freq_train   = torch.zeros(len(train_df),   dtype=torch.float32, device=device)
    raw_energy_pool  = raw_e_pool_np
    raw_freq_pool    = np.zeros(len(train_df) + len(val_df))

    print(f"[Physics] log10(E_raw) range (train): "
          f"{train_df[phys_raw_col].min():.3f} – {train_df[phys_raw_col].max():.3f}")

    # ── Feature arrays (already scaled — no StandardScaler needed) ────────────
    X_train_s  = X_train_df.values.astype(np.float32)
    X_val_s    = X_val_df.values.astype(np.float32)
    X_test_s   = X_test_df.values.astype(np.float32)
    feat_scaler = None   # no feature scaler in pre-split mode

    # Pool (train+val) — kept for report completeness; CV is disabled
    X_pool_df = pd.concat([X_train_df, X_val_df], ignore_index=True)
    y_pool_s  = pd.concat([y_train_s,  y_val_s],  ignore_index=True)

    # ── Target scaling (fitted on train only) ──────────────────────────────────
    scale_tgt = dataset_cfg.get("scale_target", True)
    if scale_tgt:
        y_train_sc, y_val_sc, y_test_sc, tgt_scaler = scale_target(
            y_train_s, y_val_s, y_test_s
        )
    else:
        tgt_scaler = None
        y_train_sc = y_train_s.values
        y_val_sc   = y_val_s.values
        y_test_sc  = y_test_s.values

    # ── Convert to tensors ─────────────────────────────────────────────────────
    X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t = to_tensors(
        X_train_s, X_val_s, X_test_s,
        y_train_sc, y_val_sc, y_test_sc,
        device=device,
    )

    # ── Model initialisation ───────────────────────────────────────────────────
    input_dim = X_train_t.shape[1]
    model_config = {
        "input_dim":      input_dim,
        "hidden_layers":  model_cfg.get("hidden_layers", [256, 128, 64]),
        "activation":     model_cfg.get("activation",    "gelu"),
        "dropout":        model_cfg.get("dropout",        0.2),
        "use_layer_norm": model_cfg.get("use_layer_norm", True),
        "use_batch_norm": model_cfg.get("use_batch_norm", False),
        "use_residual":   model_cfg.get("use_residual",   True),
        "use_attention":  model_cfg.get("use_attention",  True),
        "use_uncertainty":model_cfg.get("use_uncertainty",False),
    }
    model = OptimizedPINN(model_config)
    model.to(device)
    print(f"\n[Model] {model.get_config_string()}")
    print(f"[Model] Parameters: {model.count_parameters():,}")

    # ── Train config ───────────────────────────────────────────────────────────
    physics_exp = physics_cfg.get("experiment", "energy")
    if not physics_cfg.get("enabled", True):
        physics_exp = "none"

    coeff_a = physics_cfg.get("energy_coeff_a", 1.5)
    coeff_b = physics_cfg.get("energy_coeff_b", -3.8633)
    if physics_cfg.get("fit_intercept_on_train", True) and phys_raw_col in train_df.columns:
        train_log_e = train_df[phys_raw_col].values
        train_mags  = train_df[target_col].values
        coeff_b = float(np.mean(train_log_e - coeff_a * train_mags))
        print(f"[Physics] Dynamically fitted GR intercept on TRAIN only: b = {coeff_b:.4f} (log10(E) = {coeff_a}*M + ({coeff_b:.4f}))")

    train_config = {
        "epochs":           training_cfg["epochs"],
        "learning_rate":    training_cfg["learning_rate"],
        "weight_decay":     training_cfg.get("weight_decay",    1e-4),
        "batch_size":       training_cfg.get("batch_size",       64),
        "grad_clip":        training_cfg.get("grad_clip",        1.0),
        "patience":         training_cfg.get("patience",         150),
        "print_every":      training_cfg.get("print_every",       50),
        "scheduler":        training_cfg.get("scheduler",    "cosine"),
        "use_physics":      physics_cfg.get("enabled",          True),
        "physics_experiment": physics_exp,
        "energy_mode":      physics_cfg.get("energy_mode",     "gr_absolute"),
        "use_uncertainty":  model_cfg.get("use_uncertainty",   False),
        "lambda_energy":    physics_cfg.get("lambda_energy",     0.05),
        "lambda_frequency": physics_cfg.get("lambda_frequency",  0.0),
        "energy_coeff_a":   coeff_a,
        "energy_coeff_b":   coeff_b,
        "epsilon":          physics_cfg.get("epsilon",         1e-8),
        "physics_annealing":physics_cfg.get("annealing",       False),
    }

    # ── Mean baseline ──────────────────────────────────────────────────────────
    y_train_orig = y_train_s.values if hasattr(y_train_s, "values") else y_train_s
    y_test_orig  = y_test_s.values  if hasattr(y_test_s,  "values") else y_test_s
    baseline_metrics = compute_mean_baseline(y_train_orig, y_test_orig)

    # ── Training ───────────────────────────────────────────────────────────────
    model, history = train_optimized(
        model=model,
        X_train=X_train_t, y_train=y_train_t,
        X_val=X_val_t,     y_val=y_val_t,
        raw_energy_train=raw_energy_train,
        raw_frequency_train=raw_freq_train,
        config=train_config,
        model_name="pinn_newdata",
        device=device,
    )

    # ── Test-set evaluation ────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        test_output = model(X_test_t)

    use_unc = model_cfg.get("use_uncertainty", False)
    y_pred_scaled = (
        test_output[:, 0].cpu().numpy().flatten()
        if use_unc else test_output.cpu().numpy().flatten()
    )
    y_true_scaled = y_test_t.cpu().numpy().flatten()

    if tgt_scaler is not None:
        y_pred = tgt_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = tgt_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    else:
        y_pred, y_true = y_pred_scaled, y_true_scaled

    metrics = compute_metrics(y_true, y_pred)
    training_time = history["training_time_seconds"][0]
    metrics["training_time_seconds"] = round(training_time, 2)
    metrics["num_parameters"]  = model.count_parameters()
    metrics["actual_epochs"]   = history["actual_epochs"][0]
    metrics["best_val_loss"]   = round(min(history["val_total"]), 6)
    metrics["device"]          = str(device)

    # Baseline comparison metrics (A4)
    metrics["baseline_MAE"]  = baseline_metrics["MAE"]
    metrics["baseline_RMSE"] = baseline_metrics["RMSE"]
    metrics["baseline_R2"]   = baseline_metrics["R2"]
    metrics["beats_baseline_MAE"]  = bool(metrics["MAE"] < baseline_metrics["MAE"])
    metrics["beats_baseline_RMSE"] = bool(metrics["RMSE"] < baseline_metrics["RMSE"])
    metrics["beats_baseline_R2"]   = bool(metrics["R2"] > baseline_metrics["R2"])

    print("\n" + "=" * 55)
    print("  BASELINE COMPARISON [pre-split mode]")
    print("=" * 55)
    print(f"  PINN beats mean baseline (R2):   {'YES' if metrics['beats_baseline_R2'] else 'NO'} (PINN: {metrics['R2']:.4f} vs Baseline: {metrics['baseline_R2']:.4f})")
    print(f"  PINN beats mean baseline (MAE):  {'YES' if metrics['beats_baseline_MAE'] else 'NO'} (PINN: {metrics['MAE']:.4f} vs Baseline: {metrics['baseline_MAE']:.4f})")
    print(f"  PINN beats mean baseline (RMSE): {'YES' if metrics['beats_baseline_RMSE'] else 'NO'} (PINN: {metrics['RMSE']:.4f} vs Baseline: {metrics['baseline_RMSE']:.4f})")
    print("=" * 55 + "\n")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_loss_curves(
        history["train_total"], history["val_total"],
        save_path=os.path.join(plots_dir, "loss_curves.png"),
    )
    plot_predicted_vs_actual(
        y_true, y_pred,
        save_path=os.path.join(plots_dir, "predicted_vs_actual.png"),
    )
    plot_error_distribution(
        y_true, y_pred,
        save_path=os.path.join(plots_dir, "error_distribution.png"),
    )
    if physics_cfg.get("enabled", True):
        plot_physics_losses(
            history,
            save_path=os.path.join(plots_dir, "physics_losses.png"),
        )
    if model_cfg.get("use_attention", True):
        plot_attention_weights(
            model, X_train_t, feature_names,
            save_path=os.path.join(plots_dir, "attention_weights.png"),
        )
    if "learning_rate" in history:
        plot_learning_rate(
            history, save_path=os.path.join(plots_dir, "lr_schedule.png")
        )

    # ── Save outputs ───────────────────────────────────────────────────────────
    save_model(model, os.path.join(model_dir, "pinn_newdata.pth"))
    # feat_scaler is None in pre-split mode — skip saving
    if tgt_scaler is not None:
        save_scaler(tgt_scaler, os.path.join(model_dir, "target_scaler.pkl"))
    save_feature_names(feature_names, os.path.join(model_dir, "feature_names.json"))
    save_predictions(y_true, y_pred,  os.path.join(metrics_dir, "predictions.csv"))
    save_metrics(metrics,             os.path.join(metrics_dir, "metrics.json"))
    save_loss_history(history,        os.path.join(metrics_dir, "loss_history.json"))

    config_snap = os.path.join(metrics_dir, "config_snapshot.yaml")
    with open(config_snap, "w") as fh:
        yaml.dump(config, fh, default_flow_style=False)
    print(f"[Utils] Config snapshot saved -> {config_snap}")

    # ── Feature report (no degenerate/corr removal in pre-split mode) ──────────
    generate_feature_report(
        original_features   = original_features,
        degenerate_removed  = [],
        degenerate_reasons  = {},
        correlation_removed = [],
        correlation_groups  = {},
        final_features      = feature_names,
        target_stats        = target_stats,
        save_path           = os.path.join(reports_dir, "FEATURE_ANALYSIS_REPORT.md"),
    )

    # ── Audit report ───────────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    avg_epoch  = (sum(history["epoch_time"]) / max(1, len(history["epoch_time"])))

    write_audit_report(
        path = os.path.join(reports_dir, "MODEL_TRAINING_AUDIT.md"),
        problems_found = [
            "Original model stopped at epoch 75/500 (early stopping too tight)",
            "Network [64,32,16] too small for 49-feature input",
            "GR intercept b=4.8 assumed absolute Joules; dataset uses waveform proxy",
            "Random row-split leaked correlated multi-station recordings of same event",
            "Only 19 of 29 features used; log1p transforms not applied",
        ],
        fixes_applied = [
            "Pre-split data: train/val/test separated by event ID (no leakage)",
            "49 features used (all numeric columns incl. log1p-transformed variants)",
            "Network enlarged to [256,128,64]; dropout 0.2",
            "patience=150, cosine LR scheduler, 1000 epochs",
            "GR intercept updated to dataset-calibrated c=-3.863 (slope=1.5 fixed)",
            "lambda_energy reduced to 0.05 (weaker physics signal)",
            "log10_energy_raw converted to linear before physics constraint",
        ],
        dataset_info = {
            "Train samples":  len(train_df),
            "Val samples":    len(val_df),
            "Test samples":   len(test_df),
            "Split strategy": "Event-based (no station bleed-through)",
            "Train events":   "8 events",
            "Val events":     "2 events (M=4.6, 5.7 — unseen in train)",
            "Test events":    "2 events (M=4.2, 5.2 — unseen in train)",
            "Feature count":  len(feature_names),
            "Target range":   f"[{y_train_s.min():.2f}, {y_train_s.max():.2f}] (train)",
        },
        feature_info = {
            "Degenerate removed":  "0 (handled upstream in preprocessing_report)",
            "Correlated removed":  "0 (model uses all 49 features)",
            "Final feature count": len(feature_names),
        },
        split_info = {
            "Train": f"{len(train_df)} samples (event-based)",
            "Val":   f"{len(val_df)} samples (event-based)",
            "Test":  f"{len(test_df)} samples (event-based, held out)",
        },
        preprocessing_info = {
            "Feature scaler": "Pre-applied (StandardScaler in preprocessing_report)",
            "Target scaler":  "StandardScaler fitted on train magnitudes only",
            "Inverse transform": "Applied before metric computation",
        },
        regularisation_info = {
            "Optimizer":      "AdamW",
            "Weight decay":   training_cfg.get("weight_decay", 1e-4),
            "Dropout":        model_cfg.get("dropout", 0.2),
            "Early stopping": f"patience = {training_cfg.get('patience', 150)}",
            "LR scheduler":   training_cfg.get("scheduler", "cosine"),
        },
        physics_audit = {
            "Energy constraint":  "Gutenberg-Richter proportionality (log10(E) ~ 1.5*M + c)",
            "GR intercept c":     f"{physics_cfg.get('energy_coeff_b', 3.863)} (dataset-fitted)",
            "Physics column":     phys_raw_col,
            "lambda_energy":      physics_cfg.get("lambda_energy", 0.05),
            "Frequency":          "Disabled (lambda_frequency=0)",
        },
        timing_info = {
            "Total pipeline time": f"{total_time:.1f}s",
            "Main training time":  f"{training_time:.1f}s",
            "Avg epoch time":      f"{avg_epoch:.3f}s",
            "Epochs run":          history["actual_epochs"][0],
        },
        main_metrics     = metrics,
        baseline_metrics = baseline_metrics,
        ablation_table   = "",
        cv_summary       = {},
        limitations = [
            "Val/test events have magnitudes (4.6, 5.7, 4.2, 5.2) absent from training",
            "P-wave 3-second features still limited predictors of final magnitude",
            "49 features include many correlated pairs; attention handles but not optimal",
            "Waveform energy proxy != absolute seismic energy — GR law approximate",
        ],
    )

    # ── Done ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE  [pre-split mode]")
    print("=" * 65)
    print(f"  R2   : {metrics['R2']:.4f}  (baseline: {baseline_metrics['R2']:.4f})")
    print(f"  MAE  : {metrics['MAE']:.4f}  (baseline: {baseline_metrics['MAE']:.4f})")
    print(f"  RMSE : {metrics['RMSE']:.4f}  (baseline: {baseline_metrics['RMSE']:.4f})")
    print(f"  Epochs run  : {metrics['actual_epochs']}")
    print(f"  Total time  : {total_time:.1f}s")
    print(f"  Reports     : {reports_dir}/")
    print("=" * 65 + "\n")

    return metrics


# ============================================================================
# Main pipeline
# ============================================================================

def run_pipeline(config: dict) -> dict:
    """Execute the complete corrected PINN pipeline."""

    # Dispatch to the pre-split pipeline when the config requests it
    if config.get("dataset", {}).get("mode") == "pre_split":
        return _run_presplit_pipeline(config)

    dataset_cfg  = config["dataset"]
    model_cfg    = config["model"]
    training_cfg = config["training"]
    physics_cfg  = config["physics"]
    output_cfg   = config["output"]
    ablation_cfg = config.get("ablation", {})
    cv_cfg       = config.get("cross_validation", {})

    model_dir   = output_cfg["model_dir"]
    metrics_dir = output_cfg["metrics_dir"]
    plots_dir   = output_cfg["plots_dir"]
    reports_dir = output_cfg.get("reports_dir", "outputs/reports")

    pipeline_start = time.time()

    # ── Header ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  CORRECTED PINN — EARTHQUAKE MAGNITUDE PREDICTION")
    print("=" * 65 + "\n")

    # ── 0. Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")
    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    # ── 1. Setup directories ───────────────────────────────────────────────────
    setup_directories(config)
    os.makedirs(reports_dir, exist_ok=True)

    # ── 2. Load dataset ────────────────────────────────────────────────────────
    X_raw, y_raw, groups_raw = load_dataset(
        csv_path      = dataset_cfg["path"],
        target_column = dataset_cfg.get("target_column", "magnitude"),
        drop_columns  = dataset_cfg.get("drop_columns", ["filename"]),
        return_groups = True,
    )
    original_features = X_raw.columns.tolist()

    # ── 3. Handle missing values ───────────────────────────────────────────────
    check_missing_values(X_raw)
    X_raw = handle_missing_values(X_raw)
    y_raw = y_raw.loc[X_raw.index]
    if groups_raw is not None:
        groups_raw = groups_raw.loc[X_raw.index]

    # ── 4. Analyse target distribution ────────────────────────────────────────
    target_stats = analyze_target(y_raw)

    # ── 5. Feature audit — degenerate detection ────────────────────────────────
    var_thresh  = dataset_cfg.get("variance_threshold", 1e-6)
    corr_thresh = dataset_cfg.get("correlation_threshold", 0.95)

    degen_removed, degen_reasons = detect_degenerate_features(X_raw, var_thresh)
    X_audit = X_raw.drop(columns=degen_removed)

    # ── 6. Feature audit — correlation analysis ────────────────────────────────
    corr_removed, corr_groups = analyze_correlations(X_audit, corr_thresh)
    X_clean = X_audit.drop(columns=corr_removed)
    feature_names = X_clean.columns.tolist()

    print(f"\n[Audit] Final feature set: {len(feature_names)} features")
    print(f"  {feature_names}")

    # ── 7. 3-way split ─────────────────────────────────────────────────────────
    split_strategy = dataset_cfg.get("split_strategy", "event")
    X_train_df, X_val_df, X_test_df, y_train_s, y_val_s, y_test_s = (
        split_data_three_way(
            X_clean, y_raw,
            groups       = groups_raw.loc[X_clean.index] if groups_raw is not None else None,
            val_size     = dataset_cfg.get("val_size",  0.15),
            test_size    = dataset_cfg.get("test_size", 0.15),
            random_state = dataset_cfg.get("random_state", 42),
            split_strategy = split_strategy,
        )
    )

    # Build training pool (train + val) for cross-validation
    X_pool_df = pd.concat([X_train_df, X_val_df])
    y_pool_s  = pd.concat([y_train_s,  y_val_s])

    # ── 8. Extract raw physics features BEFORE scaling ─────────────────────────
    energy_col = physics_cfg.get("energy_feature",    "energy")
    freq_col   = physics_cfg.get("frequency_feature", "spectral_centroid")

    # Validate physics features exist after audit
    if energy_col not in X_train_df.columns:
        print(f"[Physics] WARNING: energy feature '{energy_col}' not in feature set — disabling physics.")
        physics_cfg["enabled"] = False
    if freq_col not in X_train_df.columns:
        print(f"[Physics] WARNING: frequency feature '{freq_col}' not in feature set.")

    raw_energy_train = torch.tensor(
        X_train_df[energy_col].values if energy_col in X_train_df.columns
        else np.zeros(len(X_train_df)),
        dtype=torch.float32, device=device,
    )
    raw_freq_train = torch.tensor(
        X_train_df[freq_col].values if freq_col in X_train_df.columns
        else np.zeros(len(X_train_df)),
        dtype=torch.float32, device=device,
    )

    # Pool physics tensors for CV
    raw_energy_pool = (
        X_pool_df[energy_col].values if energy_col in X_pool_df.columns
        else np.zeros(len(X_pool_df))
    )
    raw_freq_pool = (
        X_pool_df[freq_col].values if freq_col in X_pool_df.columns
        else np.zeros(len(X_pool_df))
    )

    print(f"[Physics] Energy range (train): {raw_energy_train.min():.4f} – "
          f"{raw_energy_train.max():.4f}")
    print(f"[Physics] Freq range   (train): {raw_freq_train.min():.4f} – "
          f"{raw_freq_train.max():.4f}")

    # ── 9. Feature scaling ─────────────────────────────────────────────────────
    X_train_s, X_val_s, X_test_s, feat_scaler = scale_features(
        X_train_df, X_val_df, X_test_df
    )

    # ── 10. Target scaling ─────────────────────────────────────────────────────
    scale_tgt = dataset_cfg.get("scale_target", True)
    if scale_tgt:
        y_train_sc, y_val_sc, y_test_sc, tgt_scaler = scale_target(
            y_train_s, y_val_s, y_test_s
        )
    else:
        tgt_scaler = None
        y_train_sc = y_train_s.values
        y_val_sc   = y_val_s.values
        y_test_sc  = y_test_s.values

    # ── 11. Convert to tensors ─────────────────────────────────────────────────
    X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t = to_tensors(
        X_train_s, X_val_s, X_test_s,
        y_train_sc, y_val_sc, y_test_sc,
        device=device,
    )

    # ── 12. Model initialisation ───────────────────────────────────────────────
    input_dim = X_train_t.shape[1]
    model_config = {
        "input_dim":    input_dim,
        "hidden_layers": model_cfg.get("hidden_layers", [64, 32, 16]),
        "activation":   model_cfg.get("activation",    "gelu"),
        "dropout":      model_cfg.get("dropout",        0.1),
        "use_layer_norm": model_cfg.get("use_layer_norm", True),
        "use_batch_norm": model_cfg.get("use_batch_norm", False),
        "use_residual":   model_cfg.get("use_residual",   True),
        "use_attention":  model_cfg.get("use_attention",  True),
        "use_uncertainty":model_cfg.get("use_uncertainty",False),
    }

    model = OptimizedPINN(model_config)
    model.to(device)
    print(f"\n[Model] {model.get_config_string()}")
    print(f"[Model] Parameters: {model.count_parameters():,}")

    # ── 13. Build train config ─────────────────────────────────────────────────
    physics_exp = physics_cfg.get("experiment", "energy")
    if not physics_cfg.get("enabled", True):
        physics_exp = "none"

    coeff_a = physics_cfg.get("energy_coeff_a", 1.5)
    coeff_b = physics_cfg.get("energy_coeff_b", -3.8633)
    if physics_cfg.get("fit_intercept_on_train", True) and energy_col in X_train_df.columns:
        train_log_e = np.log10(np.abs(X_train_df[energy_col].values) + physics_cfg.get("epsilon", 1e-8))
        train_mags  = y_train_s.values
        coeff_b = float(np.mean(train_log_e - coeff_a * train_mags))
        print(f"[Physics] Dynamically fitted GR intercept on TRAIN only: b = {coeff_b:.4f} (log10(E) = {coeff_a}*M + ({coeff_b:.4f}))")

    train_config = {
        "epochs":           training_cfg["epochs"],
        "learning_rate":    training_cfg["learning_rate"],
        "weight_decay":     training_cfg.get("weight_decay",    1e-4),
        "batch_size":       training_cfg.get("batch_size",       64),
        "grad_clip":        training_cfg.get("grad_clip",        1.0),
        "patience":         training_cfg.get("patience",          50),
        "print_every":      training_cfg.get("print_every",       20),
        "scheduler":        training_cfg.get("scheduler",    "plateau"),
        "use_physics":      physics_cfg.get("enabled",          True),
        "physics_experiment": physics_exp,
        "energy_mode":      physics_cfg.get("energy_mode",     "correlation"),
        "use_uncertainty":  model_cfg.get("use_uncertainty",   False),
        "lambda_energy":    physics_cfg.get("lambda_energy",     0.1),
        "lambda_frequency": physics_cfg.get("lambda_frequency",  0.0),
        "energy_coeff_a":   coeff_a,
        "energy_coeff_b":   coeff_b,
        "epsilon":          physics_cfg.get("epsilon",         1e-8),
        "physics_annealing":physics_cfg.get("annealing",       False),
    }

    # ── 14. Mean baseline ──────────────────────────────────────────────────────
    # Compare against trivial mean predictor (original magnitude scale)
    y_train_orig = y_train_s.values if hasattr(y_train_s, "values") else y_train_s
    y_test_orig  = y_test_s.values  if hasattr(y_test_s,  "values") else y_test_s
    baseline_metrics = compute_mean_baseline(y_train_orig, y_test_orig)

    # ── 15. Main training run ──────────────────────────────────────────────────
    model, history = train_optimized(
        model=model,
        X_train=X_train_t, y_train=y_train_t,
        X_val=X_val_t,     y_val=y_val_t,
        raw_energy_train=raw_energy_train,
        raw_frequency_train=raw_freq_train,
        config=train_config,
        model_name="optimized_pinn",
        device=device,
    )

    # ── 16. Test-set evaluation ────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        test_output = model(X_test_t)

    use_unc = model_cfg.get("use_uncertainty", False)
    y_pred_scaled = (
        test_output[:, 0].cpu().numpy().flatten()
        if use_unc else test_output.cpu().numpy().flatten()
    )
    y_true_scaled = y_test_t.cpu().numpy().flatten()

    # Inverse-transform to original magnitude scale for metric reporting
    if tgt_scaler is not None:
        y_pred = tgt_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = tgt_scaler.inverse_transform(
            y_true_scaled.reshape(-1, 1)).flatten()
    else:
        y_pred, y_true = y_pred_scaled, y_true_scaled

    metrics = compute_metrics(y_true, y_pred)

    # Training metadata
    training_time = history["training_time_seconds"][0]
    metrics["training_time_seconds"] = round(training_time, 2)
    metrics["num_parameters"]  = model.count_parameters()
    metrics["actual_epochs"]   = history["actual_epochs"][0]
    metrics["best_val_loss"]   = round(min(history["val_total"]), 6)
    metrics["device"]          = str(device)

    # Baseline comparison metrics (A4)
    metrics["baseline_MAE"]  = baseline_metrics["MAE"]
    metrics["baseline_RMSE"] = baseline_metrics["RMSE"]
    metrics["baseline_R2"]   = baseline_metrics["R2"]
    metrics["beats_baseline_MAE"]  = bool(metrics["MAE"] < baseline_metrics["MAE"])
    metrics["beats_baseline_RMSE"] = bool(metrics["RMSE"] < baseline_metrics["RMSE"])
    metrics["beats_baseline_R2"]   = bool(metrics["R2"] > baseline_metrics["R2"])

    print("\n" + "=" * 55)
    print("  BASELINE COMPARISON")
    print("=" * 55)
    print(f"  PINN beats mean baseline (R2):   {'YES' if metrics['beats_baseline_R2'] else 'NO'} (PINN: {metrics['R2']:.4f} vs Baseline: {metrics['baseline_R2']:.4f})")
    print(f"  PINN beats mean baseline (MAE):  {'YES' if metrics['beats_baseline_MAE'] else 'NO'} ({metrics['MAE']:.4f} vs Baseline: {metrics['baseline_MAE']:.4f})")
    print(f"  PINN beats mean baseline (RMSE): {'YES' if metrics['beats_baseline_RMSE'] else 'NO'} ({metrics['RMSE']:.4f} vs Baseline: {metrics['baseline_RMSE']:.4f})")
    print("=" * 55 + "\n")

    # ── 17. Plots ──────────────────────────────────────────────────────────────
    plot_loss_curves(
        history["train_total"], history["val_total"],
        save_path=os.path.join(plots_dir, "loss_curves.png"),
    )
    plot_predicted_vs_actual(
        y_true, y_pred,
        save_path=os.path.join(plots_dir, "predicted_vs_actual.png"),
    )
    plot_error_distribution(
        y_true, y_pred,
        save_path=os.path.join(plots_dir, "error_distribution.png"),
    )
    if physics_cfg.get("enabled", True):
        plot_physics_losses(
            history,
            save_path=os.path.join(plots_dir, "physics_losses.png"),
        )
    if model_cfg.get("use_attention", True):
        plot_attention_weights(
            model, X_train_t, feature_names,
            save_path=os.path.join(plots_dir, "attention_weights.png"),
        )
    if "learning_rate" in history:
        plot_learning_rate(
            history, save_path=os.path.join(plots_dir, "lr_schedule.png")
        )

    # ── 18. Save primary outputs ───────────────────────────────────────────────
    save_model(model,       os.path.join(model_dir, "optimized_pinn.pth"))
    if feat_scaler is not None:
        save_scaler(feat_scaler, os.path.join(model_dir, "scaler.pkl"))
    if tgt_scaler is not None:
        save_scaler(tgt_scaler, os.path.join(model_dir, "target_scaler.pkl"))
    save_feature_names(feature_names, os.path.join(model_dir, "feature_names.json"))
    save_predictions(y_true, y_pred,   os.path.join(metrics_dir, "predictions.csv"))
    save_metrics(metrics,              os.path.join(metrics_dir, "metrics.json"))
    save_loss_history(history,         os.path.join(metrics_dir, "loss_history.json"))

    config_snap = os.path.join(metrics_dir, "config_snapshot.yaml")
    with open(config_snap, "w") as fh:
        yaml.dump(config, fh, default_flow_style=False)
    print(f"[Utils] Config snapshot saved -> {config_snap}")

    # ── 19. Feature analysis report ────────────────────────────────────────────
    generate_feature_report(
        original_features   = original_features,
        degenerate_removed  = degen_removed,
        degenerate_reasons  = degen_reasons,
        correlation_removed = corr_removed,
        correlation_groups  = corr_groups,
        final_features      = feature_names,
        target_stats        = target_stats,
        save_path           = os.path.join(reports_dir, "FEATURE_ANALYSIS_REPORT.md"),
    )

    # ── 20. Ablation study ─────────────────────────────────────────────────────
    ablation_results = []
    ablation_table   = ""
    if ablation_cfg.get("enabled", True):
        ablation_results = run_ablation_study(
            X_train=X_train_t, y_train=y_train_t,
            X_val=X_val_t,     y_val=y_val_t,
            X_test=X_test_t,   y_test=y_test_t,
            raw_energy_train=raw_energy_train,
            raw_frequency_train=raw_freq_train,
            base_model_config=model_config,
            base_train_config=train_config,
            target_scaler=tgt_scaler,
            device=device,
        )
        ablation_table = format_ablation_table(ablation_results)
        print("\n[Ablation] Results:\n")
        print(ablation_table)

        # Save ablation table
        with open(os.path.join(reports_dir, "ablation_results.md"), "w") as fh:
            fh.write("# Ablation Study Results\n\n")
            fh.write(ablation_table + "\n")
        print(f"[Ablation] Table saved -> {os.path.join(reports_dir, 'ablation_results.md')}")

    # ── 21. Cross-validation ───────────────────────────────────────────────────
    cv_summary = {}
    if cv_cfg.get("enabled", True):
        if split_strategy == "event":
            cv_X = X_clean
            cv_y = y_raw
            cv_e = (
                X_clean[energy_col].values if energy_col in X_clean.columns
                else np.zeros(len(X_clean))
            )
            cv_f = (
                X_clean[freq_col].values if freq_col in X_clean.columns
                else np.zeros(len(X_clean))
            )
            cv_groups = groups_raw.loc[X_clean.index] if groups_raw is not None else None
        else:
            cv_X = X_pool_df
            cv_y = y_pool_s
            cv_e = raw_energy_pool
            cv_f = raw_freq_pool
            cv_groups = None

        cv_summary = run_kfold_cv(
            X_pool             = cv_X,
            y_pool             = cv_y,
            raw_energy_pool    = cv_e,
            raw_frequency_pool = cv_f,
            model_config       = model_config,
            train_config       = train_config,
            groups_pool        = cv_groups,
            split_strategy     = split_strategy,
            k                  = cv_cfg.get("folds", 5),
            scale_target       = scale_tgt,
            device             = device,
            random_state       = dataset_cfg.get("random_state", 42),
        )

    # ── 22. Audit report ───────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start
    avg_epoch  = (sum(history["epoch_time"])
                  / max(1, len(history["epoch_time"])))

    write_audit_report(
        path = os.path.join(reports_dir, "MODEL_TRAINING_AUDIT.md"),
        problems_found = [
            "X_val = X_test: test set used for early stopping and LR scheduling (leakage)",
            "Physics backward pass separate from data backward — gradients wiped between steps",
            "Physics not applied at batch level — skipped in mini-batch loop",
            "No target scaling despite only 11 unique magnitude values",
            "No 3-way train/val/test split — only 2-way",
            "No automatic degenerate feature detection",
            "dominant_freq all-zero (2497/2497 samples)",
            "No device management (always CPU)",
        ],
        fixes_applied = [
            "3-way split (70% train / 15% val / 15% test)",
            "Scaler fitted on train only — val and test only transformed",
            "Target StandardScaler fitted on train only",
            "Physics loss combined with data loss before single backward()",
            "Physics tensors included in TensorDataset for batch alignment",
            "Separate energy and frequency loss logging per epoch",
            "Automatic degenerate feature detection (variance threshold)",
            "Automatic correlation-based redundancy removal",
            "Device detection (CUDA / CPU)",
            "Per-epoch timing",
            "Mean-prediction baseline added for comparison",
        ],
        dataset_info = {
            "Total samples":    len(X_raw),
            "Original features": len(original_features),
            "Target range":     f"[{y_raw.min():.2f}, {y_raw.max():.2f}]",
            "Unique magnitudes": target_stats["unique_values"],
        },
        feature_info = {
            "Degenerate removed":   len(degen_removed),
            "Degenerate features":  str(degen_removed),
            "Correlated removed":   len(corr_removed),
            "Correlated features":  str(corr_removed),
            "Final feature count":  len(feature_names),
        },
        split_info = {
            "Train": f"{len(X_train_df)} samples",
            "Val":   f"{len(X_val_df)} samples",
            "Test":  f"{len(X_test_df)} samples",
            "Note":  "Test set not used for model selection",
        },
        preprocessing_info = {
            "Feature scaler":       "StandardScaler (fit on train only)",
            "Target scaler":        "StandardScaler (fit on train only)" if scale_tgt else "None",
            "Inverse transform":    "Applied before metric computation",
        },
        regularisation_info = {
            "Optimizer":     "AdamW" if training_cfg.get("weight_decay", 0) > 0 else "Adam",
            "Weight decay":  training_cfg.get("weight_decay", 0),
            "Dropout":       model_cfg.get("dropout", 0.1),
            "Early stopping": f"patience = {training_cfg.get('patience', 50)}",
            "LR scheduler":  training_cfg.get("scheduler", "plateau"),
        },
        physics_audit = {
            "Energy constraint":    "Gutenberg–Richter proportionality (log₁₀E ∝ M)",
            "Energy feature":       energy_col,
            "Energy note":          "Feature is P-wave integral ∫a²dt, not absolute Joules; proportionality enforced",
            "Frequency constraint": "M ∝ −log(f) using spectral_centroid",
            "Frequency note":       f"Correlation r=−0.058 (very weak); disabled by default (λ_freq=0)",
            "Implementation fix":   "Single joint backward pass (data + physics combined)",
            "Batch alignment":      "Physics tensors in TensorDataset — no index mismatch",
        },
        timing_info = {
            "Total pipeline time":  f"{total_time:.1f}s",
            "Main training time":   f"{training_time:.1f}s",
            "Avg epoch time":       f"{avg_epoch:.3f}s",
            "Epochs run":           history["actual_epochs"][0],
        },
        main_metrics      = metrics,
        baseline_metrics  = baseline_metrics,
        ablation_table    = ablation_table,
        cv_summary        = cv_summary,
        limitations = [
            "Only 11 unique magnitude values — inherently limited regression signal",
            "Frequency constraint (spectral_centroid) has r=−0.058 — nearly zero signal",
            "P-wave energy feature is not absolute Joules — G-R coefficients are proxies only",
            "Dataset size (~2500 samples) limits what any neural network can achieve",
            "R² may remain low if the dominant variance in magnitude is not captured by P-wave features",
        ],
    )

    # ── Done ───────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  R²   : {metrics['R2']:.4f}  (baseline: {baseline_metrics['R2']:.4f})")
    print(f"  MAE  : {metrics['MAE']:.4f}  (baseline: {baseline_metrics['MAE']:.4f})")
    print(f"  RMSE : {metrics['RMSE']:.4f}  (baseline: {baseline_metrics['RMSE']:.4f})")
    print(f"  Total time : {total_time:.1f}s")
    print(f"  Reports    : {reports_dir}/")
    print("=" * 65 + "\n")

    return metrics


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corrected PINN for Earthquake Magnitude Prediction"
    )
    parser.add_argument("--config",     type=str,   default="config/config.yaml")
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch-size", type=int,   default=None, dest="batch_size")
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
