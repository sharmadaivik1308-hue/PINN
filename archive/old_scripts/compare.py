"""
compare.py
----------
Generates comprehensive side-by-side comparison visualisations for all
trained model variants. Produces publication-quality plots for:

    1. Metrics bar chart (MAE, RMSE, R²)
    2. Loss curves comparison
    3. Predicted vs Actual side-by-side
    4. Error distributions overlay
    5. Residual plots
    6. Training time comparison

All plots are saved to the outputs/plots/ directory.

Usage:
    python compare.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

import src.utils as _utils


def _get_paths():
    """Get current output paths from utils (respects configure_paths)."""
    return _utils.PLOTS_DIR, _utils.METRICS_DIR, _utils.PREDICTIONS_DIR


# -- Colour palette for consistent styling across all plots ----------------
MODEL_COLOURS = {
    "baseline_mlp": "#2196F3",     # Blue
    "basic_pinn": "#4CAF50",       # Green
    "attention_pinn": "#FF9800",   # Orange
    "full_pinn": "#E91E63",        # Pink
}

MODEL_LABELS = {
    "baseline_mlp": "Baseline MLP",
    "basic_pinn": "Basic PINN",
    "attention_pinn": "Attention PINN",
    "full_pinn": "Full PINN",
}

# Canonical order for plotting
MODEL_ORDER = ["baseline_mlp", "basic_pinn", "attention_pinn", "full_pinn"]


def _load_available_metrics() -> Dict[str, dict]:
    """Load metrics for all available models."""
    _, METRICS_DIR, _ = _get_paths()
    metrics = {}
    for model_name in MODEL_ORDER:
        path = os.path.join(METRICS_DIR, f"{model_name}_metrics.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                metrics[model_name] = json.load(f)
    return metrics


def _load_available_histories() -> Dict[str, dict]:
    """Load loss histories for all available models."""
    _, METRICS_DIR, _ = _get_paths()
    histories = {}
    for model_name in MODEL_ORDER:
        path = os.path.join(METRICS_DIR, f"{model_name}_loss_history.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                histories[model_name] = json.load(f)
    return histories


def _load_available_predictions() -> Dict[str, pd.DataFrame]:
    """Load predictions for all available models."""
    _, _, PREDICTIONS_DIR = _get_paths()
    predictions = {}
    for model_name in MODEL_ORDER:
        path = os.path.join(PREDICTIONS_DIR, f"{model_name}_predictions.csv")
        if os.path.exists(path):
            predictions[model_name] = pd.read_csv(path)
    return predictions


def plot_metrics_comparison(metrics: Dict[str, dict]) -> None:
    """Generate a grouped bar chart comparing MAE, RMSE, and R² across models."""
    models = [m for m in MODEL_ORDER if m in metrics]
    if len(models) < 2:
        print("[Compare] Not enough models for comparison. Skipping metrics plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metric_names = ["MAE", "RMSE", "R2"]
    titles = ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)",
              "R² Score"]

    for idx, (metric_key, title) in enumerate(zip(metric_names, titles)):
        ax = axes[idx]
        values = [metrics[m].get(metric_key, 0) for m in models]
        colours = [MODEL_COLOURS.get(m, "#999") for m in models]
        labels = [MODEL_LABELS.get(m, m) for m in models]

        bars = ax.bar(labels, values, color=colours, edgecolor="white",
                      linewidth=1.5, width=0.6)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            y_pos = bar.get_height()
            if y_pos < 0:
                y_pos = bar.get_height() - 0.02
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.tick_params(labelsize=10, axis="x", rotation=15)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Model Comparison — Evaluation Metrics", fontsize=16,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    PLOTS_DIR, _, _ = _get_paths()
    save_path = os.path.join(PLOTS_DIR, "comparison_metrics.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Compare] Metrics comparison saved -> {save_path}")


def plot_loss_curves_comparison(histories: Dict[str, dict]) -> None:
    """Overlay training loss curves from all models."""
    models = [m for m in MODEL_ORDER if m in histories]
    if len(models) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Training loss
    ax = axes[0]
    for model_name in models:
        h = histories[model_name]
        losses = h.get("train_total", [])
        if losses:
            epochs = range(1, len(losses) + 1)
            ax.plot(epochs, losses,
                    label=MODEL_LABELS.get(model_name, model_name),
                    color=MODEL_COLOURS.get(model_name, "#999"),
                    linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Training Loss Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    # Validation loss
    ax = axes[1]
    for model_name in models:
        h = histories[model_name]
        losses = h.get("val_total", [])
        if losses:
            epochs = range(1, len(losses) + 1)
            ax.plot(epochs, losses,
                    label=MODEL_LABELS.get(model_name, model_name),
                    color=MODEL_COLOURS.get(model_name, "#999"),
                    linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Validation Loss Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.suptitle("Loss Curve Comparison — All Models", fontsize=16,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    PLOTS_DIR, _, _ = _get_paths()
    save_path = os.path.join(PLOTS_DIR, "comparison_loss_curves.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Compare] Loss curves comparison saved -> {save_path}")


def plot_predicted_vs_actual_comparison(
    predictions: Dict[str, pd.DataFrame],
) -> None:
    """Side-by-side predicted vs actual scatter plots."""
    models = [m for m in MODEL_ORDER if m in predictions]
    if len(models) < 2:
        return

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        df = predictions[model_name]
        y_true = df["actual_magnitude"].values
        y_pred = df["predicted_magnitude"].values

        ax.scatter(y_true, y_pred, alpha=0.5, s=15,
                   color=MODEL_COLOURS.get(model_name, "#999"),
                   edgecolors="none")

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val - margin, max_val + margin],
                [min_val - margin, max_val + margin],
                "r--", linewidth=1.5, alpha=0.7)

        ax.set_xlabel("Actual Magnitude", fontsize=12)
        ax.set_ylabel("Predicted Magnitude", fontsize=12)
        ax.set_title(MODEL_LABELS.get(model_name, model_name),
                     fontsize=13, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    fig.suptitle("Predicted vs Actual — All Models", fontsize=16,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    PLOTS_DIR, _, _ = _get_paths()
    save_path = os.path.join(PLOTS_DIR, "comparison_predicted_vs_actual.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Compare] Predicted vs Actual comparison saved -> {save_path}")


def plot_error_distribution_comparison(
    predictions: Dict[str, pd.DataFrame],
) -> None:
    """Overlay error distribution histograms for all models."""
    models = [m for m in MODEL_ORDER if m in predictions]
    if len(models) < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    for model_name in models:
        df = predictions[model_name]
        errors = df["actual_magnitude"].values - df["predicted_magnitude"].values
        ax.hist(errors, bins=40, alpha=0.5,
                color=MODEL_COLOURS.get(model_name, "#999"),
                label=MODEL_LABELS.get(model_name, model_name),
                edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Prediction Error (Actual - Predicted)", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.set_title("Error Distribution Comparison", fontsize=15,
                 fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    PLOTS_DIR, _, _ = _get_paths()
    save_path = os.path.join(PLOTS_DIR, "comparison_error_distribution.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Compare] Error distribution comparison saved -> {save_path}")


def plot_residuals_comparison(
    predictions: Dict[str, pd.DataFrame],
) -> None:
    """Generate residual plots (error vs actual magnitude) for all models."""
    models = [m for m in MODEL_ORDER if m in predictions]
    if len(models) < 2:
        return

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        df = predictions[model_name]
        y_true = df["actual_magnitude"].values
        errors = y_true - df["predicted_magnitude"].values

        ax.scatter(y_true, errors, alpha=0.5, s=15,
                   color=MODEL_COLOURS.get(model_name, "#999"),
                   edgecolors="none")
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

        ax.set_xlabel("Actual Magnitude", fontsize=12)
        ax.set_ylabel("Residual", fontsize=12)
        ax.set_title(MODEL_LABELS.get(model_name, model_name),
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    fig.suptitle("Residual Plots — All Models", fontsize=16,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    PLOTS_DIR, _, _ = _get_paths()
    save_path = os.path.join(PLOTS_DIR, "comparison_residuals.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Compare] Residual plots saved -> {save_path}")


def plot_training_time_comparison(metrics: Dict[str, dict]) -> None:
    """Bar chart comparing training times and parameter counts."""
    models = [m for m in MODEL_ORDER if m in metrics]
    if len(models) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training time
    ax = axes[0]
    times = [metrics[m].get("training_time_seconds", 0) for m in models]
    colours = [MODEL_COLOURS.get(m, "#999") for m in models]
    labels = [MODEL_LABELS.get(m, m) for m in models]

    bars = ax.bar(labels, times, color=colours, edgecolor="white",
                  linewidth=1.5, width=0.6)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}s", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_title("Training Time (seconds)", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=10, axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")

    # Parameter count
    ax = axes[1]
    params = [metrics[m].get("num_parameters", 0) for m in models]
    bars = ax.bar(labels, params, color=colours, edgecolor="white",
                  linewidth=1.5, width=0.6)
    for bar, val in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    ax.set_title("Number of Parameters", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=10, axis="x", rotation=15)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Computational Cost Comparison", fontsize=16,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    PLOTS_DIR, _, _ = _get_paths()
    save_path = os.path.join(PLOTS_DIR, "comparison_training_cost.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Compare] Training cost comparison saved -> {save_path}")


def generate_all_comparisons() -> None:
    """Generate all comparison visualisations from saved outputs."""
    print("\n" + "=" * 65)
    print("  GENERATING COMPARISON VISUALISATIONS")
    print("=" * 65 + "\n")

    metrics = _load_available_metrics()
    histories = _load_available_histories()
    predictions = _load_available_predictions()

    print(f"[Compare] Found {len(metrics)} model(s): "
          f"{', '.join(metrics.keys())}")

    if len(metrics) < 2:
        print("[Compare] Need at least 2 models for comparison. Skipping.")
        return

    plot_metrics_comparison(metrics)
    plot_loss_curves_comparison(histories)
    plot_predicted_vs_actual_comparison(predictions)
    plot_error_distribution_comparison(predictions)
    plot_residuals_comparison(predictions)
    plot_training_time_comparison(metrics)

    # Print final comparison table
    print("\n" + "=" * 90)
    print("  FINAL COMPARISON TABLE")
    print("=" * 90)
    header = (
        f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R²':>10} "
        f"{'Time (s)':>10} {'Params':>10}"
    )
    print(header)
    print("-" * 90)
    for name in MODEL_ORDER:
        if name in metrics:
            m = metrics[name]
            print(
                f"{MODEL_LABELS.get(name, name):<25} "
                f"{m.get('MAE', 0):>10.6f} "
                f"{m.get('RMSE', 0):>10.6f} "
                f"{m.get('R2', 0):>10.6f} "
                f"{m.get('training_time_seconds', 0):>10.2f} "
                f"{m.get('num_parameters', 0):>10,}"
            )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    generate_all_comparisons()
