"""
evaluate.py
-----------
Evaluation metrics and visualisation for the baseline model.
Generates MAE, RMSE, R-squared scores and publication-quality plots.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression evaluation metrics.

    Args:
        y_true: Ground truth magnitude values.
        y_pred: Predicted magnitude values.

    Returns:
        Dictionary with MAE, RMSE, and R-squared scores.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    metrics = {
        "MAE":  round(float(mae),  6),
        "RMSE": round(float(rmse), 6),
        "R2":   round(float(r2),   6),
    }

    print("\n" + "=" * 40)
    print("EVALUATION METRICS")
    print("=" * 40)
    print(f"  MAE  : {metrics['MAE']:.6f}")
    print(f"  RMSE : {metrics['RMSE']:.6f}")
    print(f"  R2   : {metrics['R2']:.6f}")
    print("=" * 40 + "\n")

    return metrics


def compute_mean_baseline(
    y_train: np.ndarray,
    y_test:  np.ndarray,
) -> Dict[str, float]:
    """Compute the trivial mean-prediction baseline.

    Predicts the training-set mean for every test sample.
    The model must outperform this to demonstrate any predictive signal.

    Args:
        y_train: Training target values (original magnitude scale).
        y_test:  Test target values (original magnitude scale).

    Returns:
        Dict with MAE, RMSE, R², and the constant prediction value.
    """
    mean_val   = float(np.mean(y_train))
    y_baseline = np.full_like(y_test, mean_val, dtype=float)

    mae  = mean_absolute_error(y_test, y_baseline)
    rmse = np.sqrt(mean_squared_error(y_test, y_baseline))
    r2   = r2_score(y_test, y_baseline)

    metrics = {
        "MAE":              round(float(mae),      6),
        "RMSE":             round(float(rmse),     6),
        "R2":               round(float(r2),       6),
        "mean_prediction":  round(mean_val,        6),
    }

    print("\n" + "=" * 50)
    print("MEAN BASELINE (predict training mean for all test samples)")
    print("=" * 50)
    print(f"  Constant prediction : {mean_val:.4f}")
    print(f"  MAE  : {metrics['MAE']:.6f}")
    print(f"  RMSE : {metrics['RMSE']:.6f}")
    print(f"  R²   : {metrics['R2']:.6f}")
    print("  (PINN must beat this to show meaningful signal)")
    print("=" * 50 + "\n")

    return metrics



def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str,
) -> None:
    """Plot training and validation loss curves.

    Args:
        train_losses: List of training loss values per epoch.
        val_losses: List of validation loss values per epoch.
        save_path: File path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Training Loss", color="#2196F3", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Validation Loss", color="#FF5722", linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("MSE Loss", fontsize=13)
    ax.set_title("Training and Validation Loss Curves", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Loss curve saved -> {save_path}")


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> None:
    """Plot predicted vs actual magnitude scatter plot.

    Args:
        y_true: Ground truth magnitudes.
        y_pred: Predicted magnitudes.
        save_path: File path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.5, s=20, color="#1976D2", edgecolors="none")

    # Plot ideal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    ax.plot(
        [min_val - margin, max_val + margin],
        [min_val - margin, max_val + margin],
        "r--",
        linewidth=1.5,
        label="Ideal (y = x)",
    )

    ax.set_xlabel("Actual Magnitude", fontsize=13)
    ax.set_ylabel("Predicted Magnitude", fontsize=13)
    ax.set_title("Predicted vs Actual Magnitude", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Predicted vs Actual plot saved -> {save_path}")


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> None:
    """Plot the distribution of prediction errors.

    Args:
        y_true: Ground truth magnitudes.
        y_pred: Predicted magnitudes.
        save_path: File path to save the plot.
    """
    errors = y_true - y_pred

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(errors, bins=40, color="#4CAF50", edgecolor="white", alpha=0.85)

    # Add mean and std annotations
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    ax.axvline(mean_err, color="#D32F2F", linestyle="--", linewidth=1.5, label=f"Mean: {mean_err:.4f}")
    ax.axvline(mean_err + std_err, color="#FF9800", linestyle=":", linewidth=1.2, label=f"+/-1 Std: {std_err:.4f}")
    ax.axvline(mean_err - std_err, color="#FF9800", linestyle=":", linewidth=1.2)

    ax.set_xlabel("Prediction Error (Actual - Predicted)", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.set_title("Prediction Error Distribution", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Error distribution plot saved -> {save_path}")
