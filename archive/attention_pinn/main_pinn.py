"""
main_pinn.py
------------
Unified entry point for training any PINN variant. Reads configuration
from config/config.yaml and supports all four model types:

    1. basic_pinn       — Physics constraints only
    2. attention_pinn   — Attention + physics constraints
    3. full_pinn        — Attention + uncertainty + physics constraints

The model variant is controlled by the 'attention.enabled' and
'uncertainty.enabled' flags in the YAML config, or by passing
an experiment preset name via the command line.

Usage:
    # Use default config settings:
    python main_pinn.py

    # Override with a specific experiment preset:
    python main_pinn.py --experiment basic_pinn
    python main_pinn.py --experiment attention_pinn
    python main_pinn.py --experiment full_pinn
"""

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch

from src.data_loader import load_dataset
from src.preprocess import (
    check_missing_values,
    handle_missing_values,
    split_data,
    scale_features,
    to_tensors,
)
from src.pinn_model import EarthquakePINN
from src.pinn_train import train_pinn, get_magnitude_prediction
from src.evaluate import (
    compute_metrics,
    plot_loss_curves,
    plot_predicted_vs_actual,
    plot_error_distribution,
)
from src.utils import (
    setup_directories,
    save_model,
    save_scaler,
    save_predictions,
    save_metrics,
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary of configuration values.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def apply_experiment_preset(config: dict, preset_name: str) -> dict:
    """Override config values with an experiment preset.

    Args:
        config: Base configuration dictionary.
        preset_name: Name of the experiment preset.

    Returns:
        Modified configuration dictionary.
    """
    presets = config.get("experiments", {})
    if preset_name not in presets:
        print(f"[Config] WARNING: Preset '{preset_name}' not found. "
              f"Available: {list(presets.keys())}")
        return config

    preset = presets[preset_name]
    print(f"[Config] Applying experiment preset: '{preset_name}'")

    config["physics"]["enabled"] = preset.get("physics_enabled",
                                               config["physics"]["enabled"])
    config["attention"]["enabled"] = preset.get("attention_enabled",
                                                 config["attention"]["enabled"])
    config["uncertainty"]["enabled"] = preset.get("uncertainty_enabled",
                                                   config["uncertainty"]["enabled"])
    config["_model_name"] = preset.get("model_name", preset_name)

    return config


def extract_raw_physics_features(
    X_df: pd.DataFrame,
    indices: np.ndarray,
    energy_col: str,
    frequency_col: str,
) -> tuple:
    """Extract raw (un-normalised) physics features for constraints.

    The physics loss requires un-normalised feature values to enforce
    seismological scaling laws. This function extracts them before
    StandardScaler normalisation.

    Args:
        X_df: Feature DataFrame (before scaling).
        indices: Row indices for the desired split (train or test).
        energy_col: Column name for energy feature.
        frequency_col: Column name for frequency feature.

    Returns:
        Tuple of (raw_energy_tensor, raw_frequency_tensor).
    """
    raw_energy = torch.tensor(
        X_df.iloc[indices][energy_col].values,
        dtype=torch.float32,
    )
    raw_frequency = torch.tensor(
        X_df.iloc[indices][frequency_col].values,
        dtype=torch.float32,
    )
    return raw_energy, raw_frequency


def run_pinn_pipeline(config: dict) -> dict:
    """Execute the complete PINN training and evaluation pipeline.

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary of evaluation metrics.
    """
    # -- Extract configuration values --------------------------------------
    dataset_path = config["dataset"]["path"]
    test_size = config["dataset"]["test_size"]
    random_state = config["dataset"]["random_state"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    print_every = config["training"]["print_every"]
    hidden_layers = config["model"]["hidden_layers"]

    use_physics = config["physics"]["enabled"]
    lambda_energy = config["physics"]["lambda_energy"]
    lambda_frequency = config["physics"]["lambda_frequency"]
    energy_feature = config["physics"]["energy_feature"]
    frequency_feature = config["physics"]["frequency_feature"]
    energy_a = config["physics"]["energy_coeff_a"]
    energy_b = config["physics"]["energy_coeff_b"]
    epsilon = config["physics"]["epsilon"]

    use_attention = config["attention"]["enabled"]
    use_uncertainty = config["uncertainty"]["enabled"]

    model_name = config.get("_model_name", "pinn")

    # Output paths
    saved_models_dir = config["output"]["saved_models_dir"]
    plots_dir = config["output"]["plots_dir"]
    predictions_dir = config["output"]["predictions_dir"]
    metrics_dir = config["output"]["metrics_dir"]

    # -- Header ------------------------------------------------------------
    print("\n" + "=" * 65)
    variant = model_name.upper().replace("_", " ")
    print(f"  {variant} — EARTHQUAKE MAGNITUDE PREDICTION")
    print("=" * 65 + "\n")

    # -- 1. Setup directories ----------------------------------------------
    setup_directories()

    # -- 2. Load dataset ---------------------------------------------------
    X, y = load_dataset(dataset_path)

    # -- 3. Handle missing values ------------------------------------------
    check_missing_values(X)
    X = handle_missing_values(X)
    y = y.loc[X.index]

    # -- 4. Train/test split -----------------------------------------------
    from sklearn.model_selection import train_test_split
    X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    # -- 5. Extract raw physics features BEFORE scaling --------------------
    raw_energy_train = torch.tensor(
        X_train_df[energy_feature].values, dtype=torch.float32,
    )
    raw_freq_train = torch.tensor(
        X_train_df[frequency_feature].values, dtype=torch.float32,
    )
    raw_energy_test = torch.tensor(
        X_test_df[energy_feature].values, dtype=torch.float32,
    )
    raw_freq_test = torch.tensor(
        X_test_df[frequency_feature].values, dtype=torch.float32,
    )

    print(f"[Physics] Raw energy range: {raw_energy_train.min():.4f} – "
          f"{raw_energy_train.max():.4f}")
    print(f"[Physics] Raw frequency range: {raw_freq_train.min():.4f} – "
          f"{raw_freq_train.max():.4f}")

    # -- 6. Feature scaling ------------------------------------------------
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train_df, X_test_df,
    )

    # -- 7. Convert to tensors ---------------------------------------------
    X_train_t, X_test_t, y_train_t, y_test_t = to_tensors(
        X_train_scaled, X_test_scaled, y_train_s, y_test_s,
    )

    # -- 8. Initialise model -----------------------------------------------
    input_dim = X_train_t.shape[1]
    model = EarthquakePINN(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        use_attention=use_attention,
        use_uncertainty=use_uncertainty,
    )

    print(f"\n[Model] {model.get_config_string()}")
    print(f"[Model] Total parameters: {model.count_parameters():,}")

    # -- 9. Train ----------------------------------------------------------
    model, history = train_pinn(
        model=model,
        X_train=X_train_t,
        y_train=y_train_t,
        X_val=X_test_t,
        y_val=y_test_t,
        raw_energy_train=raw_energy_train,
        raw_frequency_train=raw_freq_train,
        raw_energy_val=raw_energy_test,
        raw_frequency_val=raw_freq_test,
        epochs=epochs,
        lr=lr,
        lambda_energy=lambda_energy,
        lambda_frequency=lambda_frequency,
        energy_coeff_a=energy_a,
        energy_coeff_b=energy_b,
        epsilon=epsilon,
        use_uncertainty=use_uncertainty,
        use_physics=use_physics,
        print_every=print_every,
        model_name=model_name,
    )

    # -- 10. Generate predictions ------------------------------------------
    model.eval()
    with torch.no_grad():
        test_output = model(X_test_t)

    if use_uncertainty:
        y_pred = test_output[:, 0].numpy().flatten()
        uncertainty = np.exp(test_output[:, 1].numpy().flatten())
    else:
        y_pred = test_output.numpy().flatten()
        uncertainty = None

    y_true = y_test_t.numpy().flatten()

    # -- 11. Evaluate ------------------------------------------------------
    metrics = compute_metrics(y_true, y_pred)

    # Add training time to metrics
    training_time = history["training_time_seconds"][0]
    metrics["training_time_seconds"] = round(training_time, 2)
    metrics["num_parameters"] = model.count_parameters()

    # -- 12. Generate plots ------------------------------------------------
    plot_loss_curves(
        history["train_total"],
        history["val_total"],
        save_path=os.path.join(plots_dir, f"{model_name}_loss_curves.png"),
    )
    plot_predicted_vs_actual(
        y_true, y_pred,
        save_path=os.path.join(plots_dir, f"{model_name}_predicted_vs_actual.png"),
    )
    plot_error_distribution(
        y_true, y_pred,
        save_path=os.path.join(plots_dir, f"{model_name}_error_distribution.png"),
    )

    # -- 13. Physics loss curves (if physics enabled) ----------------------
    if use_physics:
        _plot_physics_losses(
            history, model_name,
            save_path=os.path.join(plots_dir, f"{model_name}_physics_losses.png"),
        )

    # -- 14. Uncertainty plot (if uncertainty enabled) ----------------------
    if use_uncertainty and uncertainty is not None:
        _plot_uncertainty(
            y_pred, uncertainty, model_name,
            save_path=os.path.join(plots_dir, f"{model_name}_uncertainty.png"),
        )

    # -- 15. Attention weights (if attention enabled) ----------------------
    if use_attention:
        _plot_attention_weights(
            model, X_train_t, X.columns.tolist(), model_name,
            save_path=os.path.join(plots_dir, f"{model_name}_attention_weights.png"),
        )

    # -- 16. Save outputs --------------------------------------------------
    save_model(model, os.path.join(saved_models_dir, f"{model_name}.pth"))
    save_scaler(scaler, os.path.join(saved_models_dir, f"{model_name}_scaler.pkl"))
    save_predictions(y_true, y_pred, os.path.join(predictions_dir,
                                                    f"{model_name}_predictions.csv"))
    save_metrics(metrics, os.path.join(metrics_dir, f"{model_name}_metrics.json"))

    # Also save loss history
    _save_loss_history(history, os.path.join(metrics_dir,
                                              f"{model_name}_loss_history.json"))

    # -- Done --------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"  PIPELINE COMPLETE — {model_name.upper()}")
    print("=" * 65)
    print(f"  Model       -> {os.path.join(saved_models_dir, model_name + '.pth')}")
    print(f"  Plots       -> {plots_dir}/")
    print(f"  Predictions -> {os.path.join(predictions_dir, model_name + '_predictions.csv')}")
    print(f"  Metrics     -> {os.path.join(metrics_dir, model_name + '_metrics.json')}")
    print("=" * 65 + "\n")

    return metrics


# ===========================================================================
# Private helper functions for visualisation
# ===========================================================================

def _plot_physics_losses(
    history: dict,
    model_name: str,
    save_path: str,
) -> None:
    """Plot individual physics loss components over training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(history["train_energy"]) + 1)

    ax.plot(epochs, history["train_energy"], label="Energy Loss",
            color="#E91E63", linewidth=1.5)
    ax.plot(epochs, history["train_frequency"], label="Frequency Loss",
            color="#9C27B0", linewidth=1.5)
    ax.plot(epochs, history["train_physics"], label="Total Physics Loss",
            color="#FF5722", linewidth=2.0, linestyle="--")

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title(f"Physics Loss Components — {model_name}", fontsize=15,
                 fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Physics loss plot saved -> {save_path}")


def _plot_uncertainty(
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    model_name: str,
    save_path: str,
) -> None:
    """Plot prediction uncertainty vs predicted magnitude."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(y_pred, uncertainty, alpha=0.5, s=20,
                         c=uncertainty, cmap="YlOrRd", edgecolors="none")
    plt.colorbar(scatter, ax=ax, label="Uncertainty (σ)")

    ax.set_xlabel("Predicted Magnitude", fontsize=13)
    ax.set_ylabel("Prediction Uncertainty (σ)", fontsize=13)
    ax.set_title(f"Prediction Uncertainty — {model_name}", fontsize=15,
                 fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Uncertainty plot saved -> {save_path}")


def _plot_attention_weights(
    model: torch.nn.Module,
    X: torch.Tensor,
    feature_names: list,
    model_name: str,
    save_path: str,
) -> None:
    """Plot mean attention weights as a horizontal bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    weights = model.get_attention_weights(X)
    if weights is None:
        return

    mean_weights = weights.mean(dim=0).numpy()

    # Sort by importance
    sorted_indices = np.argsort(mean_weights)
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_weights = mean_weights[sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn(sorted_weights / sorted_weights.max())
    ax.barh(sorted_names, sorted_weights, color=colors, edgecolor="white",
            linewidth=0.5)

    ax.set_xlabel("Attention Weight", fontsize=13)
    ax.set_title(f"Feature Importance (Attention) — {model_name}",
                 fontsize=15, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Evaluate] Attention weights plot saved -> {save_path}")


def _save_loss_history(history: dict, path: str) -> None:
    """Save loss history to a JSON file."""
    import json
    with open(path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"[Utils] Loss history saved -> {path}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    """Parse arguments, load config, and run the PINN pipeline."""
    parser = argparse.ArgumentParser(
        description="Train a PINN variant for earthquake magnitude prediction.",
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Experiment preset name from config.yaml "
             "(e.g., basic_pinn, attention_pinn, full_pinn).",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to YAML configuration file.",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply experiment preset if specified
    if args.experiment:
        config = apply_experiment_preset(config, args.experiment)

    # Run pipeline
    run_pinn_pipeline(config)


if __name__ == "__main__":
    main()
