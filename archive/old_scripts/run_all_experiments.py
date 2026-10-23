"""
run_all_experiments.py
----------------------
Master script that trains and evaluates ALL four model variants
sequentially and generates a comprehensive comparison.

Models trained:
    1. Baseline MLP      — No physics, no attention, no uncertainty
    2. Basic PINN        — Physics constraints only
    3. Attention PINN    — Physics + attention
    4. Full PINN         — Physics + attention + uncertainty

After all models are trained, this script automatically calls
compare.py to generate side-by-side comparisons.

Usage:
    python run_all_experiments.py
"""

import os
import sys
import time
import json
import yaml
import numpy as np
import torch

from src.data_loader import load_dataset
from src.preprocess import (
    check_missing_values,
    handle_missing_values,
    split_data,
    scale_features,
    to_tensors,
)
from src.model import BaselineMLP
from src.pinn_model import EarthquakePINN
from src.train import train_model
from src.pinn_train import train_pinn, get_magnitude_prediction
from src.evaluate import compute_metrics
from src.utils import (
    setup_directories,
    save_model,
    save_scaler,
    save_predictions,
    save_metrics,
    SAVED_MODELS_DIR,
    PLOTS_DIR,
    PREDICTIONS_DIR,
    METRICS_DIR,
)


def load_config(path: str = "config/config.yaml") -> dict:
    """Load the YAML configuration."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_data(config: dict) -> dict:
    """Load and preprocess the dataset once for all experiments.

    Returns a dictionary with all tensors and raw physics features
    needed by every model variant.
    """
    dataset_path = config["dataset"]["path"]
    test_size = config["dataset"]["test_size"]
    random_state = config["dataset"]["random_state"]
    energy_feature = config["physics"]["energy_feature"]
    frequency_feature = config["physics"]["frequency_feature"]

    print("\n" + "=" * 65)
    print("  DATA PREPARATION (shared across all experiments)")
    print("=" * 65 + "\n")

    # Load
    X, y = load_dataset(dataset_path)
    check_missing_values(X)
    X = handle_missing_values(X)
    y = y.loc[X.index]

    # Split
    from sklearn.model_selection import train_test_split
    X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
    )

    # Extract raw physics features BEFORE scaling
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

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train_df, X_test_df,
    )

    # Tensorise
    X_train_t, X_test_t, y_train_t, y_test_t = to_tensors(
        X_train_scaled, X_test_scaled, y_train_s, y_test_s,
    )

    feature_names = X.columns.tolist()

    return {
        "X_train_t": X_train_t,
        "X_test_t": X_test_t,
        "y_train_t": y_train_t,
        "y_test_t": y_test_t,
        "scaler": scaler,
        "feature_names": feature_names,
        "input_dim": X_train_t.shape[1],
        "raw_energy_train": raw_energy_train,
        "raw_freq_train": raw_freq_train,
        "raw_energy_test": raw_energy_test,
        "raw_freq_test": raw_freq_test,
    }


def run_baseline(data: dict, config: dict) -> dict:
    """Train and evaluate the Baseline MLP."""
    model_name = "baseline_mlp"
    print("\n" + "#" * 65)
    print(f"  EXPERIMENT 1/4: {model_name.upper()}")
    print("#" * 65)

    model = BaselineMLP(input_dim=data["input_dim"])
    print(f"[Model] {model_name} — Parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    start = time.time()
    model, train_losses, val_losses = train_model(
        model=model,
        X_train=data["X_train_t"],
        y_train=data["y_train_t"],
        X_val=data["X_test_t"],
        y_val=data["y_test_t"],
        epochs=config["training"]["epochs"],
        lr=config["training"]["learning_rate"],
        print_every=config["training"]["print_every"],
    )
    elapsed = time.time() - start

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(data["X_test_t"]).numpy().flatten()
    y_true = data["y_test_t"].numpy().flatten()

    metrics = compute_metrics(y_true, y_pred)
    metrics["training_time_seconds"] = round(elapsed, 2)
    metrics["num_parameters"] = sum(p.numel() for p in model.parameters())

    # Save
    save_model(model, os.path.join(SAVED_MODELS_DIR, f"{model_name}.pth"))
    save_scaler(data["scaler"],
                os.path.join(SAVED_MODELS_DIR, f"{model_name}_scaler.pkl"))
    save_predictions(y_true, y_pred,
                     os.path.join(PREDICTIONS_DIR, f"{model_name}_predictions.csv"))
    save_metrics(metrics, os.path.join(METRICS_DIR, f"{model_name}_metrics.json"))

    # Save loss history
    history = {"train_total": train_losses, "val_total": val_losses}
    with open(os.path.join(METRICS_DIR, f"{model_name}_loss_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Generate baseline plots with consistent naming
    from src.evaluate import (
        plot_loss_curves, plot_predicted_vs_actual, plot_error_distribution,
    )
    plot_loss_curves(train_losses, val_losses,
                     os.path.join(PLOTS_DIR, f"{model_name}_loss_curves.png"))
    plot_predicted_vs_actual(y_true, y_pred,
                             os.path.join(PLOTS_DIR, f"{model_name}_predicted_vs_actual.png"))
    plot_error_distribution(y_true, y_pred,
                             os.path.join(PLOTS_DIR, f"{model_name}_error_distribution.png"))

    return metrics


def run_pinn_variant(
    data: dict,
    config: dict,
    model_name: str,
    use_physics: bool,
    use_attention: bool,
    use_uncertainty: bool,
) -> dict:
    """Train and evaluate a specific PINN variant."""
    print("\n" + "#" * 65)
    variant_label = model_name.upper().replace("_", " ")
    print(f"  EXPERIMENT: {variant_label}")
    print("#" * 65)

    model = EarthquakePINN(
        input_dim=data["input_dim"],
        hidden_layers=config["model"]["hidden_layers"],
        use_attention=use_attention,
        use_uncertainty=use_uncertainty,
    )
    print(f"[Model] {model.get_config_string()}")
    print(f"[Model] Parameters: {model.count_parameters():,}")

    model, history = train_pinn(
        model=model,
        X_train=data["X_train_t"],
        y_train=data["y_train_t"],
        X_val=data["X_test_t"],
        y_val=data["y_test_t"],
        raw_energy_train=data["raw_energy_train"],
        raw_frequency_train=data["raw_freq_train"],
        raw_energy_val=data["raw_energy_test"],
        raw_frequency_val=data["raw_freq_test"],
        epochs=config["training"]["epochs"],
        lr=config["training"]["learning_rate"],
        lambda_energy=config["physics"]["lambda_energy"],
        lambda_frequency=config["physics"]["lambda_frequency"],
        energy_coeff_a=config["physics"]["energy_coeff_a"],
        energy_coeff_b=config["physics"]["energy_coeff_b"],
        epsilon=config["physics"]["epsilon"],
        use_uncertainty=use_uncertainty,
        use_physics=use_physics,
        print_every=config["training"]["print_every"],
        model_name=model_name,
    )

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_output = model(data["X_test_t"])

    if use_uncertainty:
        y_pred = test_output[:, 0].numpy().flatten()
        pred_uncertainty = np.exp(test_output[:, 1].numpy().flatten())
    else:
        y_pred = test_output.numpy().flatten()
        pred_uncertainty = None

    y_true = data["y_test_t"].numpy().flatten()
    metrics = compute_metrics(y_true, y_pred)

    training_time = history["training_time_seconds"][0]
    metrics["training_time_seconds"] = round(training_time, 2)
    metrics["num_parameters"] = model.count_parameters()

    # Save
    save_model(model, os.path.join(SAVED_MODELS_DIR, f"{model_name}.pth"))
    save_predictions(y_true, y_pred,
                     os.path.join(PREDICTIONS_DIR, f"{model_name}_predictions.csv"))
    save_metrics(metrics, os.path.join(METRICS_DIR, f"{model_name}_metrics.json"))

    with open(os.path.join(METRICS_DIR, f"{model_name}_loss_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Generate model-specific plots
    from src.evaluate import plot_loss_curves, plot_predicted_vs_actual, plot_error_distribution

    plot_loss_curves(history["train_total"], history["val_total"],
                     os.path.join(PLOTS_DIR, f"{model_name}_loss_curves.png"))
    plot_predicted_vs_actual(y_true, y_pred,
                             os.path.join(PLOTS_DIR, f"{model_name}_predicted_vs_actual.png"))
    plot_error_distribution(y_true, y_pred,
                             os.path.join(PLOTS_DIR, f"{model_name}_error_distribution.png"))

    # Attention weights plot
    if use_attention:
        from main_pinn import _plot_attention_weights
        _plot_attention_weights(
            model, data["X_train_t"], data["feature_names"], model_name,
            os.path.join(PLOTS_DIR, f"{model_name}_attention_weights.png"),
        )

    # Uncertainty plot
    if use_uncertainty and pred_uncertainty is not None:
        from main_pinn import _plot_uncertainty
        _plot_uncertainty(
            y_pred, pred_uncertainty, model_name,
            os.path.join(PLOTS_DIR, f"{model_name}_uncertainty.png"),
        )

    # Physics loss plot
    if use_physics:
        from main_pinn import _plot_physics_losses
        _plot_physics_losses(
            history, model_name,
            os.path.join(PLOTS_DIR, f"{model_name}_physics_losses.png"),
        )

    return metrics


def print_comparison_table(all_metrics: dict) -> None:
    """Print a formatted comparison table of all experiments."""
    print("\n" + "=" * 90)
    print("  COMPARISON TABLE — ALL EXPERIMENTS")
    print("=" * 90)

    header = (
        f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R²':>10} "
        f"{'Time (s)':>10} {'Params':>10}"
    )
    print(header)
    print("-" * 90)

    for name, m in all_metrics.items():
        row = (
            f"{name:<25} "
            f"{m.get('MAE', 'N/A'):>10.6f} "
            f"{m.get('RMSE', 'N/A'):>10.6f} "
            f"{m.get('R2', 'N/A'):>10.6f} "
            f"{m.get('training_time_seconds', 'N/A'):>10.2f} "
            f"{m.get('num_parameters', 'N/A'):>10,}"
        )
        print(row)

    print("=" * 90 + "\n")


def main() -> None:
    """Run all four experiments and generate comparison."""
    total_start = time.time()

    print("\n" + "█" * 65)
    print("  COMPLETE EXPERIMENT SUITE")
    print("  Physics-Informed Neural Network for Earthquake Prediction")
    print("█" * 65 + "\n")

    config = load_config()
    setup_directories()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Prepare data once
    data = prepare_data(config)

    # Store all metrics for comparison
    all_metrics = {}

    # -- Experiment 1: Baseline MLP ----------------------------------------
    all_metrics["baseline_mlp"] = run_baseline(data, config)

    # -- Experiment 2: Basic PINN ------------------------------------------
    all_metrics["basic_pinn"] = run_pinn_variant(
        data, config,
        model_name="basic_pinn",
        use_physics=True,
        use_attention=False,
        use_uncertainty=False,
    )

    # -- Experiment 3: Attention PINN --------------------------------------
    all_metrics["attention_pinn"] = run_pinn_variant(
        data, config,
        model_name="attention_pinn",
        use_physics=True,
        use_attention=True,
        use_uncertainty=False,
    )

    # -- Experiment 4: Full PINN (Attention + Uncertainty) -----------------
    all_metrics["full_pinn"] = run_pinn_variant(
        data, config,
        model_name="full_pinn",
        use_physics=True,
        use_attention=True,
        use_uncertainty=True,
    )

    # -- Comparison Table --------------------------------------------------
    print_comparison_table(all_metrics)

    # Save comparison metrics
    with open(os.path.join(METRICS_DIR, "all_experiments_comparison.json"), "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"[Results] Comparison saved -> "
          f"{os.path.join(METRICS_DIR, 'all_experiments_comparison.json')}")

    total_elapsed = time.time() - total_start
    print(f"\n[Total] All experiments completed in {total_elapsed:.1f} seconds.")

    # -- Generate comparison visualisations --------------------------------
    print("\n[Compare] Generating comparison visualisations...")
    from compare import generate_all_comparisons
    generate_all_comparisons()

    print("\n" + "█" * 65)
    print("  ALL EXPERIMENTS AND COMPARISONS COMPLETE")
    print("█" * 65 + "\n")


if __name__ == "__main__":
    main()
