"""
main.py
-------
Entry point for the Baseline MLP earthquake magnitude prediction pipeline.

Orchestrates:
    1. Directory setup
    2. Data loading
    3. Preprocessing (missing values, split, scale, tensorize)
    4. Model initialisation
    5. Training (300 epochs, MSE, Adam, lr=0.001)
    6. Evaluation (MAE, RMSE, R-squared)
    7. Plot generation (loss curves, scatter, error distribution)
    8. Saving outputs (model, scaler, predictions, metrics)
"""

import os
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
from src.train import train_model
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
    SAVED_MODELS_DIR,
    PLOTS_DIR,
    PREDICTIONS_DIR,
    METRICS_DIR,
)


# -- Configuration -----------------------------------------------------
DATASET_PATH = "earthquake_29_features.csv"
EPOCHS = 300
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
RANDOM_STATE = 42
PRINT_EVERY = 20


def main() -> None:
    """Run the complete baseline MLP pipeline."""

    print("\n" + "=" * 60)
    print("  BASELINE MLP - EARTHQUAKE MAGNITUDE PREDICTION")
    print("=" * 60 + "\n")

    # -- 1. Setup directories ------------------------------------------
    setup_directories()

    # -- 2. Load dataset -----------------------------------------------
    X, y = load_dataset(DATASET_PATH)

    # -- 3. Check and handle missing values ----------------------------
    check_missing_values(X)
    X = handle_missing_values(X)
    # Re-align y with X after potential row drops
    y = y.loc[X.index]

    # -- 4. Train/test split -------------------------------------------
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # -- 5. Feature scaling --------------------------------------------
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # -- 6. Convert to tensors -----------------------------------------
    X_train_t, X_test_t, y_train_t, y_test_t = to_tensors(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    # -- 7. Initialise model -------------------------------------------
    input_dim = X_train_t.shape[1]
    model = BaselineMLP(input_dim=input_dim)
    print(f"\n[Model] BaselineMLP initialised (input_dim={input_dim})")
    print(f"[Model] Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -- 8. Train ------------------------------------------------------
    model, train_losses, val_losses = train_model(
        model=model,
        X_train=X_train_t,
        y_train=y_train_t,
        X_val=X_test_t,
        y_val=y_test_t,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        print_every=PRINT_EVERY,
    )

    # -- 9. Generate predictions ---------------------------------------
    model.eval()
    with torch.no_grad():
        y_pred_t = model(X_test_t)

    y_true = y_test_t.numpy().flatten()
    y_pred = y_pred_t.numpy().flatten()

    # -- 10. Evaluate ------------------------------------------------------
    metrics = compute_metrics(y_true, y_pred)

    # -- 11. Generate plots ------------------------------------------------
    plot_loss_curves(
        train_losses,
        val_losses,
        save_path=os.path.join(PLOTS_DIR, "loss_curves.png"),
    )
    plot_predicted_vs_actual(
        y_true,
        y_pred,
        save_path=os.path.join(PLOTS_DIR, "predicted_vs_actual.png"),
    )
    plot_error_distribution(
        y_true,
        y_pred,
        save_path=os.path.join(PLOTS_DIR, "error_distribution.png"),
    )

    # -- 12. Save outputs --------------------------------------------------
    save_model(model, os.path.join(SAVED_MODELS_DIR, "baseline_mlp.pth"))
    save_scaler(scaler, os.path.join(SAVED_MODELS_DIR, "scaler.pkl"))
    save_predictions(y_true, y_pred, os.path.join(PREDICTIONS_DIR, "test_predictions.csv"))
    save_metrics(metrics, os.path.join(METRICS_DIR, "baseline_metrics.json"))

    # -- Done -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE - ALL OUTPUTS SAVED")
    print("=" * 60)
    print(f"  Model       -> {os.path.join(SAVED_MODELS_DIR, 'baseline_mlp.pth')}")
    print(f"  Scaler      -> {os.path.join(SAVED_MODELS_DIR, 'scaler.pkl')}")
    print(f"  Plots       -> {PLOTS_DIR}/")
    print(f"  Predictions -> {os.path.join(PREDICTIONS_DIR, 'test_predictions.csv')}")
    print(f"  Metrics     -> {os.path.join(METRICS_DIR, 'baseline_metrics.json')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
