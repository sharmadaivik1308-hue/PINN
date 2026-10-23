"""
utils.py
--------
Utility functions for directory setup, model saving/loading,
and persisting predictions, metrics, and feature metadata.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from typing import Dict, List


def setup_directories(config: dict) -> None:
    """Create all required output directories from config.

    Args:
        config: Configuration dictionary with 'output' section.
    """
    output = config.get("output", {})
    dirs = [
        output.get("model_dir", "outputs/model"),
        output.get("metrics_dir", "outputs/metrics"),
        output.get("plots_dir", "outputs/plots"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[Utils] Output directories verified.")


def save_model(model: nn.Module, path: str) -> None:
    """Save model state dict to a .pth file.

    Args:
        model: Trained PyTorch model.
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Utils] Model saved -> {path}")


def save_scaler(scaler: StandardScaler, path: str) -> None:
    """Save fitted StandardScaler to a pickle file.

    Args:
        scaler: Fitted StandardScaler instance.
        path: Destination file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[Utils] Scaler saved -> {path}")


def save_feature_names(feature_names: List[str], path: str) -> None:
    """Save the list of feature names used by the model.

    Args:
        feature_names: List of feature column names.
        path: Destination JSON file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"[Utils] Feature names saved -> {path}")


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str,
) -> None:
    """Save predictions alongside ground truth as CSV.

    Args:
        y_true: Ground truth magnitude values.
        y_pred: Predicted magnitude values.
        path: Destination CSV file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({
        "actual_magnitude": y_true.flatten(),
        "predicted_magnitude": y_pred.flatten(),
        "error": (y_true - y_pred).flatten(),
    })
    df.to_csv(path, index=False)
    print(f"[Utils] Predictions saved -> {path}")


def save_metrics(metrics: Dict[str, float], path: str) -> None:
    """Save evaluation metrics as a JSON file.

    Args:
        metrics: Dictionary of metric name -> value.
        path: Destination JSON file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[Utils] Metrics saved -> {path}")


def save_loss_history(history: dict, path: str) -> None:
    """Save loss history to a JSON file.

    Args:
        history: Dictionary of loss component lists.
        path: Destination JSON file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"[Utils] Loss history saved -> {path}")