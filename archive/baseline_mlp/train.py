"""
train.py
--------
Training loop for the baseline MLP. The loss computation is separated
from the training logic to allow easy replacement with a combined
Data Loss + Physics Loss in the PINN stage.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple


def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute the training loss.

    This function is intentionally separated from the training loop
    so that it can be replaced with a combined loss (data + physics)
    in the PINN stage with minimal changes.

    Args:
        predictions: Model output tensor of shape (batch_size, 1).
        targets: Ground truth tensor of shape (batch_size, 1).
        criterion: Loss function (e.g., nn.MSELoss).

    Returns:
        Scalar loss tensor.
    """
    return criterion(predictions, targets)


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    epochs: int = 300,
    lr: float = 0.001,
    print_every: int = 20,
) -> Tuple[nn.Module, List[float], List[float]]:
    """Train the model and track training/validation loss.

    Args:
        model: The neural network to train.
        X_train: Training feature tensor.
        y_train: Training target tensor.
        X_val: Validation (test) feature tensor.
        y_val: Validation (test) target tensor.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.
        print_every: Print losses every N epochs.

    Returns:
        Tuple of (trained_model, train_losses, val_losses).
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses: List[float] = []
    val_losses: List[float] = []

    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print(f"Epochs: {epochs} | LR: {lr} | Optimizer: Adam | Loss: MSE")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # --- Training step ---
        model.train()
        optimizer.zero_grad()

        predictions = model(X_train)
        loss = compute_loss(predictions, y_train, criterion)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # --- Validation step ---
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = compute_loss(val_predictions, y_val, criterion)
            val_losses.append(val_loss.item())

        # --- Logging ---
        if epoch % print_every == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:>4d}/{epochs}]  |  "
                f"Train Loss: {loss.item():.6f}  |  "
                f"Val Loss: {val_loss.item():.6f}"
            )

    print("=" * 60)
    print("TRAINING COMPLETE")
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"Final Val Loss:   {val_losses[-1]:.6f}")
    print("=" * 60 + "\n")

    return model, train_losses, val_losses
