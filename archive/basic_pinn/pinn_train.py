"""
pinn_train.py
-------------
Training loop for the Physics-Informed Neural Network. Extends the baseline
training by adding:
    1. Physics loss (energy + frequency constraints)
    2. Uncertainty-aware loss (negative log-likelihood)
    3. Component-level loss tracking for analysis

The total loss is computed as:
    L_total = L_data + λ_energy * L_energy + λ_freq * L_frequency

Where L_data is either MSE (standard) or NLL (uncertainty-aware).

All loss components are tracked separately during training for
visualisation and analysis.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple

from physics.physics_loss import compute_physics_loss


def compute_data_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    use_uncertainty: bool = False,
) -> torch.Tensor:
    """Compute the data-fitting loss.

    For standard mode: Mean Squared Error.
    For uncertainty mode: Negative log-likelihood assuming Gaussian noise.

    The uncertainty-aware NLL loss is:
        L = mean( (y - μ)² / (2σ²) + log(σ) )

    where μ = predictions[:, 0] and σ = exp(predictions[:, 1]).

    Args:
        predictions: Model output tensor.
            Shape (batch_size, 1) for standard mode.
            Shape (batch_size, 2) for uncertainty mode [μ, log_σ].
        targets: Ground truth tensor of shape (batch_size, 1).
        use_uncertainty: If True, use NLL loss with learned uncertainty.

    Returns:
        Scalar loss tensor.
    """
    if use_uncertainty:
        mu = predictions[:, 0]
        log_sigma = predictions[:, 1]

        # Clamp log_sigma to prevent training instability.
        # Without clamping, the model can exploit extreme σ values:
        #   - Very large σ makes the NLL approach −∞ (loss goes negative)
        #   - Very small σ causes gradient explosion
        # Range [-2, 2] constrains σ to [0.14, 7.39], which is reasonable
        # for earthquake magnitudes in the range 3.5–6.0.
        log_sigma = torch.clamp(log_sigma, min=-2.0, max=2.0)
        sigma = torch.exp(log_sigma)

        # Negative log-likelihood of Gaussian
        nll = torch.mean(
            ((targets.squeeze() - mu) ** 2) / (2 * sigma ** 2)
            + log_sigma
        )

        # Add a small MSE anchor to prevent the model from ignoring
        # prediction accuracy in favour of inflating uncertainty.
        mse_anchor = torch.mean((targets.squeeze() - mu) ** 2)

        return nll + 0.1 * mse_anchor
    else:
        criterion = nn.MSELoss()
        return criterion(predictions, targets)


def get_magnitude_prediction(
    predictions: torch.Tensor,
    use_uncertainty: bool = False,
) -> torch.Tensor:
    """Extract the magnitude prediction from model output.

    Args:
        predictions: Model output tensor.
        use_uncertainty: If True, extract only the magnitude column.

    Returns:
        Magnitude tensor of shape (batch_size, 1).
    """
    if use_uncertainty:
        return predictions[:, 0].unsqueeze(1)
    else:
        return predictions


def train_pinn(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    raw_energy_train: torch.Tensor,
    raw_frequency_train: torch.Tensor,
    raw_energy_val: torch.Tensor,
    raw_frequency_val: torch.Tensor,
    epochs: int = 300,
    lr: float = 0.001,
    lambda_energy: float = 0.3,
    lambda_frequency: float = 0.3,
    energy_coeff_a: float = 1.5,
    energy_coeff_b: float = 4.8,
    epsilon: float = 1e-8,
    use_uncertainty: bool = False,
    use_physics: bool = True,
    print_every: int = 20,
    model_name: str = "pinn",
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train a PINN model with combined data + physics loss.

    This function handles all four model variants:
        - Basic PINN: MSE + physics loss
        - Attention PINN: MSE + physics loss (model has attention)
        - Uncertainty PINN: NLL + physics loss
        - Full PINN: NLL + physics loss (model has attention)

    Args:
        model: EarthquakePINN model instance.
        X_train: Normalised training features tensor.
        y_train: Training target tensor.
        X_val: Normalised validation features tensor.
        y_val: Validation target tensor.
        raw_energy_train: Un-normalised energy values for training set.
        raw_frequency_train: Un-normalised spectral centroid for training set.
        raw_energy_val: Un-normalised energy values for validation set.
        raw_frequency_val: Un-normalised spectral centroid for validation set.
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimiser.
        lambda_energy: Weight for energy physics constraint.
        lambda_frequency: Weight for frequency physics constraint.
        energy_coeff_a: Coefficient a in energy–magnitude law.
        energy_coeff_b: Coefficient b in energy–magnitude law.
        epsilon: Numerical stability constant.
        use_uncertainty: Whether the model outputs uncertainty.
        use_physics: Whether to apply physics constraints.
        print_every: Print losses every N epochs.
        model_name: Name for logging purposes.

    Returns:
        Tuple of (trained_model, loss_history) where loss_history is a
        dictionary mapping loss component names to lists of per-epoch values.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialise loss history tracking
    history: Dict[str, List[float]] = {
        "train_total": [],
        "train_data": [],
        "train_energy": [],
        "train_frequency": [],
        "train_physics": [],
        "val_total": [],
        "val_data": [],
    }

    # Measure training time
    start_time = time.time()

    print("\n" + "=" * 65)
    print(f"  TRAINING: {model_name.upper()}")
    print("=" * 65)
    print(f"  Epochs: {epochs} | LR: {lr} | Optimizer: Adam")
    print(f"  Physics: {'ON' if use_physics else 'OFF'} | "
          f"Uncertainty: {'ON' if use_uncertainty else 'OFF'}")
    if use_physics:
        print(f"  λ_energy: {lambda_energy} | λ_frequency: {lambda_frequency}")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        # --- Training step ------------------------------------------------
        model.train()
        optimizer.zero_grad()

        predictions = model(X_train)

        # Data loss (MSE or NLL)
        data_loss = compute_data_loss(predictions, y_train, use_uncertainty)

        # Physics loss (if enabled)
        if use_physics:
            pred_mag = get_magnitude_prediction(predictions, use_uncertainty)
            physics_losses = compute_physics_loss(
                pred_magnitude=pred_mag,
                raw_energy=raw_energy_train,
                raw_frequency=raw_frequency_train,
                lambda_energy=lambda_energy,
                lambda_frequency=lambda_frequency,
                energy_coeff_a=energy_coeff_a,
                energy_coeff_b=energy_coeff_b,
                epsilon=epsilon,
            )
            total_loss = data_loss + physics_losses["total_physics_loss"]
        else:
            physics_losses = {
                "energy_loss": torch.tensor(0.0),
                "frequency_loss": torch.tensor(0.0),
                "total_physics_loss": torch.tensor(0.0),
            }
            total_loss = data_loss

        total_loss.backward()
        optimizer.step()

        # Record training losses
        history["train_total"].append(total_loss.item())
        history["train_data"].append(data_loss.item())
        history["train_energy"].append(physics_losses["energy_loss"].item())
        history["train_frequency"].append(physics_losses["frequency_loss"].item())
        history["train_physics"].append(physics_losses["total_physics_loss"].item())

        # --- Validation step ----------------------------------------------
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_data_loss = compute_data_loss(
                val_predictions, y_val, use_uncertainty
            )
            history["val_total"].append(val_data_loss.item())
            history["val_data"].append(val_data_loss.item())

        # --- Logging ------------------------------------------------------
        if epoch % print_every == 0 or epoch == 1:
            msg = (
                f"  Epoch [{epoch:>4d}/{epochs}]  |  "
                f"Total: {total_loss.item():.6f}  |  "
                f"Data: {data_loss.item():.6f}"
            )
            if use_physics:
                msg += f"  |  Physics: {physics_losses['total_physics_loss'].item():.6f}"
            msg += f"  |  Val: {val_data_loss.item():.6f}"
            print(msg)

    elapsed = time.time() - start_time

    print("=" * 65)
    print(f"  TRAINING COMPLETE — {model_name.upper()}")
    print(f"  Final Train Loss: {history['train_total'][-1]:.6f}")
    print(f"  Final Val Loss:   {history['val_total'][-1]:.6f}")
    print(f"  Training Time:    {elapsed:.1f} seconds")
    print("=" * 65 + "\n")

    # Store training time
    history["training_time_seconds"] = [elapsed]

    return model, history
