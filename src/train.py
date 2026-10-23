"""
train.py
--------
Training pipeline for the Physics-Informed Neural Network.

Critical fixes vs. original implementation:
─────────────────────────────────────────────────────────────────────────────
FIX 1 — JOINT OPTIMISATION
    Original: two separate backward() calls — one for data loss, one for
    physics loss — with optimizer.zero_grad() in between.  This wiped data
    gradients before the physics step, so the two objectives were never
    co-optimised.

    Fixed: total_loss = data_loss + physics_loss → single backward().

FIX 2 — BATCH-ALIGNED PHYSICS
    Original: physics was computed on the full training set in a separate
    forward pass after the mini-batch loop, then back-propagated without
    connection to the batch indices.

    Fixed: energy and frequency tensors are included in the TensorDataset
    so every mini-batch receives the correctly aligned physics inputs.

FIX 3 — SEPARATE PHYSICS LOSS LOGGING
    Energy and frequency losses are tracked independently every epoch so
    their magnitudes can be compared to the data loss and physics weights
    tuned accordingly.

FIX 4 — DEVICE MANAGEMENT
    All tensors and the model must be on the same device before training.
    The device is detected once and passed explicitly.

FIX 5 — PER-EPOCH TIMING
    Each epoch prints its wall-clock duration so bottlenecks are visible.
─────────────────────────────────────────────────────────────────────────────
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple

from physics.physics_loss import compute_physics_loss


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_data_loss(
    predictions:     torch.Tensor,
    targets:         torch.Tensor,
    use_uncertainty: bool = False,
) -> torch.Tensor:
    """MSE data loss, or heteroscedastic NLL when uncertainty head is active."""
    if use_uncertainty:
        mu        = predictions[:, 0]
        log_sigma = predictions[:, 1].clamp(min=-2.0, max=2.0)
        sigma     = torch.exp(log_sigma)
        nll       = torch.mean(
            ((targets.squeeze() - mu) ** 2) / (2 * sigma ** 2) + log_sigma
        )
        mse_anchor = torch.mean((targets.squeeze() - mu) ** 2)
        return nll + 0.1 * mse_anchor
    return nn.functional.mse_loss(predictions, targets)


def get_magnitude_prediction(
    predictions:     torch.Tensor,
    use_uncertainty: bool = False,
) -> torch.Tensor:
    """Extract the magnitude head from model output."""
    if use_uncertainty:
        return predictions[:, 0].unsqueeze(1)
    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss stops improving.

    Stores the best model state dict so that weights can be restored
    to the best checkpoint on trigger.

    Args:
        patience:     Epochs to wait for improvement before stopping.
        min_delta:    Minimum absolute improvement that resets the counter.
        restore_best: Restore best weights when stopping.
    """

    def __init__(
        self,
        patience:     int   = 50,
        min_delta:    float = 1e-6,
        restore_best: bool  = True,
    ):
        self.patience     = patience
        self.min_delta    = min_delta
        self.restore_best = restore_best
        self.counter      = 0
        self.best_loss    = float("inf")
        self.best_state:  Optional[Dict] = None
        self.should_stop  = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            if self.restore_best:
                self.best_state = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best and self.best_state is not None:
                    model.load_state_dict(self.best_state)
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_optimized(
    model:               nn.Module,
    X_train:             torch.Tensor,
    y_train:             torch.Tensor,
    X_val:               torch.Tensor,
    y_val:               torch.Tensor,
    raw_energy_train:    torch.Tensor,
    raw_frequency_train: torch.Tensor,
    config:              Dict,
    model_name:          str = "pinn",
    device:              Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, List]]:
    """Train the PINN with jointly optimised data and physics losses.

    Args:
        model:               Initialised model (already on device).
        X_train:             Training feature tensor (on device).
        y_train:             Training target tensor (on device).
        X_val:               Validation feature tensor (on device).
        y_val:               Validation target tensor (on device).
        raw_energy_train:    Raw energy values (on device).
        raw_frequency_train: Raw spectral centroid values (on device).
        config:              Training hyperparameter dict.
        model_name:          Label for log messages.
        device:              Torch device (inferred from X_train if None).

    Returns:
        (trained_model, history_dict)
    """
    if device is None:
        device = X_train.device

    # ── Parse configuration ──────────────────────────────────────────────────
    epochs           = config.get("epochs",        500)
    lr               = config.get("learning_rate", 0.001)
    weight_decay     = config.get("weight_decay",  1e-4)
    batch_size       = config.get("batch_size",    64)
    grad_clip        = config.get("grad_clip",     1.0)
    patience         = config.get("patience",      50)
    print_every      = config.get("print_every",   20)

    use_physics       = config.get("use_physics",        True)
    use_uncertainty   = config.get("use_uncertainty",    False)
    lambda_energy     = config.get("lambda_energy",      0.1)
    lambda_frequency  = config.get("lambda_frequency",   0.0)
    energy_coeff_a    = config.get("energy_coeff_a",     1.5)
    energy_coeff_b    = config.get("energy_coeff_b",     4.8)
    epsilon           = config.get("epsilon",            1e-8)
    physics_exp       = config.get("physics_experiment", "energy")
    physics_annealing = config.get("physics_annealing",  False)
    scheduler_type    = config.get("scheduler",          "plateau")

    # If physics globally disabled, force experiment to "none"
    if not use_physics:
        physics_exp = "none"

    # ── Optimizer ────────────────────────────────────────────────────────────
    # AdamW (weight decay active) for regularised experiments;
    # plain Adam for the ablation baseline where weight_decay == 0.
    if weight_decay > 0:
        optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    # plateau_patience configurable (default 30 — original hardcoded 15 was too aggressive)
    plateau_patience = config.get("scheduler_plateau_patience", 30)
    if scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=plateau_patience, min_lr=1e-6,
        )
    elif scheduler_type == "cosine":
        # Single smooth cosine decay over all epochs (no warm restarts).
        # Warm restarts (T_0=50) caused LR spikes every 50 epochs that
        # spiked validation loss and triggered early stopping prematurely.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6,
        )
    else:
        scheduler = None

    # ── Early stopping ────────────────────────────────────────────────────────
    early_stopping = EarlyStopping(patience=patience, restore_best=True)

    # ── DataLoader ────────────────────────────────────────────────────────────
    # Include physics tensors in the dataset so every mini-batch receives the
    # correctly aligned energy and frequency values — no index bookkeeping.
    train_dataset = TensorDataset(
        X_train,
        y_train,
        raw_energy_train.view(-1, 1),
        raw_frequency_train.view(-1, 1),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    # ── History ───────────────────────────────────────────────────────────────
    history: Dict[str, List] = {
        "train_total":     [],
        "train_data":      [],
        "train_physics":   [],
        "train_energy":    [],
        "train_frequency": [],
        "val_total":       [],
        "val_data":        [],
        "learning_rate":   [],
        "epoch_time":      [],
    }

    # ── Header ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  TRAINING: {model_name.upper()}")
    print("=" * 72)
    print(f"  Device      : {device}")
    print(f"  Epochs      : {epochs}  |  LR: {lr}  |  Batch: {batch_size}")
    print(f"  Weight decay: {weight_decay}  |  Scheduler: {scheduler_type}"
          f"  |  Patience: {patience}")
    print(f"  Physics     : {physics_exp.upper()}"
          f"  |  lam_energy: {lambda_energy}"
          f"  |  lam_freq: {lambda_frequency}")
    print("=" * 72)

    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ── Mini-batch training loop ──────────────────────────────────────────
        model.train()
        acc = {"total": 0.0, "data": 0.0,
               "physics": 0.0, "energy": 0.0, "frequency": 0.0}
        n_batches = 0

        anneal = 1.0
        if physics_annealing and epoch < epochs // 4:
            anneal = epoch / max(1, epochs // 4)

        for batch_X, batch_y, batch_e, batch_f in train_loader:
            optimizer.zero_grad()

            predictions = model(batch_X)
            data_loss   = compute_data_loss(predictions, batch_y, use_uncertainty)

            # ─── Physics loss (FIX 1 + FIX 2) ────────────────────────────────
            if physics_exp != "none":
                pred_mag    = get_magnitude_prediction(predictions, use_uncertainty)
                phys_result = compute_physics_loss(
                    pred_magnitude   = pred_mag,
                    raw_energy       = batch_e.squeeze(),
                    raw_frequency    = batch_f.squeeze(),
                    lambda_energy    = lambda_energy * anneal,
                    lambda_frequency = lambda_frequency * anneal,
                    energy_coeff_a   = energy_coeff_a,
                    energy_coeff_b   = energy_coeff_b,
                    epsilon          = epsilon,
                    experiment       = physics_exp,
                )
                phys_loss  = phys_result["total_physics_loss"]
                total_loss = data_loss + phys_loss          # ← single combined tensor

                acc["physics"]   += phys_loss.item()
                acc["energy"]    += phys_result["energy_loss"].item()
                acc["frequency"] += phys_result["frequency_loss"].item()
            else:
                total_loss = data_loss

            # ─── Single backward pass ─────────────────────────────────────────
            total_loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            acc["data"]  += data_loss.item()
            acc["total"] += total_loss.item()
            n_batches    += 1

        # Epoch averages
        avg = {k: v / n_batches for k, v in acc.items()}

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_pred      = model(X_val)
            val_data_loss = compute_data_loss(val_pred, y_val, use_uncertainty)

        val_loss = val_data_loss.item()

        # ── Record history ─────────────────────────────────────────────────────
        history["train_total"].append(avg["total"])
        history["train_data"].append(avg["data"])
        history["train_physics"].append(avg["physics"])
        history["train_energy"].append(avg["energy"])
        history["train_frequency"].append(avg["frequency"])
        history["val_total"].append(val_loss)
        history["val_data"].append(val_loss)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])

        epoch_t = time.time() - epoch_start
        history["epoch_time"].append(epoch_t)

        # ── LR scheduling ──────────────────────────────────────────────────────
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # ── Early stopping ─────────────────────────────────────────────────────
        if early_stopping(val_loss, model):
            print(
                f"\n  [!] Early stopping at epoch {epoch} "
                f"(best val: {early_stopping.best_loss:.6f}, "
                f"patience={patience})"
            )
            break

        # ── Epoch log ──────────────────────────────────────────────────────────
        if epoch % print_every == 0 or epoch == 1:
            cur_lr = optimizer.param_groups[0]["lr"]
            msg = (
                f"  Epoch [{epoch:>4d}/{epochs}] | "
                f"Total: {avg['total']:.5f} | "
                f"Data: {avg['data']:.5f} | "
                f"Val: {val_loss:.5f} | "
                f"LR: {cur_lr:.2e} | "
                f"t: {epoch_t:.2f}s"
            )
            if physics_exp != "none":
                msg += (
                    f" | E_phys: {avg['energy']:.5f}"
                    f" | F_phys: {avg['frequency']:.5f}"
                )
            print(msg)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    avg_epoch_t   = (sum(history["epoch_time"])
                     / max(1, len(history["epoch_time"])))
    n_epochs      = len(history["train_total"])

    print("=" * 72)
    print(f"  COMPLETE: {model_name.upper()}")
    print(f"  Final train loss : {history['train_total'][-1]:.6f}")
    print(f"  Final val loss   : {history['val_total'][-1]:.6f}")
    print(f"  Best val loss    : {early_stopping.best_loss:.6f}")
    print(f"  Epochs run       : {n_epochs}")
    print(f"  Avg epoch time   : {avg_epoch_t:.3f}s")
    print(f"  Total time       : {total_elapsed:.1f}s")
    print("=" * 72 + "\n")

    history["training_time_seconds"] = [total_elapsed]
    history["actual_epochs"]         = [n_epochs]

    return model, history
