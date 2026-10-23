"""
physics_loss.py
---------------
Aggregates physics constraints into a combined physics loss.

Experiment modes:
    "none"   — No physics (pure data-driven).
    "energy" — Energy-magnitude constraint only (Gutenberg–Richter proxy).
    "full"   — Energy + frequency-magnitude constraints.

All losses are returned both weighted (for optimisation) and unweighted
(for logging), so per-epoch magnitudes can be inspected independently.
"""

import torch
from typing import Dict

from physics.energy_constraint import energy_magnitude_loss
from physics.frequency_constraint import frequency_magnitude_loss


def compute_physics_loss(
    pred_magnitude:   torch.Tensor,
    raw_energy:       torch.Tensor,
    raw_frequency:    torch.Tensor,
    lambda_energy:    float = 0.1,
    lambda_frequency: float = 0.0,
    energy_coeff_a:   float = 1.5,
    energy_coeff_b:   float = 4.8,
    epsilon:          float = 1e-8,
    experiment:       str   = "energy",
) -> Dict[str, torch.Tensor]:
    """Compute the combined physics constraint loss.

    The loss is computed ON THE CURRENT BATCH so that gradients are
    correctly aligned with the data-loss gradients in the same backward
    pass.  This replaces the original design that used a separate
    full-dataset backward pass (which wiped data gradients).

    Args:
        pred_magnitude:   Predicted magnitudes — shape (batch, 1) or (batch,).
        raw_energy:       Raw seismic-energy proxy values — shape (batch,).
        raw_frequency:    Raw spectral centroid values   — shape (batch,).
        lambda_energy:    Weight applied to energy constraint.
        lambda_frequency: Weight applied to frequency constraint.
        energy_coeff_a:   Coefficient a in log₁₀(E) = a·M + b.
        energy_coeff_b:   Coefficient b (kept for API compatibility).
        epsilon:          Numerical stability constant.
        experiment:       "none" | "energy" | "full".

    Returns:
        Dict with keys:
            energy_loss_raw    — Unweighted energy constraint loss (scalar).
            frequency_loss_raw — Unweighted frequency constraint loss (scalar).
            energy_loss        — Weighted energy loss.
            frequency_loss     — Weighted frequency loss.
            total_physics_loss — Sum of all weighted losses.
    """
    dev = pred_magnitude.device

    loss_e_raw = torch.tensor(0.0, device=dev)
    loss_f_raw = torch.tensor(0.0, device=dev)

    if experiment in ("energy", "full"):
        loss_e_raw = energy_magnitude_loss(
            pred_magnitude, raw_energy,
            coeff_a=energy_coeff_a,
            coeff_b=energy_coeff_b,
            epsilon=epsilon,
        )

    if experiment == "full":
        loss_f_raw = frequency_magnitude_loss(
            pred_magnitude, raw_frequency,
            epsilon=epsilon,
        )

    weighted_e = lambda_energy    * loss_e_raw
    weighted_f = lambda_frequency * loss_f_raw
    total      = weighted_e + weighted_f

    return {
        "energy_loss_raw":    loss_e_raw,
        "frequency_loss_raw": loss_f_raw,
        "energy_loss":        weighted_e,
        "frequency_loss":     weighted_f,
        "total_physics_loss": total,
    }
