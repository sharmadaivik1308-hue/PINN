"""
energy_constraint.py
--------------------
Implements differentiable seismic energy constraints for PINN training:

1. "correlation" mode (Proportionality constraint):
   Enforces that predicted magnitude positively correlates with log₁₀(energy).
   Standardises both quantities to zero-mean unit-variance and penalises
   the MSE between them (equivalent to penalising 2 * (1 - Pearson_r)).
   Coefficients a and b are NOT used in this mode.

2. "gr_absolute" mode (Gutenberg–Richter scaling law):
   Enforces:
       log₁₀(E) = a * M + b
   where:
       E = seismic energy feature (raw waveform integral proxy)
       M = predicted magnitude
       a = scaling slope (default 1.5)
       b = intercept (calibrated on training data; sign convention: log10(E) = 1.5*M + b)
   Constraint residual:
       residual = log₁₀(E) - (a * M + b)
       loss = mean(residual ** 2)

Reference:
    Gutenberg, B. and Richter, C.F. (1956). Magnitude and Energy of
    Earthquakes. Annali di Geofisica, 9, 1-15.
"""

import torch


def energy_magnitude_loss(
    pred_magnitude: torch.Tensor,
    raw_energy: torch.Tensor,
    energy_mode: str = "correlation",
    coeff_a: float = 1.5,
    coeff_b: float = -3.8633,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute the energy–magnitude constraint loss.

    Args:
        pred_magnitude: Predicted magnitude tensor (batch_size,) or (batch_size, 1).
        raw_energy: Raw seismic energy proxy values (batch_size,).
        energy_mode: "correlation" (Pearson correlation proxy) | "gr_absolute" (G-R law).
        coeff_a: Slope a in log10(E) = a*M + b (used in "gr_absolute" mode).
        coeff_b: Intercept b in log10(E) = a*M + b (used in "gr_absolute" mode).
                 Sign convention note: log10(E) = a*M + b, where b is typically
                 negative (~ -3.863) for P-wave waveform energy integrals.
        epsilon: Small constant for numerical stability in log10.

    Returns:
        Scalar tensor: constraint loss penalty.
    """
    pred_mag = pred_magnitude.squeeze()
    log_energy = torch.log10(torch.abs(raw_energy) + epsilon)

    if energy_mode == "gr_absolute":
        # Direct Gutenberg-Richter relation: log10(E) = a * M + b
        # Expected log10(E) given predicted magnitude
        expected_log_e = coeff_a * pred_mag + coeff_b
        residual = log_energy - expected_log_e
        return torch.mean(residual ** 2)

    # Default: "correlation" mode (proportionality without scale/offset assumptions)
    # Standardise both to zero-mean, unit-variance across the batch
    log_e_mean = log_energy.mean()
    log_e_std = log_energy.std() + epsilon
    log_e_norm = (log_energy - log_e_mean) / log_e_std

    pred_mean = pred_mag.mean()
    pred_std = pred_mag.std() + epsilon
    pred_norm = (pred_mag - pred_mean) / pred_std

    # MSE between normalised variables = 2 * (1 - Pearson_r)
    loss = torch.mean((log_e_norm - pred_norm) ** 2)
    return loss
