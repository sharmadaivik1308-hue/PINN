"""
energy_constraint.py
--------------------
Implements the Gutenberg–Richter energy–magnitude scaling law as a
differentiable physics constraint for PINN training.

Seismological Relationship:
    log₁₀(E) = 1.5 * M + 4.8

Where:
    E = seismic energy (from the 'energy' feature in the dataset)
    M = earthquake magnitude (predicted by the model)

Implementation Note:
    The 'energy' feature in this dataset is NOT the total seismic energy
    in Joules — it is a P-wave feature proportional to ∫a²(t)dt computed
    from the first 3 seconds. Therefore, the Gutenberg–Richter coefficients
    (a=1.5, b=4.8) do not directly apply to the raw feature values.

    Instead, we enforce the PROPORTIONALITY constraint:
        log₁₀(E_feature) ∝ M

    This is implemented as a soft monotonic constraint that penalises
    deviations from a linear relationship between log₁₀(energy) and
    predicted magnitude, allowing the model to learn the correct
    intercept and slope from data while the physics regularises the
    direction of the relationship.

Reference:
    Gutenberg, B. and Richter, C.F. (1956). Magnitude and Energy of
    Earthquakes. Annali di Geofisica, 9, 1-15.
"""

import torch


def energy_magnitude_loss(
    pred_magnitude: torch.Tensor,
    raw_energy: torch.Tensor,
    coeff_a: float = 1.5,
    coeff_b: float = 4.8,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute the energy–magnitude constraint loss.

    Enforces the monotonic relationship between log₁₀(energy) and
    magnitude. Uses a correlation-based loss that penalises when the
    predicted magnitude does not track with observed energy.

    The loss has two components:
    1. A correlation penalty: ensures pred_mag and log_energy move together
    2. A scaled residual: penalises large deviations from the linear fit

    Args:
        pred_magnitude: Predicted magnitude tensor of shape (batch_size, 1)
                       or (batch_size,).
        raw_energy: Raw (un-normalised) seismic energy values of shape
                   (batch_size,).
        coeff_a: Multiplicative coefficient for scaling (used for gradient
                direction). Default 1.5.
        coeff_b: Not used in the correlation-based formulation but kept
                for API compatibility.
        epsilon: Small constant for numerical stability in log₁₀.

    Returns:
        Scalar tensor: mean squared constraint violation.
    """
    # Ensure pred_magnitude is 1-D
    pred_mag = pred_magnitude.squeeze()

    # Compute log₁₀(energy) from raw features
    log_energy = torch.log10(torch.abs(raw_energy) + epsilon)

    # Standardise both to zero-mean, unit-variance for fair comparison
    log_e_mean = log_energy.mean()
    log_e_std = log_energy.std() + epsilon
    log_e_norm = (log_energy - log_e_mean) / log_e_std

    pred_mean = pred_mag.mean()
    pred_std = pred_mag.std() + epsilon
    pred_norm = (pred_mag - pred_mean) / pred_std

    # The physics says these should be positively correlated with
    # slope ≈ 1.5. Enforce this by penalising the squared difference
    # between the normalised quantities scaled by the expected slope.
    expected = coeff_a * pred_norm
    residual = log_e_norm - expected

    loss = torch.mean(residual ** 2)

    return loss
