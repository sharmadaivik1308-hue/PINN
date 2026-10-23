"""
pinn_model.py
-------------
Configurable Physics-Informed Neural Network for earthquake magnitude
prediction. Supports four operating modes through configuration flags:

    1. Basic PINN       — Same architecture as BaselineMLP, physics loss only
    2. Attention PINN   — Feature attention layer + physics loss
    3. Uncertainty PINN — Dual-output head (μ, σ) + physics loss
    4. Full PINN        — Attention + Uncertainty + physics loss

Architecture (Full Model):
    Input (29)
        ↓
    [Attention Layer]  — Sigmoid gating (optional)
        ↓
    Linear(128) + ReLU
    Linear(64)  + ReLU
    Linear(32)  + ReLU
        ↓
    [Output Head]
        ├── Single output (1)    — magnitude only
        └── Dual output (2)      — magnitude + log_uncertainty

The attention layer learns feature importance weights via a sigmoid gate.
The uncertainty head predicts both mean magnitude and aleatoric uncertainty.

Usage:
    # Basic PINN (same as baseline architecture)
    model = EarthquakePINN(input_dim=29)

    # Attention PINN
    model = EarthquakePINN(input_dim=29, use_attention=True)

    # Full model
    model = EarthquakePINN(input_dim=29, use_attention=True,
                           use_uncertainty=True)
"""

import torch
import torch.nn as nn
from typing import List, Optional


class EarthquakePINN(nn.Module):
    """Configurable Physics-Informed Neural Network for earthquake
    magnitude prediction.

    This model extends the BaselineMLP architecture with optional
    attention-based feature weighting and uncertainty estimation.
    The physics constraints are enforced through the loss function
    (not in the model architecture itself), keeping the model modular.

    Args:
        input_dim: Number of input features (default 29).
        hidden_layers: List of hidden layer sizes (default [128, 64, 32]).
        use_attention: If True, apply learnable feature attention before
                      the backbone network.
        use_uncertainty: If True, output both magnitude prediction and
                        uncertainty estimate (2 outputs instead of 1).
    """

    def __init__(
        self,
        input_dim: int = 29,
        hidden_layers: Optional[List[int]] = None,
        use_attention: bool = False,
        use_uncertainty: bool = False,
    ):
        super(EarthquakePINN, self).__init__()

        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        self.use_attention = use_attention
        self.use_uncertainty = use_uncertainty
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers

        # -- Attention Layer (optional) ------------------------------------
        # Learns per-feature importance weights via sigmoid gating.
        # Each feature is multiplied by its learned attention weight.
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.Sigmoid(),
            )
        else:
            self.attention = None

        # -- Backbone Network ----------------------------------------------
        # Same layer sizes as BaselineMLP for fair comparison.
        backbone_layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            backbone_layers.append(nn.Linear(prev_dim, h_dim))
            backbone_layers.append(nn.ReLU())
            prev_dim = h_dim

        self.backbone = nn.Sequential(*backbone_layers)

        # -- Output Head ---------------------------------------------------
        last_hidden = hidden_layers[-1]
        if self.use_uncertainty:
            # Output: [predicted_magnitude, log_uncertainty]
            self.output_head = nn.Linear(last_hidden, 2)
        else:
            # Output: [predicted_magnitude]
            self.output_head = nn.Linear(last_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            If use_uncertainty=False:
                Tensor of shape (batch_size, 1) — predicted magnitude.
            If use_uncertainty=True:
                Tensor of shape (batch_size, 2) — [magnitude, log_uncertainty].
        """
        # Apply attention gating if enabled
        if self.attention is not None:
            attention_weights = self.attention(x)
            x = x * attention_weights

        # Backbone feature extraction
        features = self.backbone(x)

        # Output prediction
        output = self.output_head(features)

        return output

    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract learned attention weights for interpretability.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Attention weights of shape (batch_size, input_dim), or None
            if attention is not enabled.
        """
        if self.attention is not None:
            with torch.no_grad():
                return self.attention(x)
        return None

    def get_config_string(self) -> str:
        """Return a human-readable configuration summary.

        Returns:
            String describing the model configuration.
        """
        config = (
            f"EarthquakePINN("
            f"input={self.input_dim}, "
            f"hidden={self.hidden_layers}, "
            f"attention={'ON' if self.use_attention else 'OFF'}, "
            f"uncertainty={'ON' if self.use_uncertainty else 'OFF'})"
        )
        return config

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters.

        Returns:
            Integer count of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
