"""
pinn_model.py
-------------
Optimized Physics-Informed Neural Network for earthquake magnitude prediction.

Architecture components:
    1. Configurable activation functions (GELU, SiLU, LeakyReLU, ReLU)
    2. Residual connections (skip connections)
    3. Layer Normalization / Batch Normalization
    4. Dropout for regularization
    5. Improved multi-head attention mechanism
    6. Optional uncertainty estimation head
    7. Fully configurable architecture via config dict
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


# Activation function registry
ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leakyrelu": lambda: nn.LeakyReLU(0.1),
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}


class ResidualBlock(nn.Module):
    """Residual block with optional normalization and dropout.

    Implements: output = LayerNorm(x + F(x))
    where F(x) = Dropout(Activation(Linear(x)))

    If input and output dimensions differ, a projection layer is used.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        use_batch_norm: bool = False,
    ):
        super().__init__()

        self.use_residual = True
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm

        # Main path
        act_fn = ACTIVATION_MAP.get(activation, nn.GELU)
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = act_fn()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Normalization
        if use_layer_norm:
            self.norm = nn.LayerNorm(out_dim)
        elif use_batch_norm:
            self.norm = nn.BatchNorm1d(out_dim)
        else:
            self.norm = nn.Identity()

        # Projection for dimension mismatch
        if in_dim != out_dim:
            self.projection = nn.Linear(in_dim, out_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = self.projection(x)

        out = self.linear(x)
        out = self.activation(out)
        out = self.dropout(out)

        # Residual connection
        out = out + identity

        # Normalization
        out = self.norm(out)

        return out


class ScaledAttention(nn.Module):
    """Improved attention mechanism with scaling and optional temperature.

    Unlike the simple sigmoid gate in the original model, this uses
    a learnable temperature parameter and softmax-like normalization
    to produce more discriminative attention weights.
    """

    def __init__(self, input_dim: int, n_heads: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.n_heads = n_heads

        # Attention score computation
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
        )

        # Learnable temperature for sigmoid sharpness
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> tuple:
        """Compute attention-weighted features.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tuple of (weighted_x, attention_weights)
        """
        # Compute attention scores
        scores = self.attention_net(x)

        # Apply temperature-scaled sigmoid
        weights = torch.sigmoid(scores / (self.temperature + 1e-8))

        # Apply attention
        weighted_x = x * weights

        return weighted_x, weights


class OptimizedPINN(nn.Module):
    """Optimized Physics-Informed Neural Network for earthquake magnitude prediction.

    Architecture improvements:
        - Residual connections for gradient flow
        - Layer normalization for training stability
        - GELU/SiLU activations for smoother gradients
        - Dropout for regularization
        - Improved attention mechanism
        - Configurable everything

    Args:
        config: Dictionary with model configuration. Keys:
            input_dim (int): Number of input features
            hidden_layers (list): List of hidden layer sizes
            activation (str): Activation function name
            dropout (float): Dropout rate
            use_layer_norm (bool): Whether to use LayerNorm
            use_batch_norm (bool): Whether to use BatchNorm
            use_residual (bool): Whether to use skip connections
            use_attention (bool): Whether to use feature attention
            use_uncertainty (bool): Whether to output uncertainty
    """

    def __init__(self, config: Dict):
        super().__init__()

        # Parse configuration
        self.input_dim = config.get("input_dim", 29)
        self.hidden_layers = config.get("hidden_layers", [256, 128, 64])
        self.activation = config.get("activation", "gelu")
        self.dropout = config.get("dropout", 0.1)
        self.use_layer_norm = config.get("use_layer_norm", True)
        self.use_batch_norm = config.get("use_batch_norm", False)
        self.use_residual = config.get("use_residual", True)
        self.use_attention = config.get("use_attention", True)
        self.use_uncertainty = config.get("use_uncertainty", False)

        # -- Attention Layer ---------------------------------------------------
        if self.use_attention:
            self.attention = ScaledAttention(self.input_dim)
        else:
            self.attention = None

        # -- Backbone Network --------------------------------------------------
        layers = []
        prev_dim = self.input_dim

        for h_dim in self.hidden_layers:
            if self.use_residual:
                layers.append(ResidualBlock(
                    in_dim=prev_dim,
                    out_dim=h_dim,
                    activation=self.activation,
                    dropout=self.dropout,
                    use_layer_norm=self.use_layer_norm,
                    use_batch_norm=self.use_batch_norm,
                ))
            else:
                act_fn = ACTIVATION_MAP.get(self.activation, nn.GELU)
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(act_fn())
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
                if self.use_layer_norm:
                    layers.append(nn.LayerNorm(h_dim))

            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # -- Output Head -------------------------------------------------------
        last_hidden = self.hidden_layers[-1]
        if self.use_uncertainty:
            self.output_head = nn.Linear(last_hidden, 2)
        else:
            self.output_head = nn.Linear(last_hidden, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            If use_uncertainty=False:
                Tensor of shape (batch_size, 1) -- predicted magnitude
            If use_uncertainty=True:
                Tensor of shape (batch_size, 2) -- [magnitude, log_uncertainty]
        """
        # Apply attention gating if enabled
        if self.attention is not None:
            x, _ = self.attention(x)

        # Backbone feature extraction
        features = self.backbone(x)

        # Output prediction
        output = self.output_head(features)

        return output

    def get_attention_weights(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract learned attention weights for interpretability."""
        if self.attention is not None:
            with torch.no_grad():
                _, weights = self.attention(x)
                return weights
        return None

    def get_config_string(self) -> str:
        """Return a human-readable configuration summary."""
        return (
            f"OptimizedPINN("
            f"input={self.input_dim}, "
            f"hidden={self.hidden_layers}, "
            f"act={self.activation}, "
            f"dropout={self.dropout}, "
            f"residual={'ON' if self.use_residual else 'OFF'}, "
            f"layernorm={'ON' if self.use_layer_norm else 'OFF'}, "
            f"attention={'ON' if self.use_attention else 'OFF'}, "
            f"uncertainty={'ON' if self.use_uncertainty else 'OFF'})"
        )

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
