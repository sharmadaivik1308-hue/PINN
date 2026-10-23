"""
model.py
--------
Defines the baseline feed-forward neural network (MLP) for earthquake
magnitude prediction.

Architecture:
    Input (29) -> Linear(128) + ReLU -> Linear(64) + ReLU -> Linear(32) + ReLU -> Linear(1)

This is a simple baseline with no attention, dropout, batch normalisation,
uncertainty heads, or physics constraints. Those will be added in Stage 2
(PINN implementation).
"""

import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """Simple feed-forward MLP for earthquake magnitude prediction.

    A four-layer fully connected network designed as the baseline model.
    The architecture matches the base network described in the research
    methodology document, minus the attention and uncertainty components
    that will be added in the PINN stage.

    Args:
        input_dim: Number of input features (default 29).
    """

    def __init__(self, input_dim: int = 29):
        super(BaselineMLP, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Predicted magnitude tensor of shape (batch_size, 1).
        """
        return self.network(x)
