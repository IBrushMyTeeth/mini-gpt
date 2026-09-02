"""
Multi-layer perceptron module.

Provides the feed-forward neural network used within a transformer block.
The MLP projects the input embedding into a higher-dimensional hidden
representation, applies a GELU activation, and projects it back to the
original embedding dimension.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Feed-forward neural network used within a transformer block.
    
    The MLP consists of two linear projections with a GELU activation between
    them. The first projection expands the input embedding to hidden_dim and
    the second projects it back to in_dim, preserving the input shape.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,

    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    