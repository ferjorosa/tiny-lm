"""Standard Feed-Forward Network.

A simple two-layer MLP with configurable activation, commonly used in transformer models.
The hidden layer typically has 4x the dimension of the input/output.

Paper: "Attention is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """Standard two-layer feed-forward network.

    Architecture: Linear → Activation → Linear
    The intermediate dimension is typically 4x the model dimension.

    Args:
        d_model: Input and output dimension
        d_ff: Hidden layer dimension (typically 4 * d_model)
        activation: Activation function module (e.g., GELU(), ReLU())
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: nn.Module,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Expand: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Contract: (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.fc2(x)

        return x
