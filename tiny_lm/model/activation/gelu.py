"""GELU activation function."""

import torch
import torch.nn as nn


class GELU(nn.Module):
    """Gaussian Error Linear Unit activation.

    GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function
    of the standard Gaussian distribution.

    This implementation uses the tanh approximation for efficiency:
    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GELU activation.

        Args:
            x: Input tensor of any shape

        Returns:
            Tensor of same shape as input with GELU applied element-wise
        """
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )
