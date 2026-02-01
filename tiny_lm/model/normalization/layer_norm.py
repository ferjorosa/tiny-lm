"""Layer Normalization."""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer Normalization.

    Normalizes inputs across the feature dimension (last dimension).
    Includes learnable scale (gamma) and shift (beta) parameters.

    Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta

    Args:
        normalized_shape: Size of the feature dimension to normalize
        eps: Small constant for numerical stability (default: 1e-5)
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor of shape (..., normalized_shape)

        Returns:
            Normalized tensor of same shape as input
        """
        # Compute mean and variance across the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable scale and shift
        return self.gamma * x_norm + self.beta
