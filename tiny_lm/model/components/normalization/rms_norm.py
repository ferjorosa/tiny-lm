"""Root Mean Square Layer Normalization (RMSNorm).

RMSNorm scales activations using the root mean square across the feature
dimension. Unlike LayerNorm, RMSNorm does not subtract the mean and does not
use a learnable bias term.

Paper: "Root Mean Square Layer Normalization" (Zhang and Sennrich, 2019)
https://arxiv.org/abs/1910.07467
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMS normalization across the last dimension.

    Formula: y = gamma * x / sqrt(mean(x^2) + eps)

    Args:
        normalized_shape: Size of the feature dimension to normalize
        eps: Small constant for numerical stability
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., normalized_shape)

        Returns:
            Normalized tensor with the same shape as input
        """
        rms_inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x * rms_inv
        return self.gamma * x_norm
