"""Tests for GELU activation function."""

import torch
from tiny_lm.model.activation import GELU


class TestGELU:
    """Test GELU activation function."""

    def test_matches_pytorch_gelu(self):
        """Test that custom GELU matches PyTorch's tanh approximation."""
        custom_gelu = GELU()
        pytorch_gelu = torch.nn.GELU(approximate="tanh")

        # Test with various shapes
        test_cases = [
            torch.randn(10),  # 1D
            torch.randn(4, 768),  # 2D (batch, features)
            torch.randn(2, 10, 512),  # 3D (batch, seq_len, d_model)
        ]

        for x in test_cases:
            custom_out = custom_gelu(x)
            pytorch_out = pytorch_gelu(x)

            assert torch.allclose(custom_out, pytorch_out, atol=1e-6), (
                f"GELU output doesn't match PyTorch for shape {x.shape}"
            )

    def test_shape_preservation(self):
        """Test that GELU preserves input shape."""
        gelu = GELU()

        x = torch.randn(2, 10, 768)
        output = gelu(x)

        assert output.shape == x.shape, "GELU should preserve input shape"

    def test_gradients_flow(self):
        """Test that gradients flow through GELU."""
        gelu = GELU()

        x = torch.randn(5, 10, requires_grad=True)
        y = gelu(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow through GELU"
        assert not torch.isnan(x.grad).any(), "Gradients should be valid numbers"
        assert not torch.isinf(x.grad).any(), "Gradients should be finite"
