"""Tests for LayerNorm."""

import torch
from tiny_lm.model.normalization import LayerNorm


class TestLayerNorm:
    """Test LayerNorm normalization."""

    def test_matches_pytorch_layer_norm(self):
        """Test that custom LayerNorm matches PyTorch's implementation."""
        d_model = 768
        custom_ln = LayerNorm(d_model)
        pytorch_ln = torch.nn.LayerNorm(d_model)

        # Copy parameters to ensure same initialization
        with torch.no_grad():
            pytorch_ln.weight.copy_(custom_ln.gamma)
            pytorch_ln.bias.copy_(custom_ln.beta)

        # Test with various shapes
        test_cases = [
            torch.randn(10, d_model),  # 2D (batch, features)
            torch.randn(2, 10, d_model),  # 3D (batch, seq_len, d_model)
            torch.randn(4, 8, 16, d_model),  # 4D
        ]

        for x in test_cases:
            custom_out = custom_ln(x)
            pytorch_out = pytorch_ln(x)

            assert torch.allclose(custom_out, pytorch_out, atol=1e-6), (
                f"LayerNorm output doesn't match PyTorch for shape {x.shape}"
            )

    def test_normalization_properties(self):
        """Test that LayerNorm normalizes to mean=0, std=1."""
        d_model = 768
        layer_norm = LayerNorm(d_model)

        # Initialize gamma=1, beta=0 to test pure normalization
        with torch.no_grad():
            layer_norm.gamma.fill_(1.0)
            layer_norm.beta.fill_(0.0)

        x = torch.randn(2, 10, d_model)
        output = layer_norm(x)

        # Check mean and std across feature dimension
        mean = output.mean(dim=-1)
        std = output.std(dim=-1, unbiased=False)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), (
            "Mean should be ~0"
        )
        assert torch.allclose(std, torch.ones_like(std), atol=1e-5), "Std should be ~1"

    def test_shape_preservation(self):
        """Test that LayerNorm preserves input shape."""
        d_model = 512
        layer_norm = LayerNorm(d_model)

        x = torch.randn(2, 10, d_model)
        output = layer_norm(x)

        assert output.shape == x.shape, "LayerNorm should preserve input shape"

    def test_gradients_flow(self):
        """Test that gradients flow through LayerNorm."""
        d_model = 256
        layer_norm = LayerNorm(d_model)

        x = torch.randn(2, 5, d_model, requires_grad=True)
        y = layer_norm(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow through input"
        assert layer_norm.gamma.grad is not None, "Gradients should flow through gamma"
        assert layer_norm.beta.grad is not None, "Gradients should flow through beta"
        assert not torch.isnan(x.grad).any(), "Gradients should be valid"
