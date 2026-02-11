"""Tests for SwiGLU feed-forward network."""

import torch
import torch.nn.functional as F
from tiny_lm.model.feedforward import SwiGLU


class TestSwiGLU:
    """Test SwiGLU feed-forward network."""

    def test_shape_preservation(self):
        """Test that SwiGLU preserves batch and sequence dimensions."""
        d_model = 384
        swiglu = SwiGLU(d_model=d_model, hidden_dim=1536, dropout=0.0)

        for shape in [(2, 1, d_model), (2, 10, d_model), (4, 100, d_model)]:
            x = torch.randn(*shape)
            output = swiglu(x)
            assert output.shape == x.shape, (
                f"Expected shape {x.shape}, got {output.shape}"
            )

    def test_matches_manual_formula(self):
        """Test that SwiGLU matches explicit formula computation."""
        d_model = 256
        hidden_dim = 1024
        swiglu = SwiGLU(d_model=d_model, hidden_dim=hidden_dim, dropout=0.0)
        x = torch.randn(2, 10, d_model)

        output = swiglu(x)
        reference = swiglu.w2(F.silu(swiglu.w1(x)) * swiglu.w3(x))

        assert torch.allclose(output, reference, atol=1e-6), (
            "SwiGLU output should match direct formula"
        )

    def test_hidden_dim_rounding_with_multiple_of(self):
        """Test that hidden_dim is rounded up to a multiple_of value."""
        d_model = 384
        swiglu = SwiGLU(
            d_model=d_model,
            hidden_dim=1537,
            multiple_of=256,
            dropout=0.0,
        )

        assert swiglu.hidden_dim == 1792, (
            "Hidden dim should be rounded up to nearest multiple_of"
        )
        assert swiglu.w1.weight.shape == (1792, d_model)
        assert swiglu.w2.weight.shape == (d_model, 1792)
        assert swiglu.w3.weight.shape == (1792, d_model)

    def test_default_hidden_dim(self):
        """Test that default hidden dimension is 4 * d_model."""
        d_model = 320
        swiglu = SwiGLU(d_model=d_model)
        assert swiglu.hidden_dim == 4 * d_model

    def test_gradients_flow(self):
        """Test that gradients flow through all parameters."""
        d_model = 128
        hidden_dim = 512
        swiglu = SwiGLU(d_model=d_model, hidden_dim=hidden_dim, dropout=0.0)

        x = torch.randn(2, 10, d_model, requires_grad=True)
        output = swiglu(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow through input"
        assert not torch.isnan(x.grad).any(), "Input gradients should be valid"
        assert swiglu.w1.weight.grad is not None, "w1 should have gradients"
        assert swiglu.w2.weight.grad is not None, "w2 should have gradients"
        assert swiglu.w3.weight.grad is not None, "w3 should have gradients"

    def test_dropout_behavior(self):
        """Test dropout randomness in training and determinism in eval."""
        d_model = 256
        hidden_dim = 1024
        swiglu = SwiGLU(d_model=d_model, hidden_dim=hidden_dim, dropout=0.5)
        x = torch.randn(2, 10, d_model)

        swiglu.train()
        out1 = swiglu(x)
        out2 = swiglu(x)
        assert not torch.allclose(out1, out2, atol=1e-6), (
            "Dropout should cause different outputs in training mode"
        )

        swiglu.eval()
        out1 = swiglu(x)
        out2 = swiglu(x)
        assert torch.allclose(out1, out2), (
            "Outputs should be deterministic in eval mode"
        )
