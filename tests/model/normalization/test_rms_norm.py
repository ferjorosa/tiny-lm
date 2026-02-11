"""Tests for RMSNorm."""

import pytest
import torch
from tiny_lm.model.normalization import RMSNorm


class TestRMSNorm:
    """Test RMSNorm normalization."""

    def test_matches_pytorch_rms_norm(self):
        """Test that custom RMSNorm matches PyTorch RMSNorm when available."""
        if not hasattr(torch.nn, "RMSNorm"):
            pytest.skip("torch.nn.RMSNorm is not available in this PyTorch version")

        d_model = 768
        custom_rms = RMSNorm(d_model, eps=1e-6)
        pytorch_rms = torch.nn.RMSNorm(d_model, eps=1e-6)

        with torch.no_grad():
            pytorch_rms.weight.copy_(custom_rms.gamma)

        x = torch.randn(2, 10, d_model)
        custom_out = custom_rms(x)
        pytorch_out = pytorch_rms(x)

        assert torch.allclose(custom_out, pytorch_out, atol=1e-6), (
            "RMSNorm output should match PyTorch RMSNorm"
        )

    def test_rms_property(self):
        """Test that output RMS is approximately 1 when gamma is 1."""
        d_model = 256
        rms_norm = RMSNorm(d_model, eps=1e-6)

        with torch.no_grad():
            rms_norm.gamma.fill_(1.0)

        x = torch.randn(2, 10, d_model)
        output = rms_norm(x)
        rms = torch.sqrt(output.pow(2).mean(dim=-1))

        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5), (
            "Output RMS should be ~1 when gamma=1"
        )

    def test_shape_preservation(self):
        """Test that RMSNorm preserves input shape."""
        d_model = 384
        rms_norm = RMSNorm(d_model)

        x = torch.randn(2, 10, d_model)
        output = rms_norm(x)

        assert output.shape == x.shape, "RMSNorm should preserve input shape"

    def test_gradients_flow(self):
        """Test that gradients flow through RMSNorm."""
        d_model = 128
        rms_norm = RMSNorm(d_model)

        x = torch.randn(2, 5, d_model, requires_grad=True)
        y = rms_norm(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow through input"
        assert rms_norm.gamma.grad is not None, "Gradients should flow through gamma"
        assert not torch.isnan(x.grad).any(), "Gradients should be valid"
