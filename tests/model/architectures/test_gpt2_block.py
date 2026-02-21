"""Tests for GPT-2 transformer block."""

import torch
from tiny_lm.model.components.activation import GELU
from tiny_lm.model.gpt2 import GPT2Block


class TestGPT2Block:
    """Test GPT-2 transformer block."""

    def test_shape_preservation(self):
        """Test that block preserves input shape."""
        d_model = 768
        n_heads = 12
        d_ff = 3072
        context_length = 1024

        block = GPT2Block(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            context_length=context_length,
            activation=GELU(),
            attn_dropout=0.0,
            resid_dropout=0.0,
            ffn_dropout=0.0,
        )

        # Test with various sequence lengths
        for seq_len in [1, 10, 50, 256]:
            x = torch.randn(2, seq_len, d_model)
            output = block(x)

            assert output.shape == x.shape, (
                f"Expected shape {x.shape}, got {output.shape}"
            )

    def test_residual_connections(self):
        """Test that residual connections work correctly."""
        d_model = 256
        n_heads = 8
        d_ff = 1024
        context_length = 512

        block = GPT2Block(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            context_length=context_length,
            activation=GELU(),
            attn_dropout=0.0,
            resid_dropout=0.0,
            ffn_dropout=0.0,
        )

        x = torch.randn(2, 10, d_model)
        output = block(x)

        # Output should be different from input (not identity)
        assert not torch.allclose(output, x, atol=1e-3), (
            "Block should transform input, not be identity"
        )

        # But should have same shape
        assert output.shape == x.shape

    def test_gradients_flow(self):
        """Test that gradients flow through all components."""
        d_model = 128
        n_heads = 4
        d_ff = 512
        context_length = 256

        block = GPT2Block(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            context_length=context_length,
            activation=GELU(),
            attn_dropout=0.0,
            resid_dropout=0.0,
            ffn_dropout=0.0,
        )

        x = torch.randn(2, 10, d_model, requires_grad=True)
        output = block(x)
        loss = output.sum()
        loss.backward()

        # Check input gradients
        assert x.grad is not None, "Gradients should flow through input"
        assert not torch.isnan(x.grad).any(), "Input gradients should be valid"

        # Check that all submodules have gradients
        for name, param in block.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"
            assert not torch.isnan(param.grad).any(), (
                f"Gradients for {name} should be valid"
            )

    def test_dropout_behavior(self):
        """Test that dropout works in training but not in eval mode."""
        d_model = 256
        n_heads = 8
        d_ff = 1024
        context_length = 512

        block = GPT2Block(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            context_length=context_length,
            activation=GELU(),
            attn_dropout=0.5,
            resid_dropout=0.5,
            ffn_dropout=0.5,
        )

        x = torch.randn(2, 10, d_model)

        # Training mode: outputs should differ
        block.train()
        out1 = block(x)
        out2 = block(x)
        assert not torch.allclose(out1, out2, atol=1e-6), (
            "Dropout should cause different outputs in training mode"
        )

        # Eval mode: outputs should be identical
        block.eval()
        out1 = block(x)
        out2 = block(x)
        assert torch.allclose(out1, out2), (
            "Outputs should be deterministic in eval mode"
        )

    def test_different_config(self):
        """Test block with different configurations."""
        configs = [
            {"d_model": 128, "n_heads": 4, "d_ff": 512},
            {"d_model": 256, "n_heads": 8, "d_ff": 1024},
            {"d_model": 512, "n_heads": 16, "d_ff": 2048},
        ]

        for cfg in configs:
            block = GPT2Block(
                d_model=cfg["d_model"],
                n_heads=cfg["n_heads"],
                d_ff=cfg["d_ff"],
                context_length=512,
                activation=GELU(),
            )

            x = torch.randn(2, 10, cfg["d_model"])
            output = block(x)

            assert output.shape == x.shape, (
                f"Block with config {cfg} should preserve shape"
            )

    def test_with_different_activations(self):
        """Test block with different activation functions."""
        d_model = 128
        n_heads = 4
        d_ff = 512
        context_length = 256

        activations = [GELU(), torch.nn.ReLU(), torch.nn.SiLU()]

        x = torch.randn(2, 10, d_model)

        for activation in activations:
            block = GPT2Block(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                context_length=context_length,
                activation=activation,
                attn_dropout=0.0,
                resid_dropout=0.0,
                ffn_dropout=0.0,
            )

            output = block(x)
            assert output.shape == x.shape, (
                f"Block with {activation.__class__.__name__} should preserve shape"
            )

    def test_batch_independence(self):
        """Test that different batch elements are processed independently."""
        d_model = 256
        n_heads = 8
        d_ff = 1024
        context_length = 512

        block = GPT2Block(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            context_length=context_length,
            activation=GELU(),
            attn_dropout=0.0,
            resid_dropout=0.0,
            ffn_dropout=0.0,
        )

        # Create batch with different sequences
        x1 = torch.randn(1, 10, d_model)
        x2 = torch.randn(1, 10, d_model)
        x_batch = torch.cat([x1, x2], dim=0)

        # Process individually and in batch
        block.eval()
        out1 = block(x1)
        out2 = block(x2)
        out_batch = block(x_batch)

        # Results should match
        assert torch.allclose(out1, out_batch[0:1], atol=1e-6), (
            "First batch element should match individual processing"
        )
        assert torch.allclose(out2, out_batch[1:2], atol=1e-6), (
            "Second batch element should match individual processing"
        )
