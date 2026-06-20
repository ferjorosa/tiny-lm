"""Tests for GPT-2 model architecture."""

import torch
import pytest
from tiny_lm.model.gpt2 import GPT2


class TestGPT2:
    """Test GPT-2 model architecture."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        vocab_size = 1000
        d_model = 256
        n_layers = 4
        n_heads = 8
        context_length = 512

        model = GPT2(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=4 * d_model,
            context_length=context_length,
            emb_dropout=0.0,
            attn_dropout=0.0,
            resid_dropout=0.0,
            ffn_dropout=0.0,
        )

        # Test with various sequence lengths
        for seq_len in [1, 10, 50, 256]:
            batch_size = 2
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            logits = model(input_ids)

            expected_shape = (batch_size, seq_len, vocab_size)
            assert logits.shape == expected_shape, (
                f"Expected shape {expected_shape}, got {logits.shape}"
            )

    def test_different_model_sizes(self):
        """Test GPT-2 with different sizes (small, medium, large)."""
        configs = [
            # GPT-2 Small
            {"vocab_size": 1000, "d_model": 256, "n_layers": 4, "n_heads": 4},
            # GPT-2 Medium
            {"vocab_size": 1000, "d_model": 512, "n_layers": 8, "n_heads": 8},
            # GPT-2 Large
            {"vocab_size": 1000, "d_model": 768, "n_layers": 12, "n_heads": 12},
        ]

        for cfg in configs:
            model = GPT2(
                vocab_size=cfg["vocab_size"],
                d_model=cfg["d_model"],
                n_layers=cfg["n_layers"],
                n_heads=cfg["n_heads"],
                d_ff=4 * cfg["d_model"],
                context_length=256,
            )

            input_ids = torch.randint(0, cfg["vocab_size"], (2, 10))
            logits = model(input_ids)

            assert logits.shape == (2, 10, cfg["vocab_size"]), (
                f"Model with config {cfg} should work correctly"
            )

    def test_weight_tying(self):
        """Test that token embeddings and output projection share weights."""
        vocab_size = 1000
        model = GPT2(vocab_size=vocab_size, d_model=256, n_layers=2, n_heads=4)

        # Check that weights are tied
        assert model.token_emb.weight is model.lm_head.weight, (
            "Token embedding and output projection should share weights"
        )

    def test_gradients_flow(self):
        """Test that gradients flow through the entire model."""
        vocab_size = 1000
        model = GPT2(
            vocab_size=vocab_size,
            d_model=128,
            n_layers=2,
            n_heads=4,
            emb_dropout=0.0,
            attn_dropout=0.0,
            resid_dropout=0.0,
            ffn_dropout=0.0,
        )

        input_ids = torch.randint(0, vocab_size, (2, 10))
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} should have gradients"
                assert not torch.isnan(param.grad).any(), (
                    f"Gradients for {name} should be valid"
                )

    def test_exceeds_context_length_raises_error(self):
        """Test that exceeding context length raises an error."""
        vocab_size = 1000
        context_length = 64
        model = GPT2(
            vocab_size=vocab_size,
            d_model=128,
            n_layers=2,
            n_heads=4,
            context_length=context_length,
        )

        # Try with sequence longer than context length
        input_ids = torch.randint(0, vocab_size, (2, context_length + 10))

        with pytest.raises(ValueError, match="exceeds maximum"):
            model(input_ids)

    def test_dropout_behavior(self):
        """Test that dropout works in training but not in eval mode."""
        vocab_size = 1000
        model = GPT2(
            vocab_size=vocab_size,
            d_model=256,
            n_layers=4,
            n_heads=8,
            emb_dropout=0.5,
            attn_dropout=0.5,
            resid_dropout=0.5,
            ffn_dropout=0.5,
        )

        input_ids = torch.randint(0, vocab_size, (2, 10))

        # Training mode: outputs should differ
        model.train()
        out1 = model(input_ids)
        out2 = model(input_ids)
        assert not torch.allclose(out1, out2, atol=1e-6), (
            "Dropout should cause different outputs in training mode"
        )

        # Eval mode: outputs should be identical
        model.eval()
        out1 = model(input_ids)
        out2 = model(input_ids)
        assert torch.allclose(out1, out2), (
            "Outputs should be deterministic in eval mode"
        )

    def test_get_num_params(self):
        """Test parameter counting."""
        vocab_size = 1000
        d_model = 256
        model = GPT2(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=2,
            n_heads=4,
            context_length=256,
        )

        # Total parameters
        total_params = model.get_num_params(non_embedding=False)
        assert total_params > 0, "Model should have parameters"

        # Non-embedding parameters
        non_emb_params = model.get_num_params(non_embedding=True)
        assert non_emb_params > 0, "Model should have non-embedding parameters"
        assert non_emb_params < total_params, (
            "Non-embedding params should be less than total params"
        )

    def test_batch_independence(self):
        """Test that different batch elements are processed independently."""
        vocab_size = 1000
        model = GPT2(
            vocab_size=vocab_size,
            d_model=256,
            n_layers=2,
            n_heads=4,
            emb_dropout=0.0,
            attn_dropout=0.0,
            resid_dropout=0.0,
            ffn_dropout=0.0,
        )

        model.eval()

        # Create batch with different sequences
        input1 = torch.randint(0, vocab_size, (1, 10))
        input2 = torch.randint(0, vocab_size, (1, 10))
        input_batch = torch.cat([input1, input2], dim=0)

        # Process individually and in batch
        out1 = model(input1)
        out2 = model(input2)
        out_batch = model(input_batch)

        # Results should match
        assert torch.allclose(out1, out_batch[0:1], atol=1e-6), (
            "First batch element should match individual processing"
        )
        assert torch.allclose(out2, out_batch[1:2], atol=1e-6), (
            "Second batch element should match individual processing"
        )

    def test_causal_masking(self):
        """Test that model uses causal masking (can't see future tokens)."""
        vocab_size = 1000
        model = GPT2(
            vocab_size=vocab_size,
            d_model=128,
            n_layers=2,
            n_heads=4,
            emb_dropout=0.0,
            attn_dropout=0.0,
            resid_dropout=0.0,
            ffn_dropout=0.0,
        )

        model.eval()

        # Create a sequence
        input_ids = torch.randint(0, vocab_size, (1, 10))

        # Get output
        logits = model(input_ids)

        # Modify the last token
        input_ids_modified = input_ids.clone()
        input_ids_modified[:, -1] = (input_ids_modified[:, -1] + 1) % vocab_size

        logits_modified = model(input_ids_modified)

        # All positions except the last should have the same output
        # (they can't see the future modification)
        assert torch.allclose(
            logits[:, :-1, :], logits_modified[:, :-1, :], atol=1e-5
        ), "Causal mask should prevent earlier positions from seeing future tokens"
