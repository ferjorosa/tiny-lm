"""Tests for Llama 3-style model architecture."""

import pytest
import torch

from tiny_lm.model.llama3 import Llama3


class TestLlama3:
    """Test Llama 3 model architecture."""

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        vocab_size = 1000
        d_model = 256
        n_layers = 4
        n_heads = 8
        context_length = 512

        model = Llama3(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=2,
            context_length=context_length,
            emb_dropout=0.0,
            attn_dropout=0.0,
            resid_dropout=0.0,
            ffn_dropout=0.0,
        )

        for seq_len in [1, 10, 50, 256]:
            input_ids = torch.randint(0, vocab_size, (2, seq_len))
            logits = model(input_ids)
            assert logits.shape == (2, seq_len, vocab_size), (
                f"Unexpected logits shape {logits.shape}"
            )

    def test_weight_tying(self):
        """Test embedding and output projection share weights."""
        model = Llama3(
            vocab_size=1000, d_model=256, n_layers=2, n_heads=8, n_kv_heads=2
        )
        assert model.token_emb.weight is model.lm_head.weight, (
            "Token embedding and output projection should share weights"
        )

    def test_gradients_flow(self):
        """Test gradients flow through the full model."""
        vocab_size = 1000
        model = Llama3(
            vocab_size=vocab_size,
            d_model=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            context_length=128,
            emb_dropout=0.0,
            attn_dropout=0.0,
            resid_dropout=0.0,
            ffn_dropout=0.0,
        )
        input_ids = torch.randint(0, vocab_size, (2, 20))
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} should have gradients"

    def test_exceeds_context_length_raises_error(self):
        """Test context length guard."""
        model = Llama3(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            context_length=64,
        )
        input_ids = torch.randint(0, 1000, (2, 80))

        with pytest.raises(ValueError, match="exceeds maximum"):
            model(input_ids)

    def test_dropout_behavior(self):
        """Test dropout randomness in train mode and determinism in eval."""
        model = Llama3(
            vocab_size=1000,
            d_model=256,
            n_layers=2,
            n_heads=8,
            n_kv_heads=2,
            context_length=128,
            emb_dropout=0.5,
            attn_dropout=0.5,
            resid_dropout=0.5,
            ffn_dropout=0.5,
        )
        input_ids = torch.randint(0, 1000, (2, 16))

        model.train()
        out1 = model(input_ids)
        out2 = model(input_ids)
        assert not torch.allclose(out1, out2, atol=1e-6), (
            "Dropout should cause different outputs in training mode"
        )

        model.eval()
        out1 = model(input_ids)
        out2 = model(input_ids)
        assert torch.allclose(out1, out2), (
            "Outputs should be deterministic in eval mode"
        )

    def test_causal_masking(self):
        """Test that earlier tokens cannot see modified future tokens."""
        vocab_size = 1000
        model = Llama3(
            vocab_size=vocab_size,
            d_model=128,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            context_length=128,
            emb_dropout=0.0,
            attn_dropout=0.0,
            resid_dropout=0.0,
            ffn_dropout=0.0,
        )
        model.eval()

        input_ids = torch.randint(0, vocab_size, (1, 12))
        logits = model(input_ids)

        input_ids_modified = input_ids.clone()
        input_ids_modified[:, -1] = (input_ids_modified[:, -1] + 1) % vocab_size
        logits_modified = model(input_ids_modified)

        assert torch.allclose(
            logits[:, :-1, :], logits_modified[:, :-1, :], atol=1e-5
        ), "Causal masking should block future token information"
