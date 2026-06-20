"""Tests for Llama 3-style transformer block."""

import torch

from tiny_lm.model.llama3 import Llama3Block
from tiny_lm.model.components.position import RoPE


class TestLlama3Block:
    """Test Llama 3 block behavior."""

    def test_shape_preservation_with_rope(self):
        """Test block preserves shape when RoPE frequencies are provided."""
        d_model = 512
        n_heads = 8
        context_length = 1024
        seq_len = 32

        block = Llama3Block(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=4,
            context_length=context_length,
            ffn_hidden_dim=1536,
            multiple_of=256,
            attn_dropout=0.0,
            resid_dropout=0.0,
            ffn_dropout=0.0,
        )
        rope = RoPE(dim=d_model // n_heads, max_seq_len=context_length)

        x = torch.randn(2, seq_len, d_model)
        out = block(x, freqs_cis=rope(seq_len))

        assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"

    def test_shape_preservation_without_rope(self):
        """Test block also works without RoPE for compatibility."""
        block = Llama3Block(
            d_model=256,
            n_heads=8,
            n_kv_heads=2,
            context_length=256,
            attn_dropout=0.0,
            resid_dropout=0.0,
            ffn_dropout=0.0,
        )

        x = torch.randn(2, 16, 256)
        out = block(x)

        assert out.shape == x.shape

    def test_gradients_flow(self):
        """Test gradients flow through attention, norms, and SwiGLU."""
        d_model = 256
        n_heads = 8
        context_length = 512
        seq_len = 20

        block = Llama3Block(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=4,
            context_length=context_length,
            attn_dropout=0.0,
            resid_dropout=0.0,
            ffn_dropout=0.0,
        )
        rope = RoPE(dim=d_model // n_heads, max_seq_len=context_length)

        x = torch.randn(2, seq_len, d_model, requires_grad=True)
        out = block(x, freqs_cis=rope(seq_len))
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow through input"
        assert not torch.isnan(x.grad).any(), "Input gradients should be valid"
        for name, param in block.named_parameters():
            assert param.grad is not None, f"Parameter {name} should have gradients"

    def test_gqa_configuration_is_wired(self):
        """Test block attention is configured for grouped-query attention."""
        block = Llama3Block(
            d_model=128,
            n_heads=8,
            n_kv_heads=2,
            context_length=128,
            attn_dropout=0.0,
            resid_dropout=0.0,
            ffn_dropout=0.0,
        )

        assert block.attn.n_heads == 8
        assert block.attn.n_kv_heads == 2
        assert block.attn.n_rep == 4

    def test_dropout_behavior(self):
        """Test residual/attention/ffn dropout behavior in train vs eval."""
        d_model = 256
        n_heads = 8
        context_length = 256
        seq_len = 16

        block = Llama3Block(
            d_model=d_model,
            n_heads=n_heads,
            context_length=context_length,
            n_kv_heads=4,
            attn_dropout=0.5,
            resid_dropout=0.5,
            ffn_dropout=0.5,
        )
        rope = RoPE(dim=d_model // n_heads, max_seq_len=context_length)
        freqs = rope(seq_len)
        x = torch.randn(2, seq_len, d_model)

        block.train()
        out1 = block(x, freqs_cis=freqs)
        out2 = block(x, freqs_cis=freqs)
        assert not torch.allclose(out1, out2, atol=1e-6), (
            "Dropout should cause different outputs in training mode"
        )

        block.eval()
        out1 = block(x, freqs_cis=freqs)
        out2 = block(x, freqs_cis=freqs)
        assert torch.allclose(out1, out2), (
            "Outputs should be deterministic in eval mode"
        )
