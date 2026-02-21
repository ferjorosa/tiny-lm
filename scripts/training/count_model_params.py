"""Print total parameter count for a model config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from tiny_lm.model.gpt2 import GPT2
from tiny_lm.model.llama3 import Llama3
from tiny_lm.model.config import GPT2Config, Llama3Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print model parameter count.")
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model config YAML.",
    )
    return parser.parse_args()


def build_gpt2_model(config: GPT2Config) -> GPT2:
    return GPT2(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        context_length=config.context_length,
        emb_dropout=config.emb_dropout,
        attn_dropout=config.attn_dropout,
        resid_dropout=config.resid_dropout,
        ffn_dropout=config.dropout,
    )


def build_llama3_model(config: Llama3Config) -> Llama3:
    return Llama3(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        context_length=config.context_length,
        n_kv_heads=config.n_kv_heads,
        ffn_hidden_dim=config.ffn_hidden_dim,
        multiple_of=config.multiple_of,
        rope_theta=config.rope_theta,
        norm_eps=config.norm_eps,
        emb_dropout=config.emb_dropout,
        attn_dropout=config.attn_dropout,
        resid_dropout=config.resid_dropout,
        ffn_dropout=config.ffn_dropout,
        qkv_bias=config.qkv_bias,
        ffn_bias=config.ffn_bias,
    )


def detect_model_type(config_path: str | Path) -> str:
    with open(config_path) as f:
        config_dict = yaml.safe_load(f) or {}

    model_type = config_dict.get("model_type")
    if model_type in {"gpt2", "llama3"}:
        return model_type
    raise ValueError(
        "model_type is required in model config and must be one of "
        f"{{'gpt2', 'llama3'}}: {config_path}"
    )


def main() -> None:
    args = parse_args()
    model_type = detect_model_type(args.model_config)

    if model_type == "llama3":
        config = Llama3Config.from_yaml(args.model_config)
        model = build_llama3_model(config)
    else:
        config = GPT2Config.from_yaml(args.model_config)
        model = build_gpt2_model(config)

    total_params = model.get_num_params(non_embedding=False)
    non_embedding_params = model.get_num_params(non_embedding=True)

    print(f"model_type={model_type}")
    print(f"total_params={total_params:,}")
    print(f"non_embedding_params={non_embedding_params:,}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.extend(["--model-config", "configs/models/ibis.yaml"])
    main()
