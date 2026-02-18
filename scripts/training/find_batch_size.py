"""Estimate a safe batch size for a given model and dataset."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from tiny_lm.data.bin import BinDataConfig, BinTokenDataModule
from tiny_lm.model.architectures.gpt2 import GPT2
from tiny_lm.model.architectures.llama3 import Llama3
from tiny_lm.model.config import GPT2Config, Llama3Config
from tiny_lm.training import TrainingConfig
from tiny_lm.utils.precision import resolve_precision_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate max batch size that fits in GPU memory."
    )
    parser.add_argument(
        "--model-config",
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--data-config",
        help="Path to data config YAML.",
    )
    parser.add_argument(
        "--training-config",
        help="Path to training config YAML (for precision settings).",
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=8,
        help="Starting batch size for probing.",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=1024,
        help="Upper bound for probing.",
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
        attn_backend=config.attn_backend,
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


def run_one_step(
    model_config: GPT2Config | Llama3Config,
    model_type: str,
    data_config: BinDataConfig,
    batch_size: int,
    precision: str,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "llama3":
        model = build_llama3_model(model_config).to(device)
    else:
        model = build_gpt2_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    data_module = BinTokenDataModule(
        train_path=data_config.train_path,
        val_path=data_config.val_path,
        block_size=data_config.block_size,
        stride=data_config.stride,
        dtype=np.dtype(data_config.dtype),
        eos_token_id=data_config.eos_token_id,
        batch_size=batch_size,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=data_config.drop_last,
    )
    data_module.setup("fit")
    batch = next(iter(data_module.train_dataloader()))
    input_ids, targets = (t.to(device) for t in batch)

    autocast_context = contextlib.nullcontext()
    if device.type == "cuda" and precision in {"bf16", "fp16"}:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        autocast_context = torch.autocast(device_type="cuda", dtype=dtype)

    optimizer.zero_grad(set_to_none=True)
    with autocast_context:
        logits = model(input_ids)
        vocab_size = logits.size(-1)
        loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
    loss.backward()
    optimizer.step()

    peak_mem = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)

    del loss, logits, input_ids, targets, optimizer, model, data_module
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return peak_mem


def main() -> None:
    args = parse_args()
    model_type = detect_model_type(args.model_config)
    if model_type == "llama3":
        model_config = Llama3Config.from_yaml(args.model_config)
    else:
        model_config = GPT2Config.from_yaml(args.model_config)

    data_config = BinDataConfig.from_yaml(args.data_config)
    training_config = TrainingConfig.from_yaml(args.training_config)
    precision = resolve_precision_name(training_config.precision)
    accumulate = training_config.accumulate_grad_batches

    low = max(1, args.start_batch)
    high = max(low, args.max_batch)
    last_good = 0
    last_good_mem = 0.0

    batch = low
    while batch <= high:
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            peak_mem = run_one_step(
                model_config,
                model_type,
                data_config,
                batch,
                precision=precision,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                break
            raise
        last_good = batch
        last_good_mem = peak_mem
        batch *= 2

    if last_good == 0:
        raise RuntimeError("No batch size fit in memory.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"max_batch_size={last_good} "
        f"peak_mem_mb={last_good_mem:.1f} device={device} precision={precision} "
        f"model_type={model_type}"
    )
    recommended = max(1, int(last_good * 0.9))
    print(f"recommended_batch_size={recommended}")
    print(f"effective_batch_size={last_good * accumulate}")
    print(f"recommended_effective_batch_size={recommended * accumulate}")
    tokens_per_step = (
        last_good * data_config.block_size if data_config.block_size > 0 else last_good
    )
    print(f"tokens_per_step={tokens_per_step}")
    print(f"tokens_per_optimizer_step={tokens_per_step * accumulate}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--model-config",
                "configs/models/ibis.yaml",
                "--training-config",
                "configs/training/swallow-code-8k-500m.yaml",
                "--data-config",
                "configs/data/swallow-code-8k-500m.yaml",
            ]
        )
    main()
