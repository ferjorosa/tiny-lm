"""High-level orchestration for sharded tokenizer dataset preprocessing."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

from datasets.utils.logging import enable_progress_bar

from tiny_lm.dataset import load_dataset_from_config
from tiny_lm.tokenizer.config import TokenizerConfig
from tiny_lm.tokenizer.sharded.split_tokenize import tokenize_split_to_shards


def tokenize_dataset_sharded(
    tokenizer_config: str | Path,
    seed: int = 42,
    shard_size_mb: int = 512,
    flush_buffer_mb: int = 64,
    tokenize_workers: int = 4,
    max_in_flight: int | None = None,
    batch_size: int = 1_000,
) -> None:
    """Tokenize dataset and save train/val as shard directories + manifest."""
    start_time = time.perf_counter()
    enable_progress_bar()

    tok_config = TokenizerConfig.from_yaml(tokenizer_config)
    tokenize_workers = max(1, tokenize_workers)

    output_path = Path(tok_config.tokenized_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer_path = Path(tok_config.output_dir) / "tokenizer.pkl"
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    bos_token_id = tokenizer.encode_single_token(tok_config.special_tokens["bos"])
    eos_token_id = tokenizer.encode_single_token(tok_config.special_tokens["eos"])
    pad_token_id = tokenizer.encode_single_token(tok_config.special_tokens["pad"])
    unk_token_id = tokenizer.encode_single_token(tok_config.special_tokens["unk"])

    dataset, dataset_config = load_dataset_from_config(tok_config.dataset_config)
    train_split_name = dataset_config.splits["train"]
    val_split_name = dataset_config.splits.get("validation")

    if val_split_name and val_split_name in dataset:
        train_data = dataset[train_split_name]
        val_data = dataset[val_split_name]
        used_existing_split = True
    else:
        splits = dataset[train_split_name].train_test_split(
            test_size=tok_config.val_split,
            seed=seed,
            shuffle=True,
        )
        train_data = splits["train"]
        val_data = splits["test"]
        used_existing_split = False

    train_result = tokenize_split_to_shards(
        split=train_data,
        text_field=dataset_config.text_field,
        tokenizer=tokenizer,
        vocab_size=tokenizer.n_vocab,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        split_dir=output_path / "train",
        tokenize_workers=tokenize_workers,
        max_in_flight=max_in_flight,
        shard_size_mb=shard_size_mb,
        flush_buffer_mb=flush_buffer_mb,
        batch_size=batch_size,
    )
    val_result = tokenize_split_to_shards(
        split=val_data,
        text_field=dataset_config.text_field,
        tokenizer=tokenizer,
        vocab_size=tokenizer.n_vocab,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        split_dir=output_path / "val",
        tokenize_workers=tokenize_workers,
        max_in_flight=max_in_flight,
        shard_size_mb=shard_size_mb,
        flush_buffer_mb=flush_buffer_mb,
        batch_size=batch_size,
    )

    if train_result.dtype != val_result.dtype:
        raise ValueError("Train/val dtypes must match")

    metadata: dict[str, Any] = {
        "format": "sharded_bin_v1",
        "vocab_size": tokenizer.n_vocab,
        "bos_token_id": int(bos_token_id),
        "eos_token_id": int(eos_token_id),
        "pad_token_id": int(pad_token_id),
        "unk_token_id": int(unk_token_id),
        "dtype": str(train_result.dtype),
        "train_tokens": int(train_result.token_count),
        "val_tokens": int(val_result.token_count),
        "train_examples": int(train_result.example_count),
        "val_examples": int(val_result.example_count),
        "train_length_stats": train_result.length_stats,
        "val_length_stats": val_result.length_stats,
        "used_existing_split": used_existing_split,
        "tokenize_seconds": float(time.perf_counter() - start_time),
        "sharding": {
            "shard_size_mb": shard_size_mb,
            "flush_buffer_mb": flush_buffer_mb,
            "tokenize_workers": tokenize_workers,
            "max_in_flight": max_in_flight,
            "batch_size": batch_size,
            "train_shards": train_result.shards,
            "val_shards": val_result.shards,
        },
    }
    if not used_existing_split:
        metadata["val_split"] = tok_config.val_split
        metadata["seed"] = seed

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    manifest = {
        "format": "sharded_bin_v1",
        "dtype": str(train_result.dtype),
        "splits": {
            "train": train_result.shards,
            "val": val_result.shards,
        },
    }
    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"\nTokenized {train_result.token_count:,} train + {val_result.token_count:,} val tokens"
    )
    print(
        f"Saved shards to {output_path}/train and {output_path}/val "
        f"(train shards={len(train_result.shards)}, val shards={len(val_result.shards)})"
    )
