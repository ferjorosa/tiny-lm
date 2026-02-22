"""Pre-tokenize dataset and save as binary files."""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset
from datasets.utils.logging import enable_progress_bar

from tiny_lm.dataset import load_dataset_from_config
from tiny_lm.tokenizer.config import TokenizerConfig


def _init_length_stats() -> dict[str, int]:
    return {
        "char_count": 0,
        "token_count": 0,
        "min_char_len": sys.maxsize,
        "max_char_len": 0,
        "min_token_len": sys.maxsize,
        "max_token_len": 0,
    }


def _update_length_stats(stats: dict[str, int], char_len: int, token_len: int) -> None:
    stats["char_count"] += char_len
    stats["token_count"] += token_len
    stats["min_char_len"] = min(stats["min_char_len"], char_len)
    stats["max_char_len"] = max(stats["max_char_len"], char_len)
    stats["min_token_len"] = min(stats["min_token_len"], token_len)
    stats["max_token_len"] = max(stats["max_token_len"], token_len)


def _finalize_length_stats(
    stats: dict[str, int], example_count: int
) -> dict[str, dict[str, float]]:
    char_count = int(stats["char_count"])
    token_count = int(stats["token_count"])
    return {
        "char": {
            "min": float(stats["min_char_len"]),
            "max": float(stats["max_char_len"]),
            "mean": float(char_count / example_count),
        },
        "token": {
            "min": float(stats["min_token_len"]),
            "max": float(stats["max_token_len"]),
            "mean": float(token_count / example_count),
        },
    }


def tokenize_split(
    split: Dataset,
    text_field: str,
    tokenizer,
    vocab_size: int,
    bos_token_id: int,
    eos_token_id: int,
    output_path: Path,
    output_filename: str,
    num_threads: int | None = None,
    batch_size: int = 1_000,
) -> tuple[int, int, np.dtype, dict[str, dict[str, float]]]:
    """
    Tokenize a dataset split
    """

    if num_threads is None:
        num_threads = min(8, os.cpu_count() or 1)

    def tokenize_batch(batch):
        texts = [t for t in batch[text_field] if t != ""]
        ids = [
            [bos_token_id] + tokenizer.encode_ordinary(text) + [eos_token_id]
            for text in texts
        ]
        char_lengths = [len(text) for text in texts]
        token_lengths = [len(seq) for seq in ids]
        return {
            "ids": ids,
            "char_lengths": char_lengths,
            "token_lengths": token_lengths,
        }

    tokenized = split.map(
        tokenize_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_threads,
        remove_columns=split.column_names,
        desc="Tokenizing",
    )

    dtype = np.dtype(np.uint16 if vocab_size < 65536 else np.uint32)

    # Stream-write tokens to disk to avoid large memory spikes
    output_file = output_path / output_filename
    token_count = 0
    example_count = 0
    length_stats = _init_length_stats()
    with open(output_file, "wb") as f:
        for ids, char_len, token_len in zip(
            tokenized["ids"],
            tokenized["char_lengths"],
            tokenized["token_lengths"],
            strict=False,
        ):
            arr = np.asarray(ids, dtype=dtype)
            arr.tofile(f)
            token_count += arr.size
            example_count += 1
            _update_length_stats(length_stats, char_len, token_len)

    stats = _finalize_length_stats(length_stats, example_count)
    return token_count, example_count, dtype, stats


def tokenize_dataset(tokenizer_config: str | Path, seed: int = 42) -> None:
    """
    Tokenize dataset and save as binary files.

    Args:
        tokenizer_config: Path to tokenizer YAML config
        seed: Random seed for splitting (only used if no val split exists)
    """

    enable_progress_bar()

    # Load tokenizer config
    tok_config = TokenizerConfig.from_yaml(tokenizer_config)
    num_threads = tok_config.num_threads

    output_path = Path(tok_config.tokenized_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load rustbpe+tiktoken tokenizer
    tokenizer_path = Path(tok_config.output_dir) / "tokenizer.pkl"
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    bos_token_id = tokenizer.encode_single_token(tok_config.special_tokens["bos"])
    eos_token_id = tokenizer.encode_single_token(tok_config.special_tokens["eos"])
    pad_token_id = tokenizer.encode_single_token(tok_config.special_tokens["pad"])
    unk_token_id = tokenizer.encode_single_token(tok_config.special_tokens["unk"])

    # Load dataset and dataset config
    dataset, dataset_config = load_dataset_from_config(tok_config.dataset_config)

    train_split_name = dataset_config.splits["train"]
    val_split_name = dataset_config.splits.get("validation")

    # Use existing validation split or create one
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

    # Tokenize
    train_tokens_count, train_examples_count, train_dtype, train_stats = tokenize_split(
        train_data,
        dataset_config.text_field,
        tokenizer,
        tokenizer.n_vocab,
        bos_token_id,
        eos_token_id,
        output_path,
        "train.bin",
        num_threads=num_threads,
    )

    val_tokens_count, val_examples_count, val_dtype, val_stats = tokenize_split(
        val_data,
        dataset_config.text_field,
        tokenizer,
        tokenizer.n_vocab,
        bos_token_id,
        eos_token_id,
        output_path,
        "val.bin",
        num_threads=num_threads,
    )

    if train_dtype != val_dtype:
        raise ValueError("Train/val dtypes must match")

    # Save metadata
    metadata = {
        "vocab_size": tokenizer.n_vocab,
        "bos_token_id": int(bos_token_id),
        "eos_token_id": int(eos_token_id),
        "pad_token_id": int(pad_token_id),
        "unk_token_id": int(unk_token_id),
        "dtype": str(train_dtype),
        "train_tokens": int(train_tokens_count),
        "val_tokens": int(val_tokens_count),
        "train_examples": int(train_examples_count),
        "val_examples": int(val_examples_count),
        "train_length_stats": train_stats,
        "val_length_stats": val_stats,
        "used_existing_split": used_existing_split,
    }

    if not used_existing_split:
        metadata["val_split"] = tok_config.val_split
        metadata["seed"] = seed

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTokenized {train_tokens_count:,} train + {val_tokens_count:,} val tokens")
    print(f"Saved to {output_path}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize dataset and save as binary files."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to tokenizer config YAML.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (only used if no val split exists).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenize_dataset(args.config, seed=args.seed)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.extend(["--config", "configs/tokenizers/swallow-code-8k.yaml"])
    main()
