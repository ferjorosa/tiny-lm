"""Pre-tokenize dataset and save as binary files."""

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


def tokenize_split(
    split: Dataset,
    text_field: str,
    tokenizer,
    vocab_size: int,
    bos_token_id: int,
    eos_token_id: int,
    num_proc: int | None = None,
    batch_size: int = 1_000,
) -> np.ndarray:
    """
    Tokenize a dataset split
    """

    if num_proc is None:
        num_proc = min(8, os.cpu_count() or 1)

    def tokenize_batch(batch):
        texts = [t for t in batch[text_field] if t != ""]
        ids = [
            [bos_token_id] + tokenizer.encode_ordinary(text) + [eos_token_id]
            for text in texts
        ]
        return {"ids": ids}

    tokenized = split.map(
        tokenize_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=split.column_names,
        desc="Tokenizing",
    )

    dtype = np.uint16 if vocab_size < 65536 else np.uint32

    # Flatten into a single contiguous token stream
    tokens = np.fromiter(
        (tok for seq in tokenized["ids"] for tok in seq),
        dtype=dtype,
    )

    return tokens


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
    train_tokens = tokenize_split(
        train_data,
        dataset_config.text_field,
        tokenizer,
        tokenizer.n_vocab,
        bos_token_id,
        eos_token_id,
        num_proc=tok_config.num_proc,
    )

    val_tokens = tokenize_split(
        val_data,
        dataset_config.text_field,
        tokenizer,
        tokenizer.n_vocab,
        bos_token_id,
        eos_token_id,
        num_proc=tok_config.num_proc,
    )

    # Save binary files
    train_tokens.tofile(output_path / "train.bin")
    val_tokens.tofile(output_path / "val.bin")

    # Save metadata
    metadata = {
        "vocab_size": tokenizer.n_vocab,
        "bos_token_id": int(bos_token_id),
        "eos_token_id": int(eos_token_id),
        "pad_token_id": int(pad_token_id),
        "unk_token_id": int(unk_token_id),
        "dtype": str(train_tokens.dtype),
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(val_tokens)),
        "train_examples": int(len(train_data)),
        "val_examples": int(len(val_data)),
        "used_existing_split": used_existing_split,
    }

    if not used_existing_split:
        metadata["val_split"] = tok_config.val_split
        metadata["seed"] = seed

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTokenized {len(train_tokens):,} train + {len(val_tokens):,} val tokens")
    print(f"Saved to {output_path}/")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(
            "Usage: tokenize_data.py <tokenizer_config>\n"
            "Example: tokenize_data.py configs/tokenizers/tinystories-8k.yaml"
        )

    tokenize_dataset(sys.argv[1])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("configs/tokenizers/tinystories-8k.yaml")
    main()
