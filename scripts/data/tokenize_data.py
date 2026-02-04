"""Pre-tokenize dataset and save as binary files."""

import json
import sys
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from tiny_lm.data.loading import load_dataset_from_config
from tiny_lm.tokenizer.config import TokenizerConfig


def tokenize_split(
    split: Dataset,
    text_field: str,
    tokenizer: PreTrainedTokenizerFast,
    vocab_size: int,
) -> np.ndarray:
    """Tokenize a dataset split into token array."""
    tokens = []
    for ex in split:
        text = ex[text_field].strip()
        if text:
            tokens.extend(tokenizer.encode(text, add_special_tokens=True))
    dtype = np.uint16 if vocab_size < 65536 else np.uint32
    return np.array(tokens, dtype=dtype)


def tokenize_dataset(tokenizer_config: str | Path, seed: int = 42) -> None:
    """
    Tokenize dataset and save as binary files.

    Args:
        tokenizer_config: Path to tokenizer YAML config
        seed: Random seed for splitting (only used if no val split exists)
    """
    # Load tokenizer config
    tok_config = TokenizerConfig.from_yaml(tokenizer_config)

    output_path = Path(tok_config.tokenized_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(tok_config.output_dir) / "tokenizer.json")
    )

    # Load dataset and config
    dataset, dataset_config = load_dataset_from_config(tok_config.dataset_config)

    # Get split names from config
    train_split = dataset_config.splits["train"]
    val_split_name = dataset_config.splits.get("validation")

    # Use existing validation split or create one
    if val_split_name and val_split_name in dataset:
        train_data = dataset[train_split]
        val_data = dataset[val_split_name]
    else:
        splits = dataset[train_split].train_test_split(
            test_size=tok_config.val_split, seed=seed, shuffle=True
        )
        train_data = splits["train"]
        val_data = splits["test"]

    # Process both splits
    train_tokens = tokenize_split(
        train_data, dataset_config.text_field, tokenizer, tokenizer.vocab_size
    )
    val_tokens = tokenize_split(
        val_data, dataset_config.text_field, tokenizer, tokenizer.vocab_size
    )

    # Save binary files
    train_tokens.tofile(output_path / "train.bin")
    val_tokens.tofile(output_path / "val.bin")

    # Save essential metadata
    metadata = {
        "vocab_size": tokenizer.vocab_size,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "dtype": str(train_tokens.dtype),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "used_existing_split": val_split_name and val_split_name in dataset,
    }

    # Add split info if we created one
    if not (val_split_name and val_split_name in dataset):
        metadata["val_split"] = tok_config.val_split
        metadata["seed"] = seed

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Tokenized {len(train_tokens):,} train + {len(val_tokens):,} val tokens")
    print(f"Saved to {output_path}/")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(
            "Usage: tokenize_data.py <tokenizer_config>\n"
            "Example: tokenize_data.py configs/tokenizers/tinystories-8k.yaml"
        )

    tokenizer_config = sys.argv[1]
    tokenize_dataset(tokenizer_config=tokenizer_config)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("configs/tokenizers/tinystories-8k.yaml")
    main()
