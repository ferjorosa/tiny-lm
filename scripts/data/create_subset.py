"""Create a subset of tokenized data by taking first N tokens from each split."""

import argparse
import json
from pathlib import Path

import numpy as np


def create_subset(
    train_input: Path,
    val_input: Path,
    train_output: Path,
    val_output: Path,
    train_tokens: int,
    val_tokens: int,
    dtype: np.dtype = np.dtype(np.uint16),
) -> None:
    """Read bin files with memmap and write subset."""
    # Read train subset
    train_data = np.memmap(train_input, dtype=dtype, mode="r")[:train_tokens]
    train_data.tofile(train_output)

    # Read val subset
    val_data = np.memmap(val_input, dtype=dtype, mode="r")[:val_tokens]
    val_data.tofile(val_output)

    print(f"Created train subset: {train_tokens:,} tokens -> {train_output}")
    print(f"Created val subset: {val_tokens:,} tokens -> {val_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create subset of tokenized data")
    parser.add_argument("--train-input", type=Path, required=True, help="Input train.bin")
    parser.add_argument("--val-input", type=Path, required=True, help="Input val.bin")
    parser.add_argument("--train-output", type=Path, required=True, help="Output train.bin")
    parser.add_argument("--val-output", type=Path, required=True, help="Output val.bin")
    parser.add_argument("--train-tokens", type=int, required=True, help="Number of train tokens")
    parser.add_argument("--val-tokens", type=int, required=True, help="Number of val tokens")
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint16",
        choices=["uint16", "uint32"],
        help="Data type (default: uint16)",
    )

    args = parser.parse_args()

    dtype = np.dtype(args.dtype)
    create_subset(
        args.train_input,
        args.val_input,
        args.train_output,
        args.val_output,
        args.train_tokens,
        args.val_tokens,
        dtype,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.extend([
            "--train-input", "data/swallow-code-16k-tokenized/train.bin",
            "--val-input", "data/swallow-code-16k-tokenized/val.bin",
            "--train-output", "data/swallow-code-16k-tokenized-1m/train.bin",
            "--val-output", "data/swallow-code-16k-tokenized-1m/val.bin",
            "--train-tokens", "1000000",
            "--val-tokens", "10000",
        ])
    main()
