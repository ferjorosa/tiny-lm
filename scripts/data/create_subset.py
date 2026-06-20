"""Create a token-count subset from an existing tokenized dataset directory."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_metadata(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _write_subset_bin(
    input_path: Path, output_path: Path, num_tokens: int, dtype: np.dtype
) -> None:
    """Read first `num_tokens` from input and write them to output."""
    file_size = input_path.stat().st_size
    available_tokens = file_size // dtype.itemsize
    if num_tokens > available_tokens:
        raise ValueError(
            f"Requested {num_tokens:,} tokens from {input_path}, "
            f"but file has only {available_tokens:,} tokens."
        )

    data = np.memmap(input_path, dtype=dtype, mode="r")[:num_tokens]
    data.tofile(output_path)


def create_subset(
    input_dir: Path,
    output_dir: Path,
    train_tokens: int,
    val_tokens: int,
) -> None:
    """
    Create subset `train.bin`, `val.bin`, and `metadata.json` in `output_dir`.

    Expected files in `input_dir`: `train.bin`, `val.bin`, `metadata.json`.
    """
    input_train = input_dir / "train.bin"
    input_val = input_dir / "val.bin"
    input_metadata = input_dir / "metadata.json"

    output_train = output_dir / "train.bin"
    output_val = output_dir / "val.bin"
    output_metadata = output_dir / "metadata.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _read_metadata(input_metadata)
    dtype = np.dtype(metadata.get("dtype", "uint16"))

    _write_subset_bin(input_train, output_train, train_tokens, dtype)
    _write_subset_bin(input_val, output_val, val_tokens, dtype)

    metadata["dtype"] = str(dtype)
    metadata["train_tokens"] = train_tokens
    metadata["val_tokens"] = val_tokens
    # Subsetting by token count breaks document boundaries, so original
    # document-level example counts are not meaningful for this output.
    metadata["train_examples"] = -1
    metadata["val_examples"] = -1

    with open(output_metadata, "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(f"Created train subset: {train_tokens:,} tokens -> {output_train}")
    print(f"Created val subset:   {val_tokens:,} tokens -> {output_val}")
    print(f"Created metadata: {output_metadata}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a token-count subset from a tokenized dataset directory.\n"
            "Input dir must contain train.bin, val.bin, metadata.json."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory with train.bin, val.bin, metadata.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where subset files will be written",
    )
    parser.add_argument(
        "--train-tokens",
        type=int,
        required=True,
        help="Number of tokens to keep in train.bin",
    )
    parser.add_argument(
        "--val-tokens",
        type=int,
        required=True,
        help="Number of tokens to keep in val.bin",
    )
    args = parser.parse_args()

    create_subset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_tokens=args.train_tokens,
        val_tokens=args.val_tokens,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--input-dir",
                "data/swallow-code-8k-tokenized",
                "--output-dir",
                "data/swallow-code-8k-tokenized-val",
                "--train-tokens",
                "5000000",
                "--val-tokens",
                "20000000",
            ]
        )
    main()
