"""CLI entrypoint for sharded dataset tokenization."""

from __future__ import annotations

import argparse
import sys

from tiny_lm.tokenizer.sharded_tokenizer import tokenize_dataset_sharded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize dataset and save as sharded binary files."
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
    parser.add_argument(
        "--shard-size-mb",
        type=int,
        default=1024,
        help="Target shard size in MB.",
    )
    parser.add_argument(
        "--flush-buffer-mb",
        type=int,
        default=128,
        help="In-memory write buffer size in MB before flushing to disk.",
    )
    parser.add_argument(
        "--tokenize-workers",
        type=int,
        default=12,
        help="Number of parallel workers used for tokenization.",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,  # default is workers * 2
        help="Maximum number of in-flight tokenization batches (default: workers * 2).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2_000,
        help="Number of texts per tokenization batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenize_dataset_sharded(
        tokenizer_config=args.config,
        seed=args.seed,
        shard_size_mb=args.shard_size_mb,
        flush_buffer_mb=args.flush_buffer_mb,
        tokenize_workers=args.tokenize_workers,
        max_in_flight=args.max_in_flight,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["--config", "configs/tokenizers/swallow-code-16k-sharded-v1.yaml"])
    main()
