"""Download a dataset from HuggingFace with progress logs."""

import sys
from pathlib import Path

from tiny_lm.data import load_dataset_from_config
from tiny_lm.data.filters import LengthFilter


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/data/download_dataset.py <config_path>")
        sys.exit(1)

    config_path = Path(sys.argv[1])

    # Download and load dataset
    dataset, text_field = load_dataset_from_config(config_path)

    print("\nDataset loaded:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data):,} examples")

    # Apply length filter
    print("\nApplying length filter (min_length=10)...")
    length_filter = LengthFilter(min_length=10)
    filtered_dataset = length_filter.apply(dataset, text_field)

    print("\nFiltered dataset:")
    for split_name, split_data in filtered_dataset.items():
        original_count = len(dataset[split_name])
        filtered_count = len(split_data)
        removed = original_count - filtered_count
        print(f"  {split_name}: {filtered_count:,} examples ({removed} removed)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("configs/datasets/tinystories.yaml")
    main()
