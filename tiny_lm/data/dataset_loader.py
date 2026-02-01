"""Dataset loading utilities with config-based approach."""

import yaml
from pathlib import Path
from typing import Any
from datasets import load_dataset, DatasetDict


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset_from_config(config_path: str | Path) -> tuple[DatasetDict, str]:
    """
    Load a dataset based on a YAML configuration file.

    Args:
        config_path: Path to the dataset configuration YAML file

    Returns:
        Tuple of (dataset, text_field_name)
    """
    config = load_yaml_config(config_path)

    # Load dataset based on source
    source = config.get("source", "huggingface")
    if source == "huggingface":
        dataset = load_dataset(config["name"])
    elif source == "local":
        dataset = load_dataset(config["path"])
    else:
        raise ValueError(f"Unknown source: {source}")

    # Get text field name
    text_field = config["text_field"]

    return dataset, text_field
