"""Dataset loading utilities with config-based approach."""

from pathlib import Path

from datasets import DatasetDict, load_dataset

from tiny_lm.data.config import DatasetConfig


def load_dataset_from_config(
    config_path: str | Path,
) -> tuple[DatasetDict, DatasetConfig]:
    """
    Load a dataset based on a YAML configuration file.

    Args:
        config_path: Path to the dataset configuration YAML file

    Returns:
        Tuple of (dataset, dataset_config)
    """
    config = DatasetConfig.from_yaml(config_path)

    # Load dataset based on source
    if config.source == "huggingface":
        dataset = load_dataset(config.name)
    elif config.source == "local":
        dataset = load_dataset(config.name)
    else:
        raise ValueError(f"Unknown source: {config.source}")

    return dataset, config
