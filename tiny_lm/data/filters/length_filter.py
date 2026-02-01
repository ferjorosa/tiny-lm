"""Filter dataset examples by text length."""

from typing import Any
from datasets import DatasetDict


class LengthFilter:
    """Filter examples based on text length."""

    def __init__(self, min_length: int | None = None, max_length: int | None = None):
        """
        Args:
            min_length: Minimum text length in characters
            max_length: Maximum text length in characters
        """
        self.min_length = min_length
        self.max_length = max_length

    def apply(self, dataset: DatasetDict, text_field: str) -> DatasetDict:
        """
        Apply length filter to all splits in the dataset.

        Args:
            dataset: Dataset to filter
            text_field: Name of the text field to check length

        Returns:
            Filtered dataset
        """

        def filter_fn(example: dict[str, Any]) -> bool:
            text = example[text_field]
            if self.min_length and len(text) < self.min_length:
                return False
            if self.max_length and len(text) > self.max_length:
                return False
            return True

        filtered_dataset = {}
        for split_name, split_data in dataset.items():
            filtered_dataset[split_name] = split_data.filter(filter_fn)

        return DatasetDict(filtered_dataset)
