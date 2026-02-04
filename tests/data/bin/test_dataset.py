"""Tests for BinTokenDataset."""

import numpy as np
import torch

from tiny_lm.data.bin import BinTokenDataset


class TestBinTokenDataset:
    """Tests for binary token dataset."""

    def test_length_and_shift(self, tmp_path):
        """Dataset length and shift should match expected windows."""
        tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint16)
        bin_path = tmp_path / "train.bin"
        tokens.tofile(bin_path)

        dataset = BinTokenDataset(
            path=bin_path,
            block_size=4,
            stride=2,
            dtype=np.uint16,
        )

        assert len(dataset) == 3

        x, y = dataset[0]
        assert torch.equal(x, torch.tensor([1, 2, 3, 4]))
        assert torch.equal(y, torch.tensor([2, 3, 4, 5]))

    def test_mask_after_eos(self, tmp_path):
        """Targets after first EOS should be masked with -100."""
        eos_token_id = 3
        tokens = np.array([1, 3, 5, 6, 7, 8], dtype=np.uint16)
        bin_path = tmp_path / "val.bin"
        tokens.tofile(bin_path)

        dataset = BinTokenDataset(
            path=bin_path,
            block_size=4,
            stride=4,
            dtype=np.uint16,
            eos_token_id=eos_token_id,
            mask_after_eos=True,
        )

        x, y = dataset[0]
        assert torch.equal(x, torch.tensor([1, 3, 5, 6]))
        assert torch.equal(y, torch.tensor([3, -100, -100, -100]))
