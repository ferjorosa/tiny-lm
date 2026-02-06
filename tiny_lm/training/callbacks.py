"""Training callbacks."""

from __future__ import annotations

import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only


class TokensAndMemoryMonitor(pl.Callback):
    """Log tokens/sec and GPU memory usage."""

    def __init__(self, log_every_n_steps: int = 1) -> None:
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be positive")
        self.log_every_n_steps = log_every_n_steps
        self._last_time: float | None = None

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if (trainer.global_step + 1) % self.log_every_n_steps != 0:
            return

        now = time.perf_counter()
        if self._last_time is None:
            self._last_time = now
            return
        elapsed = now - self._last_time
        self._last_time = now
        if elapsed <= 0:
            return

        if isinstance(batch, (tuple, list)) and batch:
            input_ids = batch[0]
            tokens = input_ids.numel()
            tokens_per_sec = tokens / elapsed
            pl_module.log(
                "tokens_per_sec",
                tokens_per_sec,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

        if torch.cuda.is_available() and pl_module.device.type == "cuda":
            device = pl_module.device
            mem_mb = torch.cuda.memory_allocated(device) / (1024**2)
            max_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            pl_module.log(
                "gpu_mem_mb",
                mem_mb,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
            pl_module.log(
                "gpu_mem_max_mb",
                max_mem_mb,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )
