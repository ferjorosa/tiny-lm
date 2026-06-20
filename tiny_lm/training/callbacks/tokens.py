"""Token logging callback."""

from __future__ import annotations

import time

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


class TokensMonitor(pl.Callback):
    """Log tokens/sec and tokens processed."""

    def __init__(self, log_every_n_steps: int = 1) -> None:
        if log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be positive")
        self.log_every_n_steps = log_every_n_steps
        self._last_time: float | None = None
        self._total_tokens: int = 0
        self._pending_tokens: int = 0
        self._tokens_since_log: int = 0

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        input_ids = batch[0]
        # World size represents the number of devices in the training run.
        self._pending_tokens += input_ids.numel() * trainer.world_size

    @rank_zero_only
    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        step_tokens = self._pending_tokens
        self._pending_tokens = 0
        self._total_tokens += step_tokens
        self._tokens_since_log += step_tokens

        if (trainer.global_step + 1) % self.log_every_n_steps != 0:
            return

        now = time.perf_counter()
        if self._last_time is None:
            self._last_time = now
            self._tokens_since_log = 0
            return
        elapsed = now - self._last_time
        self._last_time = now
        if elapsed <= 0:
            return

        tokens_per_sec = self._tokens_since_log / elapsed
        self._tokens_since_log = 0
        pl_module.log(
            "tokens_per_sec",
            tokens_per_sec,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        pl_module.log(
            "tokens_processed",
            float(self._total_tokens),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
