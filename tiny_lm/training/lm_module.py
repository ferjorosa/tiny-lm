"""Lightning module for causal language modeling."""

from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
from torch import nn

from tiny_lm.training.config import TrainingConfig


class CausalLMModule(pl.LightningModule):
    """LightningModule for causal language modeling.

    Expects a model that returns logits of shape (batch, seq_len, vocab_size).

    Args:
        model: Causal language model returning token logits.
        config: Training configuration.
    """

    def __init__(self, model: nn.Module, config: TrainingConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=config.ignore_index)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)

    def _common_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        input_ids, targets = batch
        logits = self(input_ids)
        vocab_size = logits.size(-1)
        loss = self.loss_fn(logits.view(-1, vocab_size), targets.view(-1))
        return loss

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss = self._common_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss = self._common_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
            eps=self.config.eps,
        )
        if self.config.scheduler == "none":
            return optimizer

        min_lr_ratio = self.config.min_lr / self.config.learning_rate
        warmup_steps = self.config.warmup_steps
        max_steps = self.config.max_steps

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return step / warmup_steps
            if max_steps <= warmup_steps:
                return min_lr_ratio
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
