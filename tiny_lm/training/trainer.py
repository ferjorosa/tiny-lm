"""Trainer orchestrator for causal language model training.

Notes:
    - val_check_interval is in training batches, so we multiply by
      accumulate_grad_batches to target optimizer (global) steps.
    - ModelCheckpoint.every_n_train_steps is in optimizer (global) steps.
    - log_every_n_steps controls logger emission frequency in (global) steps.
    - Lightning's default progress bar counts batches; we override it with
      OptimizerStepProgressBar so the bar reflects optimizer steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from tiny_lm.data.bin import BinDataConfig, BinTokenDataModule
from tiny_lm.training.callbacks import (
    GpuStatsMonitor,
    OptimizerStepProgressBar,
    TokensMonitor,
)
from tiny_lm.training.config import TrainingConfig


def run(
    module: pl.LightningModule,
    training_config: TrainingConfig,
    data_config: BinDataConfig,
    run_dir: Path,
    logger: Any = False,
) -> None:
    """Orchestrate Lightning training for a causal language model.

    Builds the data module, sets up callbacks and the Trainer, then runs fit.

    Args:
        module: Pre-built Lightning module wrapping the model.
        training_config: Training hyperparameters and schedule.
        data_config: Data paths and tokenisation settings.
        run_dir: Root directory for checkpoints and logs.
        logger: Lightning logger instance, or False to disable logging.
    """
    data_module = BinTokenDataModule(
        train_path=data_config.train_path,
        val_path=data_config.val_path,
        block_size=data_config.block_size,
        stride=data_config.stride,
        dtype=np.dtype(data_config.dtype),
        eos_token_id=data_config.eos_token_id,
        batch_size=training_config.batch_size,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=data_config.drop_last,
    )

    callbacks = [
        OptimizerStepProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            every_n_train_steps=training_config.save_every_n_steps,
            save_top_k=-1,
            save_last=True,
        ),
        TokensMonitor(log_every_n_steps=training_config.system_metrics_every_n_steps),
        GpuStatsMonitor(log_every_n_steps=training_config.system_metrics_every_n_steps),
    ]

    # val_check_interval is in training batches, so multiply by accumulate_grad_batches
    # to hit the right optimizer step boundary. log_every_n_steps is already in
    # optimizer (global) steps, so no scaling is needed.
    trainer = pl.Trainer(
        default_root_dir=str(run_dir),
        accelerator="auto",
        devices="auto",
        precision=training_config.precision,
        max_steps=training_config.max_steps,
        val_check_interval=training_config.val_every_n_steps
        * training_config.accumulate_grad_batches,
        log_every_n_steps=training_config.system_metrics_every_n_steps,
        accumulate_grad_batches=training_config.accumulate_grad_batches,
        gradient_clip_val=training_config.grad_clip_norm,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(
        module,
        datamodule=data_module,
        ckpt_path=training_config.resume_from_checkpoint,
    )
