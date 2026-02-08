"""Train a causal language model with Lightning.

Notes:
    - val_check_interval is in training batches, so we multiply by
      accumulate_grad_batches to target optimizer (global) steps.
    - ModelCheckpoint.every_n_train_steps is in optimizer (global) steps.
    - log_every_n_steps controls logger emission frequency in (global) steps.
    - Lightning's default progress bar counts batches; we override it with
      OptimizerStepProgressBar so the bar reflects optimizer steps.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from tiny_lm.data.bin import BinDataConfig, BinTokenDataModule
from tiny_lm.model.architectures.gpt2 import GPT2
from tiny_lm.model.config import GPT2Config
from tiny_lm.training import (
    CausalLMModule,
    GpuStatsMonitor,
    OptimizerStepProgressBar,
    TokensMonitor,
    TrainingConfig,
)
from tiny_lm.tracking.trackio_logger import TrackioLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a causal language model.")
    parser.add_argument(
        "--model-config",
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--training-config",
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--data-config",
        help="Path to data config YAML.",
    )
    return parser.parse_args()


def build_model(config: GPT2Config) -> GPT2:
    return GPT2(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        context_length=config.context_length,
        emb_dropout=config.emb_dropout,
        attn_dropout=config.attn_dropout,
        resid_dropout=config.resid_dropout,
        ffn_dropout=config.dropout,
    )


def get_git_state() -> dict[str, str | bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        return {"git_sha": sha, "git_dirty": bool(dirty.strip())}
    except (subprocess.SubprocessError, FileNotFoundError):
        return {"git_sha": "unknown", "git_dirty": False}


def build_trackio_config(
    model_config: GPT2Config,
    training_config: TrainingConfig,
    data_config: BinDataConfig,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "data_config": asdict(data_config),
        "config_paths": {
            "model": args.model_config,
            "training": args.training_config,
            "data": args.data_config,
        },
        **get_git_state(),
    }


def build_trackio_run_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_name = Path(args.model_config).stem
    data_name = Path(args.data_config).stem
    return f"{model_name}-{data_name}-{timestamp}"


def copy_run_configs(run_dir: Path, args: argparse.Namespace) -> None:
    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.model_config, configs_dir / Path(args.model_config).name)
    shutil.copy2(args.training_config, configs_dir / Path(args.training_config).name)
    shutil.copy2(args.data_config, configs_dir / Path(args.data_config).name)


def main() -> None:
    args = parse_args()

    model_config = GPT2Config.from_yaml(args.model_config)
    training_config = TrainingConfig.from_yaml(args.training_config)
    data_config = BinDataConfig.from_yaml(args.data_config)

    if data_config.block_size > model_config.context_length:
        raise ValueError(
            "block_size cannot exceed model context_length: "
            f"{data_config.block_size} > {model_config.context_length}"
        )

    model = build_model(model_config)
    module = CausalLMModule(model=model, config=training_config)

    data_module = BinTokenDataModule(
        train_path=data_config.train_path,
        val_path=data_config.val_path,
        block_size=data_config.block_size,
        stride=data_config.stride,
        dtype=np.dtype(data_config.dtype),
        eos_token_id=data_config.eos_token_id,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=data_config.drop_last,
    )

    run_name = training_config.run_name or build_trackio_run_name(args)
    run_dir = Path("runs") / run_name
    checkpoints_dir = run_dir / "checkpoints"
    copy_run_configs(run_dir, args)

    callbacks = [
        OptimizerStepProgressBar(),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=str(checkpoints_dir),
            every_n_train_steps=training_config.save_every_n_steps,
            save_top_k=-1,
            save_last=True,
        ),
        TokensMonitor(log_every_n_steps=training_config.system_metrics_every_n_steps),
        GpuStatsMonitor(log_every_n_steps=training_config.system_metrics_every_n_steps),
    ]
    trackio_logger = TrackioLogger(
        project=os.getenv("TRACKIO_PROJECT", "tiny-lm"),
        name=run_name,
        config=build_trackio_config(
            model_config=model_config,
            training_config=training_config,
            data_config=data_config,
            args=args,
        ),
    )

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
        logger=trackio_logger,
    )

    trainer.fit(
        module,
        datamodule=data_module,
        ckpt_path=training_config.resume_from_checkpoint,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--model-config",
                "configs/models/gpt2-small.yaml",
                "--training-config",
                "configs/training/gpt2-small.yaml",
                "--data-config",
                "configs/data/tinystories-8k.yaml",
            ]
        )
    main()
