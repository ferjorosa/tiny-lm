"""Train a causal language model with Lightning."""

from __future__ import annotations

import argparse

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from tiny_lm.data.bin import BinDataConfig, BinTokenDataModule
from tiny_lm.model.architectures.gpt2 import GPT2
from tiny_lm.model.config import GPT2Config
from tiny_lm.training import CausalLMModule, TrainingConfig


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

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            save_top_k=training_config.save_top_k,
            every_n_train_steps=training_config.save_every_n_steps,
            monitor="val_loss",
            mode="min",
        ),
    ]

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision=training_config.precision,
        max_steps=training_config.max_steps,
        accumulate_grad_batches=training_config.accumulate_grad_batches,
        gradient_clip_val=training_config.grad_clip_norm,
        callbacks=callbacks,
    )

    trainer.fit(module, datamodule=data_module)


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
