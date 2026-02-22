"""Train a causal language model."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from tiny_lm.data.bin import BinDataConfig
from tiny_lm.model import build_model, load_model_config
from tiny_lm.training import CausalLMModule, TrainingConfig
from tiny_lm.training.trainer import run
from tiny_lm.training.tracking.trackio_logger import TrackioLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a causal language model.")
    parser.add_argument("--model-config", help="Path to model config YAML.")
    parser.add_argument("--training-config", help="Path to training config YAML.")
    parser.add_argument("--data-config", help="Path to data config YAML.")
    return parser.parse_args()


def get_git_state() -> dict[str, str | bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        return {"git_sha": sha, "git_dirty": bool(dirty.strip())}
    except (subprocess.SubprocessError, FileNotFoundError):
        return {"git_sha": "unknown", "git_dirty": False}


def build_run_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{Path(args.model_config).stem}-{Path(args.data_config).stem}-{timestamp}"


def copy_run_configs(run_dir: Path, args: argparse.Namespace) -> None:
    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.model_config, configs_dir / Path(args.model_config).name)
    shutil.copy2(args.training_config, configs_dir / Path(args.training_config).name)
    shutil.copy2(args.data_config, configs_dir / Path(args.data_config).name)


def main() -> None:
    args = parse_args()

    model_config = load_model_config(args.model_config)
    training_config = TrainingConfig.from_yaml(args.training_config)
    data_config = BinDataConfig.from_yaml(args.data_config)

    if data_config.block_size > model_config.context_length:
        raise ValueError(
            "block_size cannot exceed model context_length: "
            f"{data_config.block_size} > {model_config.context_length}"
        )

    run_name = training_config.run_name or build_run_name(args)
    run_dir = Path("runs") / run_name
    copy_run_configs(run_dir, args)

    logger = TrackioLogger(
        project=os.getenv("TRACKIO_PROJECT", Path(args.training_config).stem),
        name=run_name,
        config={
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "data_config": asdict(data_config),
            "config_paths": {
                "model": args.model_config,
                "training": args.training_config,
                "data": args.data_config,
            },
            **get_git_state(),
        },
    )

    model = build_model(model_config)
    module = CausalLMModule(model=model, config=training_config)

    run(module, training_config, data_config, run_dir, logger)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--model-config",
                "configs/models/ibis-16.yaml",
                "--training-config",
                "configs/training/swallow-code-8k.yaml",
                "--data-config",
                "configs/data/swallow-code-8k.yaml",
            ]
        )
    main()
