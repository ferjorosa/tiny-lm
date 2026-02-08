"""Export and upload a tiny-lm checkpoint to a public Hugging Face Hub repo.

This script exports the model weights as SafeTensors, copies the model and
tokenizer configs, and uploads everything to a public Hugging Face Hub repo.
It also stores the original Lightning checkpoint for reproducibility.

Expected inputs:
- A Lightning checkpoint (.ckpt).
- The model and training config YAMLs used for the run.
- The tokenizer pickle plus its YAML config.

Output files uploaded:
- model.safetensors (weights)
- model_config.yaml (model settings)
- tokenizer.pkl (tiktoken encoding)
- tokenizer_config.yaml (tokenizer settings)
- checkpoint.ckpt (original checkpoint)
- README.md (usage and architecture summary)
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_model as save_safetensors
from dotenv import load_dotenv
from huggingface_hub import HfApi
import yaml

from tiny_lm.model.architectures.gpt2 import GPT2
from tiny_lm.model.config import GPT2Config
from tiny_lm.utils.precision import precision_to_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload tiny-lm model to HF Hub.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target repo id, e.g. username/tiny-lm-tinystories-8k-2l",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--model-config",
        required=True,
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--training-config",
        required=True,
        help="Path to training config YAML (precision).",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to tokenizer.pkl (tiktoken encoding).",
    )
    parser.add_argument(
        "--tokenizer-config",
        required=True,
        help="Path to tokenizer YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default="hf_export",
        help="Local folder to assemble files before upload.",
    )
    return parser.parse_args()


def load_checkpoint_state(checkpoint_path: str, device: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("model.") for key in state_dict):
        state_dict = {
            key[len("model.") :]: value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
    return state_dict


def build_readme(config: GPT2Config) -> str:
    return f"""---
language: en
tags:
  - gpt2
  - tinystories
  - tiny-lm
datasets:
  - roneneldan/TinyStories
---

# TinyStories GPT-2 (2L, 8k vocab)

This model was trained with the
[tiny-lm](https://github.com/ferjorosa/tiny-lm) repository on the
[TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
(see paper: https://arxiv.org/abs/2305.07759).

## Architecture

- GPT-2 style decoder-only transformer
- Layers: {config.n_layers}
- Vocab size: {config.vocab_size}
- Context length: {config.context_length}
- d_model: {config.d_model}
- n_heads: {config.n_heads}
- d_ff: {config.d_ff}

## Files

- `model.safetensors`: model weights (SafeTensors)
- `model_config.yaml`: tiny-lm model config
- `tokenizer.pkl`: tiktoken encoding
- `tokenizer_config.yaml`: tokenizer settings (BOS/EOS)
- `checkpoint.ckpt`: original Lightning checkpoint

## Usage

This is a tiny-lm model (not Transformers-compatible). Load it with tiny-lm:

```python
import pickle
import torch
from tiny_lm.model.architectures.gpt2 import GPT2
from tiny_lm.model.config import GPT2Config

config = GPT2Config.from_yaml("model_config.yaml")
model = GPT2(
    vocab_size=config.vocab_size,
    d_model=config.d_model,
    n_layers=config.n_layers,
    n_heads=config.n_heads,
    d_ff=config.d_ff,
    context_length=config.context_length,
    emb_dropout=0.0,
    attn_dropout=0.0,
    resid_dropout=0.0,
    ffn_dropout=0.0,
)
from safetensors.torch import load_file as load_safetensors

state = load_safetensors("model.safetensors")
model.load_state_dict(state, strict=True)
model.eval()

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
```
"""


def get_hf_token() -> str:
    load_dotenv()
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    if not token:
        raise RuntimeError(
            "Missing HF token. Set HF_TOKEN (or HUGGINGFACE_TOKEN) in the environment."
        )
    return token


def main(args: argparse.Namespace) -> None:
    device = "cpu"
    checkpoint_path = Path(args.checkpoint)
    model_config_path = Path(args.model_config)
    training_config_path = Path(args.training_config)
    tokenizer_path = Path(args.tokenizer)
    tokenizer_config_path = Path(args.tokenizer_config)

    config = GPT2Config.from_yaml(model_config_path)
    model = GPT2(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        context_length=config.context_length,
        emb_dropout=0.0,
        attn_dropout=0.0,
        resid_dropout=0.0,
        ffn_dropout=0.0,
    )
    state_dict = load_checkpoint_state(str(checkpoint_path), device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    export_dir = Path(args.output_dir) / args.repo_id.replace("/", "__")
    export_dir.mkdir(parents=True, exist_ok=True)

    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    precision = training_config.get("precision")
    export_dtype = precision_to_dtype(precision)

    model_cast = model.to(dtype=export_dtype)
    save_safetensors(model_cast, export_dir / "model.safetensors")
    shutil.copy2(model_config_path, export_dir / "model_config.yaml")
    shutil.copy2(tokenizer_path, export_dir / "tokenizer.pkl")
    shutil.copy2(tokenizer_config_path, export_dir / "tokenizer_config.yaml")
    shutil.copy2(checkpoint_path, export_dir / "checkpoint.ckpt")

    readme_path = export_dir / "README.md"
    readme_path.write_text(build_readme(config), encoding="utf-8")

    token = get_hf_token()
    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        private=False,
        exist_ok=True,
        token=token,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(export_dir),
        token=token,
        commit_message="Upload tiny-lm TinyStories model",
    )

    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--repo-id",
                "ferjorosa/tiny-lm-tinystories-8k-gpt2-2l",
                "--checkpoint",
                "runs/gpt2-8k-2l-tinystories-8k-20260208-103335/checkpoints/last.ckpt",
                "--model-config",
                "runs/gpt2-8k-2l-tinystories-8k-20260208-103335/configs/gpt2-8k-2l.yaml",
                "--training-config",
                "runs/gpt2-8k-2l-tinystories-8k-20260208-103335/configs/gpt2-8k.yaml",
                "--tokenizer",
                "tokenizers/tinystories-8k/tokenizer.pkl",
                "--tokenizer-config",
                "configs/tokenizers/tinystories-8k.yaml",
            ]
        )
    parsed_args = parse_args()
    main(parsed_args)
