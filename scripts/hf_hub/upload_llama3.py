"""Export and upload a tiny-lm LLaMA 3 checkpoint to a public Hugging Face Hub repo.

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

from tiny_lm.model.llama3 import Llama3
from tiny_lm.model.config import Llama3Config
from tiny_lm.utils.precision import precision_to_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload tiny-lm LLaMA 3 model to HF Hub.")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target repo id, e.g. username/tiny-lm-swallow-code-8k-ibis-16",
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


def build_readme(config: Llama3Config) -> str:
    return f"""---
language:
  - en
  - ja
tags:
  - llama3
  - swallow-code
  - tiny-lm
  - code
datasets:
  - tokyotech-llm/swallow-code
---

# Swallow Code Ibis-16 (LLaMA 3, 16 layers, 8k vocab)

This model was trained with the
[tiny-lm](https://github.com/ferjorosa/tiny-lm) repository on the
[SwallowCode dataset](https://huggingface.co/datasets/tokyotech-llm/swallow-code).

The goal is educational: a compact pretraining run for studying tokenization,
data pipelines, and transformer training end to end.

## Data

The tokenizer is a custom 8k-token BPE trained on SwallowCode with Karpathy's
[`rustbpe`](https://github.com/karpathy/rustbpe) approach, then exported as a
`tiktoken` encoding for inference.

The dataset was split into 99% train and 1% validation before tokenization.
The resulting tokenized files contain 25.95B train tokens and 262M validation
tokens. Training used 1,024-token contiguous windows over the token stream.

## Training

The model was trained with PyTorch Lightning using bf16 mixed precision.

- Context window: 1,024 tokens
- Batch size: 64 sequences
- Gradient accumulation: 4
- Effective batch size: 262,144 tokens per optimizer step
- Training budget: about 25B tokens over 95,350 optimizer steps
- Optimizer: AdamW with cosine LR decay and 1% warmup
- Peak LR: 6e-4
- Weight decay: 0.1

## Architecture

- LLaMA 3 style decoder-only transformer
- Parameters: 18,886,912 total; 16,789,760 non-embedding
- Layers: {config.n_layers}
- Vocab size: {config.vocab_size}
- Context length: {config.context_length}
- d_model: {config.d_model}
- n_heads: {config.n_heads}
- n_kv_heads: {config.n_kv_heads}
- ffn_hidden_dim: {config.ffn_hidden_dim or (4 * config.d_model)}
- RoPE theta: {config.rope_theta}
- Norm epsilon: {config.norm_eps}

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
from tiny_lm.model.llama3 import Llama3
from tiny_lm.model.config import Llama3Config

config = Llama3Config.from_yaml("model_config.yaml")
model = Llama3(
    vocab_size=config.vocab_size,
    d_model=config.d_model,
    n_layers=config.n_layers,
    n_heads=config.n_heads,
    context_length=config.context_length,
    n_kv_heads=config.n_kv_heads,
    ffn_hidden_dim=config.ffn_hidden_dim,
    multiple_of=config.multiple_of,
    rope_theta=config.rope_theta,
    norm_eps=config.norm_eps,
    emb_dropout=0.0,
    attn_dropout=0.0,
    resid_dropout=0.0,
    ffn_dropout=0.0,
    qkv_bias=config.qkv_bias,
    ffn_bias=config.ffn_bias,
    attn_backend=config.attn_backend,
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

    config = Llama3Config.from_yaml(model_config_path)
    model = Llama3(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        context_length=config.context_length,
        n_kv_heads=config.n_kv_heads,
        ffn_hidden_dim=config.ffn_hidden_dim,
        multiple_of=config.multiple_of,
        rope_theta=config.rope_theta,
        norm_eps=config.norm_eps,
        emb_dropout=0.0,
        attn_dropout=0.0,
        resid_dropout=0.0,
        ffn_dropout=0.0,
        qkv_bias=config.qkv_bias,
        ffn_bias=config.ffn_bias,
        attn_backend=config.attn_backend,
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
        commit_message="Upload tiny-lm LLaMA 3 Swallow Code model",
    )

    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--repo-id",
                "ferjorosa/tiny-lm-swallow-code-8k-ibis-16",
                "--checkpoint",
                "runs/ibis-16-swallow-code-8k-20260218-174533/checkpoints/last.ckpt",
                "--model-config",
                "runs/ibis-16-swallow-code-8k-20260218-174533/configs/ibis-16.yaml",
                "--training-config",
                "configs/training/swallow-code-8k.yaml",
                "--tokenizer",
                "tokenizers/swallow-code-8k/tokenizer.pkl",
                "--tokenizer-config",
                "configs/tokenizers/swallow-code-8k.yaml",
            ]
        )
    parsed_args = parse_args()
    main(parsed_args)
