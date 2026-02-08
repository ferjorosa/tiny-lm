"""Minimal GPT-2 inference script from SafeTensors weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pickle

import tiktoken
import torch
from safetensors.torch import load_file as load_safetensors

from tiny_lm.model.architectures.gpt2 import GPT2
from tiny_lm.model.config import GPT2Config
from tiny_lm.tokenizer.config import TokenizerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal GPT-2 inference.")
    parser.add_argument(
        "--safetensors",
        type=str,
        required=True,
        help="Path to SafeTensors weights (.safetensors).",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer.pkl (tiktoken encoding).",
    )
    parser.add_argument(
        "--tokenizer-config",
        type=str,
        required=True,
        help="Path to tokenizer YAML config (sets BOS/EOS tokens).",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--add-bos",
        action="store_true",
        help="Prepend BOS token if the tokenizer defines one.",
    )
    return parser.parse_args()


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits
    return logits / temperature


def sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return torch.multinomial(probs, num_samples=1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative > top_p
    cutoff[..., 1:] = cutoff[..., :-1].clone()
    cutoff[..., 0] = False
    sorted_probs[cutoff] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled)


@torch.no_grad()
def generate(
    model: GPT2,
    tokenizer: tiktoken.Encoding,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
    add_bos: bool,
    bos_token_id: int | None,
    eos_token_id: int | None,
) -> str:
    prompt_ids: list[int] = []
    if add_bos and bos_token_id is not None:
        prompt_ids.append(bos_token_id)
    prompt_ids.extend(tokenizer.encode_ordinary(prompt))
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    context_length = model.context_length

    for _ in range(max_new_tokens):
        if input_ids.shape[1] > context_length:
            input_ids = input_ids[:, -context_length:]
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]
        if temperature <= 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token_logits = apply_temperature(next_token_logits, temperature)
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = sample_top_p(probs, top_p)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return tokenizer.decode(input_ids[0].tolist())


def main() -> None:
    args = parse_args()
    weights_path = Path(args.safetensors)
    if not weights_path.exists():
        raise FileNotFoundError(f"SafeTensors file not found: {weights_path}")

    config = GPT2Config.from_yaml(args.model_config)
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
    state_dict = load_safetensors(str(weights_path))
    # save_model stores tied weights only once. In GPT-2 models the input
    # embeddings and lm_head share the same tensor, so the safetensors file
    # only keeps one key. Re-create the missing alias so load_state_dict
    # sees both names.
    if "token_emb.weight" not in state_dict and "lm_head.weight" in state_dict:
        state_dict["token_emb.weight"] = state_dict["lm_head.weight"]
    model.load_state_dict(state_dict, strict=True)
    model.to(args.device)
    model.eval()

    tok_config = TokenizerConfig.from_yaml(args.tokenizer_config)
    with open(args.tokenizer, "rb") as f:
        tokenizer = pickle.load(f)
    bos_token_id = tokenizer.encode_single_token(tok_config.special_tokens["bos"])
    eos_token_id = tokenizer.encode_single_token(tok_config.special_tokens["eos"])
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
        add_bos=args.add_bos,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )
    print(output)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--safetensors",
                "hf_export/ferjorosa__tiny-lm-tinystories-8k-gpt2-2l/model.safetensors",
                "--model-config",
                "configs/models/gpt2-8k-2l.yaml",
                "--tokenizer",
                "tokenizers/tinystories-8k/tokenizer.pkl",
                "--tokenizer-config",
                "configs/tokenizers/tinystories-8k.yaml",
                "--prompt",
                "Once upon a time",
                "--max-new-tokens",
                "300",
                "--temperature",
                "0.8",
                "--top-p",
                "0.95",
                "--add-bos",
            ]
        )
    main()
