"""Model factory: build the right model from a config."""

from __future__ import annotations

import torch.nn as nn

from tiny_lm.model.config import GPT2Config, Llama3Config


def build_model(model_config: GPT2Config | Llama3Config) -> nn.Module:
    """Instantiate the right model class for the given config."""
    if isinstance(model_config, GPT2Config):
        from tiny_lm.model.gpt2 import GPT2

        return GPT2.from_config(model_config)
    if isinstance(model_config, Llama3Config):
        from tiny_lm.model.llama3 import Llama3

        return Llama3.from_config(model_config)
    raise TypeError(f"Unsupported model config type: {type(model_config)}")
