"""Precision helpers shared across training and export."""

from __future__ import annotations

from typing import Literal

import torch


PrecisionName = Literal["bf16", "fp16", "fp32", "fp64"]


def resolve_precision_name(precision: object) -> PrecisionName:
    if precision in {"bf16-mixed", "bf16-true"}:
        return "bf16"
    if precision in {"16-mixed", "16"} or precision == 16:
        return "fp16"
    if precision in {"32-true", "32"} or precision == 32:
        return "fp32"
    if precision in {"64"} or precision == 64:
        return "fp64"
    raise ValueError(f"Unsupported precision: {precision}")


def precision_to_dtype(precision: object) -> torch.dtype:
    precision_name = resolve_precision_name(precision)
    if precision_name == "bf16":
        return torch.bfloat16
    if precision_name == "fp16":
        return torch.float16
    if precision_name == "fp64":
        return torch.float64
    return torch.float32
