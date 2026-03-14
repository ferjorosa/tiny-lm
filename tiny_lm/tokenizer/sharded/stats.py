"""Length-stat aggregation utilities for tokenized dataset splits."""

from __future__ import annotations

import sys


def init_length_stats() -> dict[str, int]:
    return {
        "char_count": 0,
        "token_count": 0,
        "min_char_len": sys.maxsize,
        "max_char_len": 0,
        "min_token_len": sys.maxsize,
        "max_token_len": 0,
    }


def update_length_stats(stats: dict[str, int], char_len: int, token_len: int) -> None:
    stats["char_count"] += char_len
    stats["token_count"] += token_len
    stats["min_char_len"] = min(stats["min_char_len"], char_len)
    stats["max_char_len"] = max(stats["max_char_len"], char_len)
    stats["min_token_len"] = min(stats["min_token_len"], token_len)
    stats["max_token_len"] = max(stats["max_token_len"], token_len)


def merge_length_stats(dst: dict[str, int], src: dict[str, int]) -> None:
    dst["char_count"] += src["char_count"]
    dst["token_count"] += src["token_count"]
    dst["min_char_len"] = min(dst["min_char_len"], src["min_char_len"])
    dst["max_char_len"] = max(dst["max_char_len"], src["max_char_len"])
    dst["min_token_len"] = min(dst["min_token_len"], src["min_token_len"])
    dst["max_token_len"] = max(dst["max_token_len"], src["max_token_len"])


def empty_length_stats_summary() -> dict[str, dict[str, float]]:
    return {
        "char": {"min": 0.0, "max": 0.0, "mean": 0.0},
        "token": {"min": 0.0, "max": 0.0, "mean": 0.0},
    }


def finalize_length_stats(
    stats: dict[str, int], example_count: int
) -> dict[str, dict[str, float]]:
    if example_count == 0:
        return empty_length_stats_summary()

    char_count = int(stats["char_count"])
    token_count = int(stats["token_count"])
    return {
        "char": {
            "min": float(stats["min_char_len"]),
            "max": float(stats["max_char_len"]),
            "mean": float(char_count / example_count),
        },
        "token": {
            "min": float(stats["min_token_len"]),
            "max": float(stats["max_token_len"]),
            "mean": float(token_count / example_count),
        },
    }
