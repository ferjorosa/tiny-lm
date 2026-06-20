"""Trackio logger integration for PyTorch Lightning."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Literal

import trackio
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only


def _normalize_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in config.items():
        normalized[key] = asdict(value) if is_dataclass(value) else value
    return normalized


class TrackioLogger(Logger):
    """Minimal Trackio logger for Lightning metrics and configs."""

    def __init__(
        self,
        project: str,
        name: str | None,
        resume: Literal["never", "allow", "must"] = "allow",
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._project = project
        self._name = name
        self._resume = resume
        self._config = _normalize_config(config or {})
        self._run = None

    @property
    def name(self) -> str:
        return "trackio"

    @property
    def version(self) -> str:
        if self._run is None:
            return "unknown"
        return getattr(self._run, "id", "unknown")

    @property
    def experiment(self):
        self._ensure_init()
        return self._run

    @rank_zero_only
    def log_hyperparams(self, params: Any) -> None:
        self._ensure_init()
        if hasattr(self._run, "config"):
            try:
                self._run.config.update(params, allow_val_change=True)
            except Exception:
                pass

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._ensure_init()
        normalized = {key: _normalize_value(value) for key, value in metrics.items()}
        trackio.log(normalized, step=step)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._run is not None:
            trackio.finish()

    def save(self) -> None:
        return None

    def _ensure_init(self) -> None:
        if self._run is not None:
            return
        kwargs: dict[str, Any] = {"project": self._project}
        if self._name:
            kwargs["name"] = self._name
        if self._resume:
            kwargs["resume"] = self._resume
        if self._config:
            kwargs["config"] = self._config
        self._run = trackio.init(**kwargs)
