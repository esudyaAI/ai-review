"""Repository-level configuration for the reviewer."""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable

import yaml


class ConfigError(Exception):
    """Raised when .ai-review.yml contains unusable values."""


@dataclass(frozen=True)
class Limits:
    max_files: int = 60
    max_changed_lines: int = 3000


@dataclass(frozen=True)
class Config:
    test_command: str = ""
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    model_high_risk: str = ""
    high_risk_paths: tuple[str, ...] = ()
    limits: Limits = field(default_factory=Limits)
    # Per-model USD price per million tokens, e.g. {"gpt-x": {"input": 0.75,
    # "output": 4.50, "cached_input": 0.50}}. A model with no entry here still
    # gets its token counts reported, just without a cost figure.
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)

    def validate(self) -> "Config":
        if self.provider != "openai":
            raise ConfigError(f"unsupported provider: {self.provider}")
        if not self.model:
            raise ConfigError("model must not be empty")
        return self

    def model_for(self, changed_files: Iterable[str]) -> str:
        # A single high-risk file pulls the whole review to the stronger model.
        for path in changed_files:
            for pattern in self.high_risk_paths:
                if fnmatch(path, pattern) or fnmatch(path, pattern.replace("/**", "/*")):
                    return self.model_high_risk or self.model
        return self.model


def load_config(path: Path) -> Config:
    if not path.exists():
        # Without this the run dies on Config().validate() with "model must not be
        # empty", which reads as a bad value rather than an absent file. That cost
        # this very repository two weeks of a review job that failed at startup in
        # two seconds under allow_failure, while the documentation said review was
        # connected.
        raise ConfigError(f".ai-review.yml not found at {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    limits_raw = raw.get("limits") or {}
    defaults = Limits()
    config = Config(
        test_command=raw.get("test_command", ""),
        provider=raw.get("provider", "openai"),
        base_url=raw.get("base_url", "https://api.openai.com/v1"),
        model=raw.get("model", ""),
        model_high_risk=raw.get("model_high_risk", ""),
        high_risk_paths=tuple(raw.get("high_risk_paths") or ()),
        limits=Limits(
            max_files=limits_raw.get("max_files", defaults.max_files),
            max_changed_lines=limits_raw.get("max_changed_lines", defaults.max_changed_lines),
        ),
        pricing=raw.get("pricing") or {},
    )
    return config.validate()
