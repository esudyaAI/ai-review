"""Repository-level configuration for the reviewer."""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable

import yaml

KNOWN_LINTERS = frozenset({"golangci-lint", "eslint", "ruff", "shellcheck", "none"})


class ConfigError(Exception):
    """Raised when .ai-review.yml contains unusable values."""


@dataclass(frozen=True)
class Limits:
    max_files: int = 60
    max_changed_lines: int = 3000


@dataclass(frozen=True)
class Config:
    linter: str = "none"
    test_command: str = ""
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    model_high_risk: str = ""
    high_risk_paths: tuple[str, ...] = ()
    limits: Limits = field(default_factory=Limits)

    def validate(self) -> "Config":
        if self.linter not in KNOWN_LINTERS:
            raise ConfigError(f"unknown linter: {self.linter}")
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
        return Config().validate()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    limits_raw = raw.get("limits") or {}
    defaults = Limits()
    config = Config(
        linter=raw.get("linter", "none"),
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
    )
    return config.validate()
