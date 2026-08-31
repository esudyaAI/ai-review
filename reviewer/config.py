"""Repository-level configuration for the reviewer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Iterable

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
    # Regular expressions that make a file high-risk by what is written in it,
    # whatever its path.
    #
    # Added because `migrations/*` in high_risk_paths sent EVERY migration to the
    # strong model. Measured 2026-08-31: a migration whose whole body was
    # `ALTER TABLE ... ADD COLUMN IF NOT EXISTS location text` cost $0.1258
    # against $0.03 on the base model.
    #
    # Adding a column breaks no reader. Narrowing the schema does: DROP COLUMN,
    # RENAME, SET NOT NULL, ALTER COLUMN ... TYPE, DELETE FROM, TRUNCATE. Those
    # are what the expensive model is for, and they are a property of the
    # content, not of the directory the file sits in.
    #
    # Matched case-insensitively: SQL is written both ways.
    high_risk_content: tuple[str, ...] = ()
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

    def is_high_risk(self, path: str, content: str | None = None) -> bool:
        """High risk by path, or by content when the content is available.

        content=None is not "safe" -- it is "unknown", and the decision falls
        back to the path. A deleted file, or one that could not be read, must
        not become cheap to review by virtue of being unreadable.
        """
        if any(
            fnmatch(path, pattern) or fnmatch(path, pattern.replace("/**", "/*"))
            for pattern in self.high_risk_paths
        ):
            return True
        if content is None or not self.high_risk_content:
            return False
        return any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in self.high_risk_content
        )

    def model_for(
        self,
        changed_files: Iterable[str],
        read_content: "Callable[[str], str | None] | None" = None,
    ) -> str:
        # Still used for adjudication, where the subject is a single file.
        return self.model_high_risk or self.model if any(
            self.is_high_risk(path, read_content(path) if read_content else None)
            for path in changed_files
        ) else self.model

    def split_by_risk(
        self,
        changed_files: Iterable[str],
        read_content: "Callable[[str], str | None] | None" = None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """(high risk, everything else), original order preserved.

        A single high-risk file used to pull the whole diff to the stronger model:
        one auth file in a nine-file merge request meant all nine were reviewed at
        $30/M output instead of $4.50/M. Splitting lets each half be reviewed by the
        model its own risk deserves, without narrowing high_risk_paths - narrowing
        would buy the same saving by looking away from risky files.

        With no distinct strong model there is nothing to route to, and a second
        pass would only double the fixed cost of a review, so everything stays in
        one half.
        """
        paths = tuple(changed_files)
        if not self.model_high_risk or self.model_high_risk == self.model:
            return (), paths
        # Read once per file: reading is a disk hit, and the answer is used twice.
        risky = {
            p: self.is_high_risk(p, read_content(p) if read_content else None)
            for p in paths
        }
        high = tuple(p for p in paths if risky[p])
        rest = tuple(p for p in paths if not risky[p])
        return high, rest


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
        high_risk_content=tuple(raw.get("high_risk_content") or ()),
        limits=Limits(
            max_files=limits_raw.get("max_files", defaults.max_files),
            max_changed_lines=limits_raw.get("max_changed_lines", defaults.max_changed_lines),
        ),
        pricing=raw.get("pricing") or {},
    )
    return config.validate()
