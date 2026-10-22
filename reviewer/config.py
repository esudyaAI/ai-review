"""Repository-level configuration for the reviewer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Iterable, Mapping

import yaml


class ConfigError(Exception):
    """Raised when .ai-review.yml contains unusable values."""


SUPPORTED_PROVIDERS = frozenset({"openai", "anthropic"})

# Models a provider gets when the group switch moves a repository to it without
# naming models. The repositories' own .ai-review.yml name OpenAI models, which
# mean nothing to Anthropic, so the switch has to bring its own.
#
# Sonnet for the ordinary half of a diff, Opus for the high-risk half - chosen
# 2026-10-22 for the prepaid Claude budget that ends on 2026-10-31.
DEFAULT_MODELS = {
    "anthropic": ("claude-sonnet-5-5", "claude-opus-5-5"),
}

# USD per million tokens, from platform.claude.com/docs/en/about-claude/pricing,
# read 2026-10-22. "cache_write" is the 5-minute cache write; "cached_input" is a
# cache hit. Built in because the repositories price only their OpenAI models,
# and a switched run would otherwise report every Claude call as unpriced -
# while the reason to switch is to spend a fixed budget and watch it.
#
# Haiku's price here is its tier for prompts up to 100K tokens; above that it is
# five times higher. Haiku is not used by default, so the line under-prices
# only a run that chose it explicitly and sent a long prompt.
#
# The OpenAI rows are from developers.openai.com/api/docs/models/<model>, read
# 2026-10-22, standard tier. They exist to fill keys a repository left out: all
# three priced gpt-5.4-mini without cached_input, so its cached tokens (~85 % of
# the prompt) were billed at the full input rate in the cost line. Above 272K
# tokens in one prompt gpt-5.5 costs 2x input and 1.5x output; the line does
# not model that tier.
BUILTIN_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cached_input": 0.20, "cache_write": 5.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cached_input": 0.10, "cache_write": 2.50},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cached_input": 0.01, "cache_write": 0.125},
    "gpt-5.4-mini-2026-03-17": {"input": 0.75, "output": 4.50, "cached_input": 0.075},
    "gpt-5.5-2026-04-23": {"input": 5.00, "output": 30.00, "cached_input": 0.50},
}


def _fill_price_keys(pricing: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """The repository's prices with keys it left out filled from BUILTIN_PRICING.

    A key the repository wrote wins. A model it did not price is not added:
    filling keys is not inventing entries.
    """
    return {
        model: {**BUILTIN_PRICING.get(model, {}), **(prices or {})}
        for model, prices in pricing.items()
    }


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
    # Файлы, изменение которых само по себе ревьюировать незачем: документация,
    # лицензия, картинки. Если диффом задеты ТОЛЬКО они, прогон отменяется до
    # первого обращения к модели.
    #
    # Замер по 20 последним смёрженным MR jobbytime-go: 4 из 20 не трогали
    # ничего, кроме .md. Ревью каждого стоило столько же, сколько ревью кода.
    #
    # Умолчание пустое намеренно. В одном репозитории .md — заметки, в другом —
    # содержимое сайта или контракт API. Угадывать за репозиторий значит однажды
    # молча не проверить то, что проверить стоило.
    #
    # CLAUDE.md в такой список класть не следует, и это не мелочь: он задаёт
    # правила, по которым ревьюируется весь остальной код. Правка в нём меняет
    # поведение всех будущих прогонов — это ближе к коду, чем к документации.
    skip_paths: tuple[str, ...] = ()
    # Файлы, которые собирает закоммиченный сборщик из закоммиченных входных
    # данных и про которые в репозитории есть тест, что они ему всё ещё равны.
    #
    # Это НЕ skip_paths пошире. skip_paths отвечает на вопрос «стоит ли вообще
    # запускать ревью этого merge request», и .md в дифф с кодом попадает
    # намеренно: README нередко и есть то место, где видно, что изменение
    # описано неверно. Здесь вопрос другой — читать такой файл незачем в любом
    # диффе: решений в нём нет, они в сборщике, а восемь тысяч строк стоят
    # настоящих денег за токены.
    #
    # Поэтому эти файлы уходят из диффа ДО проверки размера и до промпта.
    # Замер, ради которого список заведён: jobbytime-matcher !133, 21.09.2026 —
    # «изменено 8924 строк, лимит 3000», код выхода 3, не прочитано ничего,
    # включая сам сборщик; 8000 строк из 8924 — собранный им словарь.
    #
    # Умолчание пустое: незнакомый файл сюда не попадает и ревью получает.
    generated_paths: tuple[str, ...] = ()
    limits: Limits = field(default_factory=Limits)
    # Per-model USD price per million tokens, e.g. {"gpt-x": {"input": 0.75,
    # "output": 4.50, "cached_input": 0.50}}. A model with no entry here still
    # gets its token counts reported, just without a cost figure.
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)

    def validate(self) -> "Config":
        if self.provider not in SUPPORTED_PROVIDERS:
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

    def partition_generated(
        self, changed_files: Iterable[str]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """(files to review, generated files), original order preserved.

        An empty `generated_paths` leaves everything on the left: a repository
        that has not named its generated files has none as far as the reviewer
        is concerned.
        """
        paths = tuple(changed_files)
        if not self.generated_paths:
            return paths, ()
        generated = tuple(path for path in paths if self._is_generated(path))
        if not generated:
            return paths, ()
        remaining = tuple(path for path in paths if path not in set(generated))
        return remaining, generated

    def _is_generated(self, path: str) -> bool:
        return any(
            fnmatch(path, pattern) or fnmatch(path, pattern.replace("/**", "/*"))
            for pattern in self.generated_paths
        )

    def is_review_pointless(self, changed_files: Iterable[str]) -> bool:
        """Дифф целиком состоит из файлов, ради которых ревью не запускают.

        Пустой дифф сюда не попадает: это не «нечего ревьюировать», а странное
        состояние, и выходить на нём нулём значило бы его спрятать.
        """
        paths = tuple(changed_files)
        if not paths or not self.skip_paths:
            return False
        return all(
            any(
                fnmatch(path, pattern) or fnmatch(path, pattern.replace("/**", "/*"))
                for pattern in self.skip_paths
            )
            for path in paths
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
        skip_paths=tuple(raw.get("skip_paths") or ()),
        generated_paths=tuple(raw.get("generated_paths") or ()),
        limits=Limits(
            max_files=limits_raw.get("max_files", defaults.max_files),
            max_changed_lines=limits_raw.get("max_changed_lines", defaults.max_changed_lines),
        ),
        pricing=_fill_price_keys(raw.get("pricing") or {}),
    )
    return config.validate()


def apply_env_overrides(config: Config, env: Mapping[str, str]) -> Config:
    """The repository's config, moved to another provider by CI variables.

    AI_REVIEW_PROVIDER is meant to be set on the GitLab group: one variable moves
    every repository at once and back again, with no merge request per
    repository. AI_REVIEW_MODEL and AI_REVIEW_MODEL_HIGH_RISK name the models;
    without them a provider other than the repository's gets DEFAULT_MODELS.

    An empty variable is the same as an absent one, so returning to OpenAI does
    not depend on whether the variable was deleted or emptied. Everything that
    is not about the model - paths, limits, test command - stays as the
    repository wrote it.
    """
    provider = (env.get("AI_REVIEW_PROVIDER") or "").strip()
    if not provider:
        return config
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigError(f"unsupported provider in AI_REVIEW_PROVIDER: {provider}")

    if provider == config.provider:
        default_model, default_high_risk = config.model, config.model_high_risk
    elif provider in DEFAULT_MODELS:
        default_model, default_high_risk = DEFAULT_MODELS[provider]
    else:
        # Switching *to* openai from a repository written for another provider:
        # there is no OpenAI model to guess, so the variables must name one.
        default_model, default_high_risk = "", ""

    model = (env.get("AI_REVIEW_MODEL") or "").strip() or default_model
    model_high_risk = (env.get("AI_REVIEW_MODEL_HIGH_RISK") or "").strip() or default_high_risk
    return replace(
        config,
        provider=provider,
        model=model,
        model_high_risk=model_high_risk,
        # Repository prices win key by key: a repository that priced a model
        # knows better; what it left out comes from the built-in table.
        pricing={**BUILTIN_PRICING, **_fill_price_keys(config.pricing)},
    ).validate()
