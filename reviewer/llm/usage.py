"""Token usage and cost accounting, shared across every OpenAIProvider instance
within a single review run (the CLI creates one tracker in main() and passes it
into every provider the run creates, so usage accumulates across the main review
pass and any adjudication calls)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


class UsageTracker:
    def __init__(self) -> None:
        self._by_model: dict[str, ModelUsage] = {}

    def record(
        self, model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0
    ) -> None:
        usage = self._by_model.setdefault(model, ModelUsage())
        usage.prompt_tokens += prompt_tokens
        usage.completion_tokens += completion_tokens
        usage.cached_tokens += cached_tokens

    def by_model(self) -> dict[str, ModelUsage]:
        return dict(self._by_model)


def cost_for(model: str, usage: ModelUsage, pricing: dict[str, dict[str, float]]) -> float | None:
    """Cost in USD for this model's accumulated usage, or None if the model has no
    price entry in `pricing` (or that entry is missing input/output prices)."""
    prices = pricing.get(model)
    if not prices or "input" not in prices or "output" not in prices:
        return None
    # Cached tokens are a subset of prompt_tokens, not additional - OpenAI's usage
    # reporting counts them within prompt_tokens.
    cached_tokens = min(usage.cached_tokens, usage.prompt_tokens)
    non_cached_input = usage.prompt_tokens - cached_tokens
    cached_price = prices.get("cached_input", prices["input"])
    cost = (
        non_cached_input * prices["input"]
        + cached_tokens * cached_price
        + usage.completion_tokens * prices["output"]
    )
    return cost / 1_000_000


def report_lines(
    by_model: dict[str, ModelUsage], pricing: dict[str, dict[str, float]]
) -> tuple[list[str], list[str]]:
    """Formats one report line per model for stdout, plus stderr warnings for any
    model with no usable price entry. Never raises - a missing price is reported,
    not fatal."""
    stdout_lines: list[str] = []
    stderr_warnings: list[str] = []
    for model, usage in by_model.items():
        cost = cost_for(model, usage, pricing)
        if cost is None:
            stdout_lines.append(
                f"AI review: {model} — вход {usage.prompt_tokens} токенов "
                f"(из них кеш {usage.cached_tokens}), выход {usage.completion_tokens} токенов, "
                "стоимость неизвестна (нет цены в pricing)"
            )
            stderr_warnings.append(
                f"AI review: для модели '{model}' нет цены в pricing .ai-review.yml, "
                "стоимость прогона не посчитана"
            )
        else:
            stdout_lines.append(
                f"AI review: {model} — вход {usage.prompt_tokens} токенов "
                f"(из них кеш {usage.cached_tokens}), выход {usage.completion_tokens} токенов, "
                f"стоимость ${cost:.4f}"
            )
    return stdout_lines, stderr_warnings
