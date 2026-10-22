"""Token usage and cost accounting, shared across every provider instance
within a single review run (the CLI creates one tracker in main() and passes it
into every provider the run creates, so usage accumulates across the main review
pass and any adjudication calls).

Usage is recorded twice over: by model, which is what cost is computed from, and
by phase, which is what answers "where did the tokens go". The second view exists
because the first one cannot: a single number per run cannot distinguish a diff
that grew from a tool loop that read half the repository.
"""

from __future__ import annotations

from dataclasses import dataclass

REVIEW_PHASE = "review"
ADJUDICATION_PHASE = "adjudication"

# Phase names as they appear in the report line.
PHASE_LABELS = {REVIEW_PHASE: "ревью", ADJUDICATION_PHASE: "адъюдикация"}


@dataclass
class ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    # Tokens written to the prompt cache. Anthropic bills them at 1.25x input and
    # reports them apart from reads; OpenAI has no such charge and leaves this 0.
    # Like cached_tokens, they are part of prompt_tokens, not on top of it.
    cache_write_tokens: int = 0


class UsageTracker:
    def __init__(self) -> None:
        self._by_model: dict[str, ModelUsage] = {}
        self._by_phase: dict[str, ModelUsage] = {}

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        phase: str = REVIEW_PHASE,
        cache_write_tokens: int = 0,
    ) -> None:
        for bucket, key in ((self._by_model, model), (self._by_phase, phase)):
            usage = bucket.setdefault(key, ModelUsage())
            usage.prompt_tokens += prompt_tokens
            usage.completion_tokens += completion_tokens
            usage.cached_tokens += cached_tokens
            usage.cache_write_tokens += cache_write_tokens

    def by_model(self) -> dict[str, ModelUsage]:
        return dict(self._by_model)

    def by_phase(self) -> dict[str, ModelUsage]:
        return dict(self._by_phase)


def cost_for(model: str, usage: ModelUsage, pricing: dict[str, dict[str, float]]) -> float | None:
    """Cost in USD for this model's accumulated usage, or None if the model has no
    price entry in `pricing` (or that entry is missing input/output prices)."""
    prices = pricing.get(model)
    if not prices or "input" not in prices or "output" not in prices:
        return None
    # Cached tokens are a subset of prompt_tokens, not additional - OpenAI's usage
    # reporting counts them within prompt_tokens. Cache writes are a subset too:
    # the Anthropic provider folds them into prompt_tokens the same way.
    cached_tokens = min(usage.cached_tokens, usage.prompt_tokens)
    cache_write_tokens = min(usage.cache_write_tokens, usage.prompt_tokens - cached_tokens)
    non_cached_input = usage.prompt_tokens - cached_tokens - cache_write_tokens
    cached_price = prices.get("cached_input", prices["input"])
    cache_write_price = prices.get("cache_write", prices["input"])
    cost = (
        non_cached_input * prices["input"]
        + cached_tokens * cached_price
        + cache_write_tokens * cache_write_price
        + usage.completion_tokens * prices["output"]
    )
    return cost / 1_000_000


def merge_loop_stats(all_stats):
    """One set of counters for a review that ran in more than one pass.

    Since the diff is split by risk, a run has a loop per half. Reporting only the
    first half's counters would understate the shape of the run exactly when it
    matters - a big diff is precisely the one that gets split.
    """
    stats = list(all_stats)
    if not stats:
        return None
    merged = type(stats[0])()
    for one in stats:
        merged.iterations += one.iterations
        merged.tool_calls += one.tool_calls
        merged.tool_result_bytes += one.tool_result_bytes
    return merged


def breakdown_line(by_phase: dict[str, ModelUsage]) -> str | None:
    """One line naming where the input tokens went, or None when there is nothing
    to split: with a single phase the number is already on the cost line, and
    repeating it would only add noise."""
    phases = {name: usage for name, usage in by_phase.items() if usage.prompt_tokens}
    if len(phases) < 2:
        return None

    total = sum(usage.prompt_tokens for usage in phases.values())
    parts = []
    for name, usage in sorted(phases.items(), key=lambda item: -item[1].prompt_tokens):
        share = round(100 * usage.prompt_tokens / total)
        parts.append(f"{PHASE_LABELS.get(name, name)} {usage.prompt_tokens} ({share}%)")

    return "AI review: разбивка входа — " + ", ".join(parts)


def loop_line(loop_stats) -> str:
    """The shape of the review pass: how many round trips it took and how much
    repository content it pulled in.

    Deliberately independent of the phase breakdown. A run with no disputed
    threads has one phase - and it is the ordinary case, the one whose numbers
    are most worth seeing."""
    return (
        f"AI review: цикл ревью — итераций ревью {loop_stats.iterations}, "
        f"вызовов инструментов {loop_stats.tool_calls}, "
        f"прочитано {loop_stats.tool_result_bytes} байт"
    )


def report_lines(
    by_model: dict[str, ModelUsage],
    pricing: dict[str, dict[str, float]],
    by_phase: dict[str, ModelUsage] | None = None,
    loop_stats=None,
) -> tuple[list[str], list[str]]:
    """Formats one report line per model for stdout, plus stderr warnings for any
    model with no usable price entry. Never raises - a missing price is reported,
    not fatal."""
    stdout_lines: list[str] = []
    stderr_warnings: list[str] = []
    for model, usage in by_model.items():
        cost = cost_for(model, usage, pricing)
        # Named only when present, so an OpenAI line keeps its shape.
        writes = f", запись в кеш {usage.cache_write_tokens}" if usage.cache_write_tokens else ""
        if cost is None:
            stdout_lines.append(
                f"AI review: {model} — вход {usage.prompt_tokens} токенов "
                f"(из них кеш {usage.cached_tokens}{writes}), выход {usage.completion_tokens} токенов, "
                "стоимость неизвестна (нет цены в pricing)"
            )
            stderr_warnings.append(
                f"AI review: для модели '{model}' нет цены в pricing .ai-review.yml, "
                "стоимость прогона не посчитана"
            )
        else:
            stdout_lines.append(
                f"AI review: {model} — вход {usage.prompt_tokens} токенов "
                f"(из них кеш {usage.cached_tokens}{writes}), выход {usage.completion_tokens} токенов, "
                f"стоимость ${cost:.4f}"
            )

    if by_phase:
        line = breakdown_line(by_phase)
        if line is not None:
            stdout_lines.append(line)
    if loop_stats is not None:
        stdout_lines.append(loop_line(loop_stats))
    return stdout_lines, stderr_warnings
