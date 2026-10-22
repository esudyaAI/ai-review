"""Which provider a run gets, decided once from config and environment."""

from __future__ import annotations

from typing import Callable, Mapping

import anthropic

from reviewer.config import Config
from reviewer.llm.anthropic_provider import AnthropicProvider
from reviewer.llm.base import LLMProvider
from reviewer.llm.fallback import FallbackProvider, FallbackSwitch
from reviewer.llm.openai_provider import OpenAIProvider
from reviewer.llm.usage import REVIEW_PHASE, UsageTracker

# Everything the Anthropic SDK raises about the service rather than about our
# code: status errors (401 key, 402 billing, 403, 404 model, 429 spend cap,
# 5xx/529 after the SDK's own retries) and connection failures. A 400 is in
# here too - spend limits are reported as 400 - so a request bug of ours would
# also fall back; the log line names the error, and that is what the trial run
# after each release is for.
ANTHROPIC_UNAVAILABLE = (anthropic.APIError,)


def provider_factory(
    config: Config,
    repo_config: Config,
    env: Mapping[str, str],
    usage_tracker: UsageTracker,
    switch: FallbackSwitch | None = None,
) -> Callable[..., LLMProvider]:
    """make_provider(model, phase) for this run.

    `config` is what the run uses (after the group switch); `repo_config` is the
    repository's .ai-review.yml as written, which is where the fallback's OpenAI
    models come from. Keys are read here, at startup, so a missing one is a
    technical error before any work - as OPENAI_API_KEY always was.
    """
    if config.provider == "openai":
        openai_key = env["OPENAI_API_KEY"]

        def make_openai(model: str, phase: str = REVIEW_PHASE) -> LLMProvider:
            return OpenAIProvider(
                api_key=openai_key,
                base_url=config.base_url,
                model=model,
                usage_tracker=usage_tracker,
                phase=phase,
            )

        return make_openai

    anthropic_key = env["ANTHROPIC_API_KEY"]
    openai_key = env.get("OPENAI_API_KEY") or ""
    can_fall_back = bool(openai_key) and repo_config.provider == "openai" and bool(repo_config.model)
    switch = switch or FallbackSwitch()

    def fallback_model_for(model: str) -> str:
        # The risk of a file does not change with the provider: the half the
        # strong Claude model was reviewing falls back to the strong OpenAI one.
        strong = config.model_high_risk and config.model_high_risk != config.model
        if strong and model == config.model_high_risk and repo_config.model_high_risk:
            return repo_config.model_high_risk
        return repo_config.model

    def make_claude(model: str, phase: str = REVIEW_PHASE) -> LLMProvider:
        primary = AnthropicProvider(
            api_key=anthropic_key,
            model=model,
            usage_tracker=usage_tracker,
            phase=phase,
        )
        fallback_model = fallback_model_for(model)
        make_fallback = None
        if can_fall_back:

            def make_fallback() -> LLMProvider:
                return OpenAIProvider(
                    api_key=openai_key,
                    base_url=repo_config.base_url,
                    model=fallback_model,
                    usage_tracker=usage_tracker,
                    phase=phase,
                )

        return FallbackProvider(
            primary,
            make_fallback,
            switch,
            errors=ANTHROPIC_UNAVAILABLE,
            fallback_label=fallback_model,
        )

    return make_claude
