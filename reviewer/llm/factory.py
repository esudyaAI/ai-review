"""Which provider a run gets, decided once from config and environment."""

from __future__ import annotations

from typing import Callable, Mapping

from reviewer.config import Config
from reviewer.llm.base import LLMProvider
from reviewer.llm.fallback import FallbackProvider, FallbackSwitch
from reviewer.llm.openai_provider import OpenAIProvider
from reviewer.llm.usage import REVIEW_PHASE, UsageTracker

# The anthropic package is imported only when a run is switched to Claude.
#
# In jobbytime-ai-review itself the ai-review job runs the branch's reviewer/
# on top of the released image's dependencies, so a branch that adds a
# dependency is reviewed by an image without it. !36 died on its own gate with
# "No module named 'anthropic'" - and with exit 1, which reads as findings. An
# OpenAI run has no use for the package and must not need it.
#
# What trips the fallback is anthropic.APIError: everything the SDK raises about
# the service rather than about our code - status errors (401 key, 402 billing,
# 403, 404 model, 429 spend cap, 5xx/529 after the SDK's own retries) and
# connection failures. A 400 is in there too - spend limits are reported as
# 400 - so a request bug of ours would also fall back; the log line names the
# error, and that is what the trial run after each release is for.


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

    import anthropic

    from reviewer.llm.anthropic_provider import AnthropicProvider

    openai_key = env.get("OPENAI_API_KEY") or ""
    # AI_REVIEW_THINKING=between_tools: the output-spend lever for Sonnet,
    # switched on the group like the provider itself, no merge request needed.
    thinking = (env.get("AI_REVIEW_THINKING") or "").strip() or None
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
            thinking=thinking,
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
            errors=(anthropic.APIError,),
            fallback_label=fallback_model,
        )

    return make_claude
