"""When Claude is unavailable the review continues on OpenAI.

The Claude budget is prepaid and ends on a date. Running out of it, an expired
key or a revoked one must not turn every merge request in the group red: the
reviewer is a blocking gate, and a red gate over billing would stop merges
until someone noticed. So the run falls back to the OpenAI configuration the
repository already has, says so in the job log, and stays there for the rest
of the run.
"""

import httpx2
import pytest
from anthropic import APIConnectionError, AuthenticationError

from reviewer.config import Config
from reviewer.llm.anthropic_provider import AnthropicProvider
from reviewer.llm.base import ProviderResponse, ToolCall
from reviewer.llm.factory import provider_factory
from reviewer.llm.fallback import FallbackProvider, FallbackSwitch
from reviewer.llm.openai_provider import OpenAIProvider
from reviewer.llm.usage import UsageTracker


class Unavailable(Exception):
    pass


class FakeProvider:
    def __init__(self, name, error=None):
        self.name = name
        self.error = error
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return ProviderResponse(tool_calls=(ToolCall(id="c", name=self.name, arguments={}),), text=None)


def make(primary, fallback=None, switch=None):
    switch = switch or FallbackSwitch()
    return FallbackProvider(
        primary,
        (lambda: fallback) if fallback is not None else None,
        switch,
        errors=(Unavailable,),
        fallback_label="gpt-base",
    )


def test_a_healthy_primary_is_used_and_the_fallback_is_never_built():
    built = []
    primary = FakeProvider("claude")
    provider = FallbackProvider(
        primary, lambda: built.append(1) or FakeProvider("openai"), FallbackSwitch(),
        errors=(Unavailable,), fallback_label="gpt-base",
    )
    assert provider.complete([], []).tool_calls[0].name == "claude"
    assert built == []


def test_an_unavailable_primary_hands_the_same_call_to_the_fallback(capsys):
    primary = FakeProvider("claude", error=Unavailable("credit balance is too low"))
    fallback = FakeProvider("openai")

    result = make(primary, fallback).complete([{"role": "user", "content": "go"}], [])

    assert result.tool_calls[0].name == "openai"
    err = capsys.readouterr().err
    assert "OpenAI" in err
    assert "gpt-base" in err
    assert "credit balance is too low" in err


def test_after_the_switch_trips_the_primary_is_not_asked_again():
    # One failed call per run, not one per loop step: each would cost a round
    # trip and, after SDK retries, several seconds.
    switch = FallbackSwitch()
    primary = FakeProvider("claude", error=Unavailable("401"))
    fallback = FakeProvider("openai")
    provider = make(primary, fallback, switch)

    provider.complete([], [])
    provider.complete([], [])

    assert primary.calls == 1
    assert fallback.calls == 2


def test_the_switch_is_shared_by_every_provider_of_the_run(capsys):
    # Review halves and adjudication each get their own provider. Once Claude is
    # known to be unavailable, none of them should try it again, and the log
    # should say so once.
    switch = FallbackSwitch()
    first_primary = FakeProvider("claude", error=Unavailable("401"))
    second_primary = FakeProvider("claude")
    make(first_primary, FakeProvider("openai"), switch).complete([], [])
    result = make(second_primary, FakeProvider("openai"), switch).complete([], [])

    assert second_primary.calls == 0
    assert result.tool_calls[0].name == "openai"
    assert capsys.readouterr().err.count("OpenAI") == 1


def test_without_a_fallback_the_error_is_raised():
    primary = FakeProvider("claude", error=Unavailable("401"))
    with pytest.raises(Unavailable):
        make(primary, fallback=None).complete([], [])


def test_errors_outside_the_list_are_not_masked():
    # A bug of ours (a TypeError in message conversion) must surface, not be
    # quietly reviewed by another provider.
    primary = FakeProvider("claude", error=TypeError("bug"))
    with pytest.raises(TypeError):
        make(primary, FakeProvider("openai")).complete([], [])


# --- the factory: which provider a run gets ---------------------------------------

REPO = Config(provider="openai", model="gpt-base", model_high_risk="gpt-strong")
SWITCHED = Config(provider="anthropic", model="claude-sonnet-5-5", model_high_risk="claude-opus-5-5")


def test_an_openai_run_gets_the_openai_provider_as_before():
    make_provider = provider_factory(REPO, REPO, {"OPENAI_API_KEY": "sk"}, UsageTracker())
    provider = make_provider("gpt-base")
    assert isinstance(provider, OpenAIProvider)


def test_an_openai_run_still_requires_its_key():
    with pytest.raises(KeyError):
        provider_factory(REPO, REPO, {}, UsageTracker())


def test_a_switched_run_requires_the_anthropic_key():
    with pytest.raises(KeyError):
        provider_factory(SWITCHED, REPO, {"OPENAI_API_KEY": "sk"}, UsageTracker())


def test_a_switched_run_gets_claude_with_an_openai_fallback():
    make_provider = provider_factory(
        SWITCHED, REPO, {"ANTHROPIC_API_KEY": "ak", "OPENAI_API_KEY": "sk"}, UsageTracker()
    )
    provider = make_provider("claude-sonnet-5-5")
    assert isinstance(provider, FallbackProvider)
    assert isinstance(provider.primary, AnthropicProvider)
    assert isinstance(provider.build_fallback(), OpenAIProvider)


@pytest.mark.parametrize(
    "claude_model, openai_model",
    [("claude-sonnet-5-5", "gpt-base"), ("claude-opus-5-5", "gpt-strong")],
)
def test_the_fallback_keeps_the_risk_level(claude_model, openai_model):
    # A high-risk half reviewed by Opus falls back to the repository's strong
    # model, not to the cheap one - the risk did not change with the provider.
    make_provider = provider_factory(
        SWITCHED, REPO, {"ANTHROPIC_API_KEY": "ak", "OPENAI_API_KEY": "sk"}, UsageTracker()
    )
    assert make_provider(claude_model).build_fallback()._model == openai_model


def test_a_switched_run_without_an_openai_key_has_no_fallback():
    make_provider = provider_factory(SWITCHED, REPO, {"ANTHROPIC_API_KEY": "ak"}, UsageTracker())
    assert make_provider("claude-sonnet-5-5").build_fallback() is None


def test_real_anthropic_unavailability_errors_trip_the_fallback(capsys):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    errors = [
        AuthenticationError("invalid x-api-key", response=httpx2.Response(401, request=request), body=None),
        APIConnectionError(request=request),
    ]
    for error in errors:
        make_provider = provider_factory(
            SWITCHED, REPO, {"ANTHROPIC_API_KEY": "ak", "OPENAI_API_KEY": "sk"}, UsageTracker()
        )
        provider = make_provider("claude-sonnet-5-5")
        provider.primary = FakeProvider("claude", error=error)
        provider._make_fallback = lambda: FakeProvider("openai")
        assert provider.complete([], []).tool_calls[0].name == "openai"


def test_an_openai_run_does_not_need_the_anthropic_package(monkeypatch):
    # In this repository the ai-review job runs the branch's reviewer/ on top of
    # the released image's dependencies. A branch that adds a dependency is
    # therefore reviewed by an image without it - !36 died on its own gate with
    # "No module named 'anthropic'", and exit 1, which reads as findings. Every
    # repository still on OpenAI would hit the same if the image lagged.
    import importlib
    import sys

    for name in list(sys.modules):
        if name == "anthropic" or name.startswith("anthropic.") or name in (
            "reviewer.llm.factory",
            "reviewer.llm.anthropic_provider",
        ):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "anthropic", None)  # any import of it fails

    factory = importlib.import_module("reviewer.llm.factory")
    make_provider = factory.provider_factory(REPO, REPO, {"OPENAI_API_KEY": "sk"}, UsageTracker())

    assert type(make_provider("gpt-base")).__name__ == "OpenAIProvider"
