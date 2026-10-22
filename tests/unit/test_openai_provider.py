import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from openai import BadRequestError

from reviewer.llm.openai_provider import OpenAIProvider
from reviewer.llm.usage import UsageTracker


def make_usage(prompt_tokens, completion_tokens, cached_tokens=None):
    details = SimpleNamespace(cached_tokens=cached_tokens) if cached_tokens is not None else None
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=details,
    )


def make_response(tool_calls=(), content=None, usage=None):
    message = SimpleNamespace(tool_calls=tool_calls or None, content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


def make_tool_call(call_id, name, arguments):
    function = SimpleNamespace(name=name, arguments=json.dumps(arguments))
    return SimpleNamespace(id=call_id, function=function)


def make_bad_request_error() -> BadRequestError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return BadRequestError(
        "tool_choice='required' is not supported",
        response=httpx.Response(400, request=request),
        body=None,
    )


@pytest.fixture()
def provider():
    return OpenAIProvider(api_key="test-key", base_url="https://api.openai.com/v1", model="test-model")


def test_complete_requests_tool_choice_required(provider):
    response = make_response(tool_calls=[make_tool_call("c1", "list_files", {"path_glob": "*.go"})])
    with patch.object(provider._client.chat.completions, "create", return_value=response) as mock_create:
        result = provider.complete([{"role": "user", "content": "go"}], [{"type": "function"}])

    assert mock_create.call_count == 1
    assert mock_create.call_args.kwargs["tool_choice"] == "required"
    assert result.tool_calls[0].name == "list_files"
    assert result.tool_calls[0].arguments == {"path_glob": "*.go"}


def test_falls_back_to_auto_when_required_is_rejected(provider, capsys):
    fallback_response = make_response(
        tool_calls=[make_tool_call("c1", "submit_findings", {"summary": "s", "findings": []})]
    )
    with patch.object(
        provider._client.chat.completions,
        "create",
        side_effect=[make_bad_request_error(), fallback_response],
    ) as mock_create:
        result = provider.complete([{"role": "user", "content": "go"}], [{"type": "function"}])

    assert mock_create.call_count == 2
    assert mock_create.call_args_list[0].kwargs["tool_choice"] == "required"
    assert mock_create.call_args_list[1].kwargs["tool_choice"] == "auto"
    assert result.tool_calls[0].name == "submit_findings"
    assert "tool_choice='required'" in capsys.readouterr().err


def test_complete_records_usage_into_the_shared_tracker():
    tracker = UsageTracker()
    provider = OpenAIProvider(
        api_key="test-key", base_url="https://api.openai.com/v1", model="test-model",
        usage_tracker=tracker,
    )
    response = make_response(
        tool_calls=[make_tool_call("c1", "list_files", {})],
        usage=make_usage(prompt_tokens=1000, completion_tokens=200, cached_tokens=300),
    )
    with patch.object(provider._client.chat.completions, "create", return_value=response):
        provider.complete([{"role": "user", "content": "go"}], [{"type": "function"}])

    usage = tracker.by_model()["test-model"]
    assert usage.prompt_tokens == 1000
    assert usage.completion_tokens == 200
    assert usage.cached_tokens == 300


def test_complete_records_usage_across_multiple_calls_on_the_same_provider():
    tracker = UsageTracker()
    provider = OpenAIProvider(
        api_key="test-key", base_url="https://api.openai.com/v1", model="test-model",
        usage_tracker=tracker,
    )
    responses = [
        make_response(
            tool_calls=[make_tool_call("c1", "list_files", {})],
            usage=make_usage(prompt_tokens=500, completion_tokens=50),
        ),
        make_response(
            tool_calls=[make_tool_call("c2", "list_files", {})],
            usage=make_usage(prompt_tokens=600, completion_tokens=60),
        ),
    ]
    with patch.object(provider._client.chat.completions, "create", side_effect=responses):
        provider.complete([{"role": "user", "content": "go"}], [{"type": "function"}])
        provider.complete([{"role": "user", "content": "go"}], [{"type": "function"}])

    usage = tracker.by_model()["test-model"]
    assert usage.prompt_tokens == 1100
    assert usage.completion_tokens == 110


def test_complete_without_a_tracker_does_not_error_on_usage(provider):
    response = make_response(
        tool_calls=[make_tool_call("c1", "list_files", {})],
        usage=make_usage(prompt_tokens=1000, completion_tokens=200),
    )
    with patch.object(provider._client.chat.completions, "create", return_value=response):
        result = provider.complete([{"role": "user", "content": "go"}], [{"type": "function"}])
    assert result.tool_calls[0].name == "list_files"


def test_complete_records_usage_under_the_phase_it_was_created_for():
    # The adjudication pass and the review pass share one tracker; without a phase
    # on the provider their tokens are indistinguishable in the report.
    tracker = UsageTracker()
    provider = OpenAIProvider(
        api_key="k",
        base_url="https://example.invalid/v1",
        model="test-model",
        usage_tracker=tracker,
        phase="adjudication",
    )
    response = make_response(
        tool_calls=(),
        content="text",
        usage=make_usage(prompt_tokens=700, completion_tokens=70),
    )

    with patch.object(provider._client.chat.completions, "create", return_value=response):
        provider.complete([{"role": "user", "content": "hi"}], [])

    assert tracker.by_phase()["adjudication"].prompt_tokens == 700
    assert "review" not in tracker.by_phase()


def test_a_required_tool_is_forced_by_name(provider):
    # The last loop step demands the verdict without narrowing the tool list,
    # which keeps the prompt prefix - and OpenAI's prompt cache - unchanged.
    response = make_response(tool_calls=[make_tool_call("c1", "submit_findings", {"summary": "s", "findings": []})])
    with patch.object(provider._client.chat.completions, "create", return_value=response) as mock_create:
        provider.complete([{"role": "user", "content": "go"}], [{"type": "function"}], required_tool="submit_findings")

    assert mock_create.call_args.kwargs["tool_choice"] == {
        "type": "function",
        "function": {"name": "submit_findings"},
    }


def test_a_rejected_forced_tool_falls_back_to_auto(provider):
    fallback_response = make_response(tool_calls=[make_tool_call("c1", "submit_findings", {})])
    with patch.object(
        provider._client.chat.completions, "create",
        side_effect=[make_bad_request_error(), fallback_response],
    ) as mock_create:
        provider.complete([{"role": "user", "content": "go"}], [{"type": "function"}], required_tool="submit_findings")

    assert mock_create.call_args_list[1].kwargs["tool_choice"] == "auto"
