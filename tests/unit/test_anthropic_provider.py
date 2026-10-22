"""The Claude provider speaks the reviewer's OpenAI-shaped conversation.

The review loop and adjudication build messages in OpenAI's format and must not
learn a second one. Everything Anthropic-specific - the separate system prompt,
tool_use/tool_result blocks, thinking blocks that have to be sent back
unchanged, cache accounting - is the provider's business and is held here.
"""

from unittest.mock import MagicMock

import httpx2
import pytest
from anthropic import BadRequestError
from anthropic.types import Message

from reviewer.llm.anthropic_provider import (
    AnthropicProvider,
    to_anthropic_messages,
    to_anthropic_tools,
)
from reviewer.llm.usage import UsageTracker

READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}


def make_message(content, usage=None, stop_reason="tool_use"):
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage or {"input_tokens": 1, "output_tokens": 1},
        }
    )


def make_provider(model="claude-sonnet-5-5", tracker=None, phase="review"):
    client = MagicMock()
    provider = AnthropicProvider(
        api_key="test-key", model=model, usage_tracker=tracker, phase=phase, client=client
    )
    return provider, client.messages.create


def bad_request(text):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return BadRequestError(text, response=httpx2.Response(400, request=request), body=None)


# --- tools --------------------------------------------------------------------


def test_tools_are_converted_to_input_schema():
    assert to_anthropic_tools([READ_FILE]) == [
        {
            "name": "read_file",
            "description": "Read a file.",
            "input_schema": READ_FILE["function"]["parameters"],
        }
    ]


# --- messages -------------------------------------------------------------------


def test_system_message_goes_to_the_system_parameter():
    system, messages = to_anthropic_messages(
        [{"role": "system", "content": "rules"}, {"role": "user", "content": "diff"}], {}
    )
    assert system == "rules"
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "diff"}]}]


def test_tool_calls_and_results_become_tool_use_and_tool_result_blocks():
    openai_shaped = [
        {"role": "user", "content": "diff"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.go"}'}},
                {"id": "c2", "type": "function", "function": {"name": "grep", "arguments": '{"pattern": "x"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "<tool_result>\nA\n</tool_result>"},
        {"role": "tool", "tool_call_id": "c2", "content": "Tool error: bad pattern"},
    ]
    _, messages = to_anthropic_messages(openai_shaped, {})

    assert messages[1] == {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "c1", "name": "read_file", "input": {"path": "a.go"}},
            {"type": "tool_use", "id": "c2", "name": "grep", "input": {"pattern": "x"}},
        ],
    }
    # Parallel results travel in ONE user message: two consecutive user turns
    # are a 400 on Anthropic's side.
    assert len(messages) == 3
    assert messages[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "c1", "content": "<tool_result>\nA\n</tool_result>"},
            {"type": "tool_result", "tool_use_id": "c2", "content": "Tool error: bad pattern", "is_error": True},
        ],
    }


def test_a_reminder_after_tool_results_joins_them_and_comes_last():
    # The loop appends BUDGET_EXHAUSTED_REMINDER as a user message right after the
    # tool results. Anthropic requires tool_result blocks first in that message.
    openai_shaped = [
        {"role": "user", "content": "diff"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "A"},
        {"role": "user", "content": "budget exhausted"},
    ]
    _, messages = to_anthropic_messages(openai_shaped, {})

    assert messages[-1]["content"] == [
        {"type": "tool_result", "tool_use_id": "c1", "content": "A"},
        {"type": "text", "text": "budget exhausted"},
    ]


def test_an_empty_assistant_text_is_dropped_and_the_user_turns_merge():
    # A plain-text answer of "" is what the loop records when the model replied
    # with nothing usable. An empty text block is rejected by the API.
    _, messages = to_anthropic_messages(
        [
            {"role": "user", "content": "diff"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "call a tool"},
        ],
        {},
    )
    assert messages == [
        {"role": "user", "content": [{"type": "text", "text": "diff"}, {"type": "text", "text": "call a tool"}]}
    ]


def test_the_original_assistant_blocks_are_replayed_for_known_tool_calls():
    # Thinking is always on for Sonnet 5.5 and Opus 5.5, and inside a tool-use
    # turn its blocks must go back unchanged. The OpenAI-shaped history loses
    # them, so the provider keeps the response's blocks by tool call id.
    raw = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "tool_use", "id": "c1", "name": "read_file", "input": {"path": "a.go"}},
    ]
    _, messages = to_anthropic_messages(
        [
            {"role": "user", "content": "diff"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.go"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "A"},
        ],
        {"c1": raw},
    )
    assert messages[1] == {"role": "assistant", "content": raw}


# --- the request ----------------------------------------------------------------


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "claude-opus-5-5"])
def test_models_that_reject_forced_tool_use_get_auto(model):
    # platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use,
    # read 2026-10-22: on Opus 5.5 and Sonnet 5.5 "any and tool return a 400 error".
    provider, create = make_provider(model=model)
    create.return_value = make_message([{"type": "tool_use", "id": "t", "name": "read_file", "input": {}}])

    provider.complete([{"role": "user", "content": "go"}], [READ_FILE])

    assert create.call_args.kwargs["tool_choice"] == {"type": "auto"}


def test_models_that_accept_forced_tool_use_get_any():
    provider, create = make_provider(model="claude-haiku-5-5")
    create.return_value = make_message([{"type": "tool_use", "id": "t", "name": "read_file", "input": {}}])

    provider.complete([{"role": "user", "content": "go"}], [READ_FILE])

    assert create.call_args.kwargs["tool_choice"] == {"type": "any"}


def test_a_rejected_forced_tool_choice_is_retried_with_auto(capsys):
    provider, create = make_provider(model="claude-haiku-5-5")
    create.side_effect = [
        bad_request("tool_choice any is not supported for this model"),
        make_message([{"type": "tool_use", "id": "t", "name": "read_file", "input": {}}]),
    ]

    result = provider.complete([{"role": "user", "content": "go"}], [READ_FILE])

    assert [c.kwargs["tool_choice"] for c in create.call_args_list] == [{"type": "any"}, {"type": "auto"}]
    assert result.tool_calls[0].name == "read_file"
    assert "tool_choice" in capsys.readouterr().err


def test_other_bad_requests_are_not_swallowed():
    provider, create = make_provider(model="claude-haiku-5-5")
    create.side_effect = bad_request("messages: text content blocks must be non-empty")

    with pytest.raises(BadRequestError):
        provider.complete([{"role": "user", "content": "go"}], [READ_FILE])
    assert create.call_count == 1


def test_the_request_asks_for_automatic_prompt_caching():
    # Every loop step resends the whole conversation. Without caching the run
    # costs about four times more, and the $200 budget would not last the period.
    provider, create = make_provider()
    create.return_value = make_message([{"type": "tool_use", "id": "t", "name": "read_file", "input": {}}])

    provider.complete([{"role": "system", "content": "rules"}, {"role": "user", "content": "go"}], [READ_FILE])

    kwargs = create.call_args.kwargs
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    assert kwargs["system"] == "rules"
    assert kwargs["model"] == "claude-sonnet-5-5"
    assert kwargs["max_tokens"] > 0


def test_no_tools_means_no_tool_choice():
    provider, create = make_provider()
    create.return_value = make_message([{"type": "text", "text": "ok"}], stop_reason="end_turn")

    provider.complete([{"role": "user", "content": "go"}], [])

    assert "tools" not in create.call_args.kwargs
    assert "tool_choice" not in create.call_args.kwargs


def test_the_client_gets_an_explicit_timeout():
    # Without one the SDK computes a timeout from max_tokens and refuses
    # non-streaming requests it expects to run past ten minutes.
    provider = AnthropicProvider(api_key="test-key", model="claude-sonnet-5-5")
    assert provider._client.timeout == 300.0
    assert provider._client.max_retries == 3


# --- the response ---------------------------------------------------------------


def test_tool_use_blocks_become_tool_calls_and_text_is_kept():
    provider, create = make_provider()
    create.return_value = make_message(
        [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": "Reading."},
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.go"}},
        ]
    )

    result = provider.complete([{"role": "user", "content": "go"}], [READ_FILE])

    assert [(c.id, c.name, c.arguments) for c in result.tool_calls] == [("t1", "read_file", {"path": "a.go"})]
    assert result.text == "Reading."


def test_a_response_without_text_has_text_none():
    provider, create = make_provider()
    create.return_value = make_message([{"type": "tool_use", "id": "t1", "name": "read_file", "input": {}}])
    assert provider.complete([{"role": "user", "content": "go"}], [READ_FILE]).text is None


def test_the_next_request_replays_the_thinking_of_the_previous_turn():
    provider, create = make_provider()
    first = [
        {"type": "thinking", "thinking": "", "signature": "sig-1"},
        {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.go"}},
    ]
    create.side_effect = [
        make_message(first),
        make_message([{"type": "tool_use", "id": "t2", "name": "read_file", "input": {}}]),
    ]

    provider.complete([{"role": "user", "content": "go"}], [READ_FILE])
    # What the loop appends after the first answer, in its own format.
    history = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.go"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "A"},
    ]
    provider.complete(history, [READ_FILE])

    sent = create.call_args_list[1].kwargs["messages"]
    assert sent[1] == {"role": "assistant", "content": first}


def test_usage_counts_cache_reads_and_writes_as_part_of_the_prompt():
    # Anthropic's input_tokens excludes both; the tracker's prompt_tokens is the
    # whole prompt, as with OpenAI, so the two providers' lines compare.
    tracker = UsageTracker()
    provider, create = make_provider(tracker=tracker, phase="adjudication")
    create.return_value = make_message(
        [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {}}],
        usage={
            "input_tokens": 50,
            "output_tokens": 70,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 100,
        },
    )

    provider.complete([{"role": "user", "content": "go"}], [READ_FILE])

    usage = tracker.by_model()["claude-sonnet-5-5"]
    assert usage.prompt_tokens == 1050
    assert usage.cached_tokens == 900
    assert usage.cache_write_tokens == 100
    assert usage.completion_tokens == 70
    assert tracker.by_phase()["adjudication"].prompt_tokens == 1050
