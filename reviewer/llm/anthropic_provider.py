"""Anthropic (Claude) provider.

The review loop and adjudication speak OpenAI's message format: a system
message, `tool_calls` on assistant messages, `role: tool` results. This
provider translates that conversation for the Messages API at the edge and
translates the answer back, so nothing upstream learns a second format and the
OpenAI fallback can pick up the same conversation mid-run.

Three things differ from OpenAI and are handled only here:

* Forced tool use. Opus 5.5 and Sonnet 5.5 answer `tool_choice: any` with a
  400 (platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use,
  read 2026-10-22). They get `auto`; the loop already reminds a model that
  answered with text instead of a tool.
* Thinking. It is always on for those models and cannot be disabled. Inside a
  tool-use turn the thinking blocks must go back exactly as they came, and the
  OpenAI-shaped history has no place for them. So the provider keeps every
  response's blocks by tool call id and replays them when that assistant
  message is sent again.
* Cache accounting. `input_tokens` excludes cache reads and writes; the tracker
  is given the whole prompt, as OpenAI reports it, with reads and writes as
  parts of it.
"""

from __future__ import annotations

import json
import sys

from anthropic import Anthropic, BadRequestError

from reviewer.llm.base import ProviderResponse, ToolCall
from reviewer.llm.usage import REVIEW_PHASE, UsageTracker

# Models that reject forced tool use. Matched by prefix, so a dated snapshot of
# the same model is covered too.
FORCED_TOOL_CHOICE_UNSUPPORTED = (
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-fable-5-1",
    "claude-mythos-5-1",
)

# Room for adaptive thinking plus the verdict. Measured on OpenAI 15-22.10: the
# average run wrote 1.1K output tokens; thinking is billed as output on top.
DEFAULT_MAX_TOKENS = 16_000

# Explicit, and longer than OpenAI's 120 s: a prompt of ~300K tokens with
# thinking takes longer, and without an explicit timeout the SDK derives one
# from max_tokens and refuses non-streaming requests it expects to be long.
DEFAULT_TIMEOUT = 300.0


def _supports_forced_tool_choice(model: str) -> bool:
    return not model.startswith(FORCED_TOOL_CHOICE_UNSUPPORTED)


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """OpenAI function tools -> Anthropic tool definitions."""
    converted = []
    for tool in tools:
        function = tool["function"]
        converted.append(
            {
                "name": function["name"],
                "description": function.get("description", ""),
                "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return converted


def _arguments(raw: str | None) -> dict:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _blocks_for(message: dict, raw_by_call_id: dict[str, list[dict]]) -> list[dict]:
    role = message["role"]
    if role == "tool":
        content = message.get("content") or "(empty result)"
        block = {"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": content}
        # The loop's own error text starts this way; repository content is
        # wrapped in <tool_result> and never does.
        if content.startswith("Tool error:"):
            block["is_error"] = True
        return [block]

    if role == "assistant" and message.get("tool_calls"):
        calls = message["tool_calls"]
        raw = raw_by_call_id.get(calls[0]["id"])
        if raw is not None:
            return list(raw)
        blocks = []
        if message.get("content"):
            blocks.append({"type": "text", "text": message["content"]})
        for call in calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["function"]["name"],
                    "input": _arguments(call["function"].get("arguments")),
                }
            )
        return blocks

    text = message.get("content") or ""
    # An empty text block is a 400. An empty assistant answer carries nothing,
    # so it is dropped and the turns around it merge.
    return [{"type": "text", "text": text}] if text else []


def to_anthropic_messages(
    messages: list[dict], raw_by_call_id: dict[str, list[dict]]
) -> tuple[str | None, list[dict]]:
    """OpenAI-shaped conversation -> (system prompt, Anthropic messages).

    Consecutive messages of one Anthropic role are merged: parallel tool results
    become one user message, and a reminder the loop appends after them joins
    that message. In a user message tool_result blocks come first.
    """
    system_parts: list[str] = []
    converted: list[dict] = []
    for message in messages:
        if message["role"] == "system":
            if message.get("content"):
                system_parts.append(message["content"])
            continue
        role = "assistant" if message["role"] == "assistant" else "user"
        blocks = _blocks_for(message, raw_by_call_id)
        if not blocks:
            continue
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"].extend(blocks)
        else:
            converted.append({"role": role, "content": blocks})

    for message in converted:
        if message["role"] == "user":
            results = [b for b in message["content"] if b.get("type") == "tool_result"]
            rest = [b for b in message["content"] if b.get("type") != "tool_result"]
            message["content"] = results + rest

    system = "\n\n".join(system_parts) if system_parts else None
    return system, converted


class AnthropicProvider:
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float = DEFAULT_TIMEOUT,
        usage_tracker: UsageTracker | None = None,
        phase: str = REVIEW_PHASE,
        client=None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._client = client or Anthropic(api_key=api_key, timeout=timeout, max_retries=3)
        self._model = model
        self._usage_tracker = usage_tracker
        self._phase = phase
        self._max_tokens = max_tokens
        # Response blocks by tool call id, replayed when the loop sends that
        # assistant message again. Lives as long as the provider, which is one
        # review pass or one adjudication.
        self._raw_by_call_id: dict[str, list[dict]] = {}

    def complete(self, messages: list[dict], tools: list[dict]) -> ProviderResponse:
        system, converted = to_anthropic_messages(messages, self._raw_by_call_id)
        request = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": converted,
            # Automatic caching: the breakpoint follows the end of the growing
            # conversation, so each loop step reads the previous one from cache.
            "cache_control": {"type": "ephemeral"},
        }
        if system:
            request["system"] = system
        forced = False
        if tools:
            request["tools"] = to_anthropic_tools(tools)
            forced = _supports_forced_tool_choice(self._model)
            request["tool_choice"] = {"type": "any"} if forced else {"type": "auto"}

        try:
            response = self._client.messages.create(**request)
        except BadRequestError as exc:
            if not forced or "tool_choice" not in str(exc):
                raise
            print(
                f"AI review: провайдер отклонил tool_choice=any ({exc}), повтор с auto",
                file=sys.stderr,
            )
            request["tool_choice"] = {"type": "auto"}
            response = self._client.messages.create(**request)

        self._record_usage(response)

        calls = []
        texts = []
        for block in response.content:
            if block.type == "tool_use":
                arguments = block.input if isinstance(block.input, dict) else {}
                calls.append(ToolCall(id=block.id, name=block.name, arguments=arguments))
            elif block.type == "text" and block.text:
                texts.append(block.text)

        if calls:
            raw = [block.model_dump(mode="json", exclude_none=True) for block in response.content]
            for call in calls:
                self._raw_by_call_id[call.id] = raw

        return ProviderResponse(tool_calls=tuple(calls), text="\n".join(texts) if texts else None)

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if self._usage_tracker is None or usage is None:
            return
        cache_read = usage.cache_read_input_tokens or 0
        cache_write = usage.cache_creation_input_tokens or 0
        self._usage_tracker.record(
            self._model,
            usage.input_tokens + cache_read + cache_write,
            usage.output_tokens,
            cache_read,
            phase=self._phase,
            cache_write_tokens=cache_write,
        )
