"""Drives the model until it submits a structured verdict."""

from __future__ import annotations

import json

from reviewer.findings import (
    SUBMIT_FINDINGS_TOOL,
    Finding,
    FindingsSchemaError,
    parse_findings,
)
from reviewer.llm.base import LLMProvider, ToolCall
from reviewer.tools import TOOL_SCHEMAS, RepoTools, ToolError

MAX_SCHEMA_RETRIES = 1


class ReviewIncomplete(Exception):
    """Raised when the model never produced a usable verdict."""


class ReviewLoop:
    def __init__(
        self,
        provider: LLMProvider,
        tools: RepoTools,
        max_iterations: int = 25,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._max_iterations = max_iterations

    def run(self, system_prompt: str, user_prompt: str) -> list[Finding]:
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        available = [*TOOL_SCHEMAS, SUBMIT_FINDINGS_TOOL]
        schema_retries = 0

        for _ in range(self._max_iterations):
            response = self._provider.complete(messages, available)
            if not response.tool_calls:
                raise ReviewIncomplete("model answered with plain text instead of submit_findings")

            submit = next((c for c in response.tool_calls if c.name == "submit_findings"), None)
            if submit is not None:
                try:
                    return parse_findings(submit.arguments)
                except FindingsSchemaError as exc:
                    if schema_retries >= MAX_SCHEMA_RETRIES:
                        raise ReviewIncomplete(f"submit_findings stayed invalid: {exc}") from exc
                    schema_retries += 1
                    messages.append(_assistant_message(response.tool_calls))
                    messages.append(_tool_message(submit.id, f"Invalid payload: {exc}. Retry."))
                    continue

            messages.append(_assistant_message(response.tool_calls))
            for call in response.tool_calls:
                messages.append(_tool_message(call.id, self._serve(call)))

        raise ReviewIncomplete("model never called submit_findings")

    def _serve(self, call: ToolCall) -> str:
        try:
            return self._tools.dispatch(call.name, call.arguments) or "(empty result)"
        except ToolError as exc:
            return f"Tool error: {exc}"
        except KeyError as exc:
            return f"Tool error: missing argument {exc}"


def _assistant_message(calls) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in calls
        ],
    }


def _tool_message(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}
