"""Drives the model until it submits a structured verdict."""

from __future__ import annotations

import json
from dataclasses import dataclass

from reviewer.findings import (
    SUBMIT_FINDINGS_TOOL,
    Finding,
    FindingsSchemaError,
    parse_findings,
)
from reviewer.llm.base import LLMProvider, ToolCall
from reviewer.tools import TOOL_SCHEMAS, RepoTools, ToolError

MAX_SCHEMA_RETRIES = 1
MAX_TEXT_REMINDERS = 2
MIN_INVESTIGATION_CALLS = 3
MAX_INVESTIGATION_REMINDERS = 2

TOOL_REQUIRED_REMINDER = (
    "Ты обязан вызвать инструмент: read_file, grep, list_files или submit_findings. "
    "Обычный текст недопустим."
)

INVESTIGATION_REQUIRED_REMINDER = (
    "Ты не проверил код инструментами. Прежде чем сдавать вердикт, открой "
    "изменённые файлы целиком, найди тесты на новые функции через grep по "
    "*_test.go, проверь язык комментариев."
)

# Names of the read-only investigation tools (read_file/grep/list_files) - distinct
# from submit_findings, which is the verdict itself, not investigation.
INVESTIGATION_TOOL_NAMES = frozenset(schema["function"]["name"] for schema in TOOL_SCHEMAS)


class ReviewIncomplete(Exception):
    """Raised when the model never produced a usable verdict."""


@dataclass
class LoopStats:
    """What the run cost in shape, not in money.

    Every iteration resends the whole conversation, so the result of a read_file
    early on is paid for again by each call after it. Tokens alone cannot show
    that; iterations and bytes read can."""

    iterations: int = 0
    tool_calls: int = 0
    tool_result_bytes: int = 0


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
        self.stats = LoopStats()

    def run(self, system_prompt: str, user_prompt: str) -> list[Finding]:
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        available = [*TOOL_SCHEMAS, SUBMIT_FINDINGS_TOOL]
        schema_retries = 0
        text_reminders = 0
        investigation_reminders = 0
        investigation_calls = 0

        for _ in range(self._max_iterations):
            # Counted after the answer, not before it: a provider error would
            # otherwise report a round trip that never happened, and this number
            # exists precisely to explain where the tokens went.
            response = self._provider.complete(messages, available)
            self.stats.iterations += 1
            if not response.tool_calls:
                if text_reminders >= MAX_TEXT_REMINDERS:
                    raise ReviewIncomplete(
                        "model kept answering with plain text instead of calling a tool"
                    )
                text_reminders += 1
                messages.append({"role": "assistant", "content": response.text or ""})
                messages.append({"role": "user", "content": TOOL_REQUIRED_REMINDER})
                continue

            submit = next((c for c in response.tool_calls if c.name == "submit_findings"), None)
            if submit is not None:
                # A verdict formed without reading any code is a guess, not a review.
                # Nudge the model to investigate first; after MAX_INVESTIGATION_REMINDERS
                # give up nudging and accept whatever verdict it hands us - further
                # blocking would only burn tokens with no guarantee of compliance.
                if (
                    investigation_calls < MIN_INVESTIGATION_CALLS
                    and investigation_reminders < MAX_INVESTIGATION_REMINDERS
                ):
                    investigation_reminders += 1
                    messages.append(_assistant_message(response.tool_calls))
                    messages.append(_tool_message(submit.id, INVESTIGATION_REQUIRED_REMINDER))
                    for call in response.tool_calls:
                        if call.id == submit.id:
                            continue
                        messages.append(_tool_message(call.id, self._serve(call)))
                        if call.name in INVESTIGATION_TOOL_NAMES:
                            investigation_calls += 1
                    continue

                try:
                    return parse_findings(submit.arguments)
                except FindingsSchemaError as exc:
                    if schema_retries >= MAX_SCHEMA_RETRIES:
                        raise ReviewIncomplete(f"submit_findings stayed invalid: {exc}") from exc
                    schema_retries += 1
                    messages.append(_assistant_message(response.tool_calls))
                    messages.append(_tool_message(submit.id, f"Invalid payload: {exc}. Retry."))
                    for call in response.tool_calls:
                        if call.id == submit.id:
                            continue
                        messages.append(_tool_message(call.id, self._serve(call)))
                        if call.name in INVESTIGATION_TOOL_NAMES:
                            investigation_calls += 1
                    continue

            messages.append(_assistant_message(response.tool_calls))
            for call in response.tool_calls:
                messages.append(_tool_message(call.id, self._serve(call)))
                if call.name in INVESTIGATION_TOOL_NAMES:
                    investigation_calls += 1

        raise ReviewIncomplete("model never called submit_findings")

    def _serve(self, call: ToolCall) -> str:
        if call.name in INVESTIGATION_TOOL_NAMES:
            self.stats.tool_calls += 1
        try:
            result = self._tools.dispatch(call.name, call.arguments) or "(empty result)"
            # read_file/grep/list_files can return arbitrary repository content - the
            # largest prompt-injection surface, since the model chooses what to read.
            # Wrap it as data, same as the diff and CLAUDE.md, so it's never mistaken
            # for an instruction. Our own error text below is not repository content,
            # so it's left unwrapped.
            wrapped = f"<tool_result>\n{result}\n</tool_result>"
            if call.name in INVESTIGATION_TOOL_NAMES:
                self.stats.tool_result_bytes += len(wrapped.encode("utf-8"))
            return wrapped
        except ToolError as exc:
            return f"Tool error: {exc}"
        except KeyError as exc:
            return f"Tool error: missing argument {exc}"
        except (ValueError, TypeError) as exc:
            return f"Tool error: invalid argument ({exc})"


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
