"""Drives the model until it submits a structured verdict."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

from reviewer.findings import (
    SUBMIT_FINDINGS_TOOL,
    Finding,
    FindingsSchemaError,
    parse_findings,
)
from reviewer.llm.base import LLMProvider, ToolCall
from reviewer.reads import Read, reads_from_call
from reviewer.tools import TOOL_SCHEMAS, RepoTools, ToolError

MAX_SCHEMA_RETRIES = 1
MAX_TEXT_REMINDERS = 2
MIN_INVESTIGATION_CALLS = 3
MAX_INVESTIGATION_REMINDERS = 2

# The step budget of one review pass. 25 was a fixed ceiling until 2026-10-09
# and fits a merge request of about ten files. Each file beyond that adds a
# step, up to MAX_ITERATIONS: two runs over the forty files of jobbytime-go
# !340 spent all 25 steps reading and submitted nothing ($4.49, no review).
BASE_ITERATIONS = 25
BASE_FILES = 10
MAX_ITERATIONS = 45

BUDGET_EXHAUSTED_REMINDER = (
    "Бюджет шагов исчерпан. Сдай вердикт сейчас через submit_findings — по тому, "
    "что уже прочитал. Другие инструменты на этом шаге не выполняются."
)

# The answer to a read made on the last step instead of the verdict. Not served:
# the budget is spent, and reading on would only grow the bill.
FINAL_STEP_REFUSAL = (
    "Не выполнено: бюджет шагов исчерпан. Вызови submit_findings сейчас — по тому, "
    "что уже прочитал."
)

VERDICT_TOOL = "submit_findings"


def iteration_budget(changed_files: int) -> int:
    """Steps for a pass over changed_files files: BASE_ITERATIONS, one more per
    file beyond BASE_FILES, never more than MAX_ITERATIONS."""
    extra = max(0, changed_files - BASE_FILES)
    return min(BASE_ITERATIONS + extra, MAX_ITERATIONS)

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
        max_iterations: int | None = None,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._max_iterations = max_iterations
        self.stats = LoopStats()
        self._reads: list[Read] = []

    def run(self, system_prompt: str, user_prompt: str, changed_files: int | None = None) -> list[Finding]:
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        available = [*TOOL_SCHEMAS, SUBMIT_FINDINGS_TOOL]
        schema_retries = 0
        text_reminders = 0
        investigation_reminders = 0
        investigation_calls = 0
        self._reads = []
        # An explicit max_iterations wins (tests, callers that know better);
        # otherwise the pass is sized by the files it reviews.
        limit = (
            self._max_iterations
            if self._max_iterations is not None
            else iteration_budget(changed_files or 0)
        )

        # One extra step past the budget, granted only when the last step missed
        # the verdict. See below.
        extra_step_granted = False
        for step in range(limit + 1):
            if step == limit and not extra_step_granted:
                break
            # The last step demands the verdict and says so: a pass that read a
            # lot ends with what it found, not with "never called submit_findings"
            # and a paid rerun.
            #
            # It demands it through required_tool, with the SAME tool list as every
            # other step. Until 2026-10-22 the list was narrowed to submit_findings
            # alone; on Claude any change of tool definitions invalidates the whole
            # prompt cache, so that one step was paid at full input price over a
            # conversation of ~300K tokens.
            last = step >= limit - 1
            if step == limit - 1:
                messages.append({"role": "user", "content": BUDGET_EXHAUSTED_REMINDER})
            # Counted after the answer, not before it: a provider error would
            # otherwise report a round trip that never happened, and this number
            # exists precisely to explain where the tokens went.
            response = self._complete(messages, available, VERDICT_TOOL if last else None)
            self.stats.iterations += 1

            if last and not any(c.name == VERDICT_TOOL for c in response.tool_calls):
                # A provider that cannot force a tool (Opus/Sonnet 5.5 reject it)
                # may still read or talk on the last step. One more step, with the
                # read refused rather than served; a second miss is final.
                if extra_step_granted:
                    break
                # A model that has already been told twice to call a tool and keeps
                # talking gets no third chance here either: MAX_TEXT_REMINDERS is
                # the rule, the extra step is not a way around it.
                if not response.tool_calls and text_reminders >= MAX_TEXT_REMINDERS:
                    raise ReviewIncomplete(
                        "model kept answering with plain text instead of calling a tool"
                    )
                extra_step_granted = True
                if response.tool_calls:
                    messages.append(_assistant_message(response.tool_calls))
                    for call in response.tool_calls:
                        messages.append(_tool_message(call.id, FINAL_STEP_REFUSAL))
                else:
                    messages.append({"role": "assistant", "content": response.text or ""})
                    messages.append({"role": "user", "content": FINAL_STEP_REFUSAL})
                continue
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
                # On the last step there is no next step to investigate in.
                if (
                    not last
                    and investigation_calls < MIN_INVESTIGATION_CALLS
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
                    findings = parse_findings(submit.arguments)
                    reads = tuple(self._reads)
                    return [replace(finding, reads=reads) for finding in findings]
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

    def _complete(self, messages: list[dict], tools: list[dict], required_tool: str | None):
        # Passed only when there is something to require, so a provider written
        # against the two-argument contract keeps working on ordinary steps.
        if required_tool is None:
            return self._provider.complete(messages, tools)
        return self._provider.complete(messages, tools, required_tool=required_tool)

    def _serve(self, call: ToolCall) -> str:
        if call.name in INVESTIGATION_TOOL_NAMES:
            self.stats.tool_calls += 1
        try:
            raw = self._tools.dispatch(call.name, call.arguments)
            self._reads.extend(reads_from_call(call.name, call.arguments, raw or ""))
            result = raw or "(empty result)"
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
