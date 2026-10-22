import json

import pytest

from reviewer.llm.base import ProviderResponse, ToolCall
from reviewer.findings import SUBMIT_FINDINGS_TOOL
from reviewer.llm.loop import BUDGET_EXHAUSTED_REMINDER, ReviewIncomplete, ReviewLoop, iteration_budget
from reviewer.tools import RepoTools


class ScriptedProvider:
    """Returns pre-programmed responses and records what it was asked."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []
        self.required: list[str | None] = []

    def complete(self, messages, tools, required_tool=None):
        self.calls.append(list(messages))
        self.required.append(required_tool)
        if not self.responses:
            raise AssertionError("provider ran out of scripted responses")
        return self.responses.pop(0)


def submit_call(findings, call_id="c1"):
    return ProviderResponse(
        tool_calls=(
            ToolCall(
                id=call_id,
                name="submit_findings",
                arguments={"summary": "итог", "findings": findings},
            ),
        ),
        text=None,
    )


VALID_FINDING = {
    "category": "tests",
    "file": "internal/billing.go",
    "line_start": 3,
    "line_end": 5,
    "title": "Charge не покрыт тестом",
    "problem": "Новая ветка обработки ошибки не проверяется",
    "evidence": "в billing_test.go нет кейса с ошибкой",
    "suggested_fix": "Добавить тест на ошибку Stripe",
    "confidence": "high",
}


@pytest.fixture()
def tools(tmp_path):
    (tmp_path / "internal").mkdir()
    (tmp_path / "internal" / "billing.go").write_text("package billing\n", encoding="utf-8")
    return RepoTools(root=tmp_path)


def test_loop_returns_findings_from_submit(tools):
    provider = ScriptedProvider([
        list_files_call("i1"), list_files_call("i2"), list_files_call("i3"),
        submit_call([VALID_FINDING]),
    ])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert len(findings) == 1
    assert findings[0].title == "Charge не покрыт тестом"


def test_loop_serves_tool_calls_then_finishes(tools):
    provider = ScriptedProvider([
        ProviderResponse(
            tool_calls=(ToolCall(id="t1", name="list_files", arguments={"path_glob": "internal/*"}),),
            text=None,
        ),
        list_files_call("t2"),
        list_files_call("t3"),
        submit_call([]),
    ])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert findings == []
    tool_messages = [m for m in provider.calls[-1] if m.get("role") == "tool"]
    assert "internal/billing.go" in tool_messages[0]["content"]


def test_successful_tool_result_is_wrapped_as_data_for_the_model(tools):
    """read_file/grep/list_files results are the largest prompt-injection surface (the
    model chooses what to read) - they must be wrapped the same way the diff and
    CLAUDE.md are, so content inside a file is never mistaken for an instruction."""
    provider = ScriptedProvider([
        ProviderResponse(
            tool_calls=(ToolCall(id="t1", name="list_files", arguments={"path_glob": "internal/*"}),),
            text=None,
        ),
        list_files_call("t2"),
        list_files_call("t3"),
        submit_call([]),
    ])
    ReviewLoop(provider, tools).run("system", "user")
    tool_messages = [m for m in provider.calls[-1] if m.get("role") == "tool"]
    content = tool_messages[0]["content"]
    assert content.startswith("<tool_result>\n")
    assert content.rstrip().endswith("</tool_result>")
    assert "internal/billing.go" in content


def test_tool_error_is_not_wrapped_as_tool_result(tools):
    # Our own error text is not repository content, so it stays unwrapped.
    provider = ScriptedProvider([
        ProviderResponse(
            tool_calls=(ToolCall(id="t1", name="read_file", arguments={"path": "../secret"}),),
            text=None,
        ),
        list_files_call("t2"),
        list_files_call("t3"),
        submit_call([]),
    ])
    ReviewLoop(provider, tools).run("system", "user")
    tool_messages = [m for m in provider.calls[-1] if m.get("role") == "tool"]
    assert "<tool_result>" not in tool_messages[0]["content"]


def test_loop_reports_tool_errors_back_to_the_model(tools):
    provider = ScriptedProvider([
        ProviderResponse(
            tool_calls=(ToolCall(id="t1", name="read_file", arguments={"path": "../secret"}),),
            text=None,
        ),
        list_files_call("t2"),
        list_files_call("t3"),
        submit_call([]),
    ])
    ReviewLoop(provider, tools).run("system", "user")
    tool_messages = [m for m in provider.calls[-1] if m.get("role") == "tool"]
    assert "escapes the repository" in tool_messages[0]["content"]


def test_loop_retries_once_on_broken_schema(tools):
    broken = {"category": "bug"}
    provider = ScriptedProvider([
        list_files_call("i1"), list_files_call("i2"), list_files_call("i3"),
        submit_call([broken]), submit_call([VALID_FINDING]),
    ])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert len(findings) == 1
    assert not provider.responses


def test_loop_fails_when_schema_stays_broken(tools):
    broken = {"category": "bug"}
    provider = ScriptedProvider([
        list_files_call("i1"), list_files_call("i2"), list_files_call("i3"),
        submit_call([broken]), submit_call([broken]), submit_call([broken]),
    ])
    with pytest.raises(ReviewIncomplete):
        ReviewLoop(provider, tools).run("system", "user")


def test_loop_fails_when_model_never_submits(tools):
    # Three steps of budget plus the one extra step the last step grants when it
    # misses the verdict (2026-10-22): four reads without a verdict is the end.
    responses = [
        ProviderResponse(
            tool_calls=(ToolCall(id=f"t{i}", name="list_files", arguments={}),), text=None
        )
        for i in range(4)
    ]
    provider = ScriptedProvider(responses)
    with pytest.raises(ReviewIncomplete):
        ReviewLoop(provider, tools, max_iterations=3).run("system", "user")
    assert len(provider.calls) == 4


def test_loop_fails_when_model_answers_with_plain_text(tools):
    # With a budget of one, that step is also the last: text there earns the one
    # extra step (2026-10-22), and text again ends the review.
    provider = ScriptedProvider([
        ProviderResponse(tool_calls=(), text="всё хорошо"),
        ProviderResponse(tool_calls=(), text="всё ещё хорошо"),
    ])
    with pytest.raises(ReviewIncomplete):
        ReviewLoop(provider, tools, max_iterations=1).run("system", "user")
    assert len(provider.calls) == 2


def test_loop_recovers_after_one_plain_text_response(tools):
    provider = ScriptedProvider([
        ProviderResponse(tool_calls=(), text="всё хорошо, замечаний нет"),
        list_files_call("t1"),
        list_files_call("t2"),
        list_files_call("t3"),
        submit_call([]),
    ])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert findings == []
    reminder_messages = [m for m in provider.calls[-1] if m.get("role") == "user"]
    assert any("обязан вызвать инструмент" in m["content"] for m in reminder_messages)


def test_loop_fails_after_three_consecutive_text_responses(tools):
    provider = ScriptedProvider([
        ProviderResponse(tool_calls=(), text="раз"),
        ProviderResponse(tool_calls=(), text="два"),
        ProviderResponse(tool_calls=(), text="три"),
    ])
    with pytest.raises(ReviewIncomplete):
        ReviewLoop(provider, tools, max_iterations=3).run("system", "user")
    assert len(provider.calls) == 3


def list_files_call(call_id):
    return ProviderResponse(
        tool_calls=(ToolCall(id=call_id, name="list_files", arguments={}),), text=None
    )


def test_submit_before_enough_investigation_is_reminded_not_accepted(tools):
    provider = ScriptedProvider([
        list_files_call("t1"),
        list_files_call("t2"),
        submit_call([VALID_FINDING], call_id="c1"),  # only 2 investigation calls so far
        list_files_call("t3"),
        submit_call([VALID_FINDING], call_id="c2"),  # now 3 - accepted
    ])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert len(findings) == 1
    reminder_messages = [m for m in provider.calls[-1] if m.get("role") == "tool"]
    assert any("не проверил код инструментами" in m["content"] for m in reminder_messages)


def test_submit_is_eventually_accepted_after_investigation_reminders_exhausted(tools):
    # Model never investigates at all - after MAX_INVESTIGATION_REMINDERS (2) the loop
    # stops nudging and accepts the verdict rather than blocking forever.
    provider = ScriptedProvider([
        submit_call([VALID_FINDING], call_id="c1"),
        submit_call([VALID_FINDING], call_id="c2"),
        submit_call([VALID_FINDING], call_id="c3"),
    ])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert len(findings) == 1
    assert not provider.responses


def test_investigation_reminder_serves_sibling_tool_calls_in_the_same_batch(tools):
    provider = ScriptedProvider([
        ProviderResponse(
            tool_calls=(
                ToolCall(id="s1", name="submit_findings",
                         arguments={"summary": "итог", "findings": [VALID_FINDING]}),
                ToolCall(id="t1", name="list_files", arguments={}),
            ),
            text=None,
        ),
        list_files_call("t2"),
        list_files_call("t3"),
        submit_call([VALID_FINDING], call_id="c2"),
    ])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert len(findings) == 1
    first_round_tool_messages = {
        m["tool_call_id"] for m in provider.calls[1] if m.get("role") == "tool"
    }
    assert first_round_tool_messages == {"s1", "t1"}


def test_arguments_are_parsed_from_json_string(tools):
    payload = json.dumps({"summary": "s", "findings": []})
    provider = ScriptedProvider([
        list_files_call("i1"), list_files_call("i2"), list_files_call("i3"),
        ProviderResponse(
            tool_calls=(ToolCall(id="c1", name="submit_findings", arguments=json.loads(payload)),),
            text=None,
        ),
    ])
    assert ReviewLoop(provider, tools).run("system", "user") == []


def test_loop_serves_sibling_tool_calls_when_submit_in_same_batch_is_invalid(tools):
    broken = {"category": "bug"}
    provider = ScriptedProvider([
        list_files_call("i1"), list_files_call("i2"), list_files_call("i3"),
        ProviderResponse(
            tool_calls=(
                ToolCall(
                    id="s1",
                    name="submit_findings",
                    arguments={"summary": "итог", "findings": [broken]},
                ),
                ToolCall(id="t1", name="list_files", arguments={"path_glob": "internal/*"}),
            ),
            text=None,
        ),
        submit_call([VALID_FINDING]),
    ])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert len(findings) == 1

    tool_messages = {
        m["tool_call_id"]: m["content"] for m in provider.calls[-1] if m.get("role") == "tool"
    }
    assert "s1" in tool_messages
    assert "t1" in tool_messages
    assert "internal/billing.go" in tool_messages["t1"]


def test_loop_reports_invalid_argument_type_to_the_model(tools):
    provider = ScriptedProvider([
        ProviderResponse(
            tool_calls=(
                ToolCall(
                    id="t1",
                    name="read_file",
                    arguments={"path": "internal/billing.go", "offset": "abc"},
                ),
            ),
            text=None,
        ),
        list_files_call("t2"),
        list_files_call("t3"),
        submit_call([]),
    ])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert findings == []

    tool_messages = [m for m in provider.calls[-1] if m.get("role") == "tool"]
    assert "invalid argument" in tool_messages[0]["content"]


# --- Review loop counters ----------------------------------------------------
#
# The conversation grows inside a single run: every iteration resends the whole
# exchange, including the results of read_file and grep. So "how many tokens the
# review took" explains nothing on its own - the iteration count and the volume
# read do. Without these counters any statement about the cause of the growth
# stays a guess.


def test_loop_counts_iterations_tool_calls_and_bytes_read(tmp_path):
    (tmp_path / "main.go").write_text("package main\n\nfunc main() {}\n", encoding="utf-8")
    tools = RepoTools(root=tmp_path)

    read = ProviderResponse(
        tool_calls=(ToolCall(id="t1", name="read_file", arguments={"path": "main.go"}),),
        text=None,
    )
    grep = ProviderResponse(
        tool_calls=(ToolCall(id="t2", name="grep", arguments={"pattern": "func"}),),
        text=None,
    )
    listing = ProviderResponse(
        tool_calls=(ToolCall(id="t3", name="list_files", arguments={}),),
        text=None,
    )
    provider = ScriptedProvider([read, grep, listing, submit_call([])])
    loop = ReviewLoop(provider, tools)

    loop.run("system", "user")

    assert loop.stats.iterations == 4
    assert loop.stats.tool_calls == 3
    # Whatever the tools returned is paid for again as prompt tokens on every
    # later iteration; the byte count is what makes that visible.
    assert loop.stats.tool_result_bytes > len("package main")


def test_loop_stats_start_at_zero_before_a_run(tmp_path):
    loop = ReviewLoop(ScriptedProvider([submit_call([])]), RepoTools(root=tmp_path))

    assert (loop.stats.iterations, loop.stats.tool_calls, loop.stats.tool_result_bytes) == (0, 0, 0)


class ExplodingProvider:
    """Answers once, then fails - a transient API error mid-run."""

    def __init__(self, first):
        self._first = first
        self.calls = 0

    def complete(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return self._first
        raise RuntimeError("provider is down")


def test_iteration_is_counted_only_after_the_model_answered(tmp_path):
    """A counter that ticks before the call reports round trips that never happened.
    For a number whose whole job is to explain where the tokens went, an inflated
    count is worse than no count. Found by review on !8."""
    (tmp_path / "main.go").write_text("package main\n", encoding="utf-8")
    read = ProviderResponse(
        tool_calls=(ToolCall(id="t1", name="read_file", arguments={"path": "main.go"}),),
        text=None,
    )
    loop = ReviewLoop(ExplodingProvider(read), RepoTools(root=tmp_path))

    with pytest.raises(RuntimeError):
        loop.run("system", "user")

    assert loop.stats.iterations == 1


# --- What each finding was formed on --------------------------------------------
# Findings are submitted all at once, so the runtime cannot know which read served
# which finding. What it can record as fact is everything this pass read before the
# verdict, attached to every finding of the pass. The model is not asked to list its
# reads: that would be a claim, and it would change the prompt in the very period
# when the effect of !26 is being measured.

from reviewer.reads import Read  # noqa: E402


def read_file_call(call_id, path, **extra):
    return ProviderResponse(
        tool_calls=(ToolCall(id=call_id, name="read_file", arguments={"path": path, **extra}),),
        text=None,
    )


def grep_call(call_id, pattern):
    return ProviderResponse(
        tool_calls=(ToolCall(id=call_id, name="grep", arguments={"pattern": pattern}),),
        text=None,
    )


def test_findings_carry_what_the_pass_read_before_the_verdict(tools):
    provider = ScriptedProvider([
        read_file_call("r1", "internal/billing.go"),
        grep_call("r2", "package"),
        list_files_call("r3"),
        submit_call([VALID_FINDING]),
    ])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert findings[0].reads == (
        Read("read_file", "internal/billing.go", 1, 1),
        Read("grep", "internal/billing.go", 1, 1),
        Read("list_files", "**/*", None, None),
    )


def test_a_refused_read_is_not_recorded_as_read(tools, tmp_path):
    (tmp_path / ".env").write_text("SECRET=x\n", encoding="utf-8")
    provider = ScriptedProvider([
        read_file_call("r1", ".env"),
        read_file_call("r2", "missing.go"),
        list_files_call("r3"),
        list_files_call("r4"),
        submit_call([VALID_FINDING]),
    ])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert [r.path for r in findings[0].reads] == ["**/*", "**/*"]


def test_reads_served_alongside_a_premature_submit_count(tools):
    # The reminder branch serves the other calls of the same response; the model
    # saw their results before its accepted verdict, so they are part of the record.
    premature = ProviderResponse(
        tool_calls=(
            ToolCall(id="r1", name="read_file", arguments={"path": "internal/billing.go"}),
            ToolCall(id="s1", name="submit_findings", arguments={"summary": "", "findings": []}),
        ),
        text=None,
    )
    provider = ScriptedProvider([
        premature, list_files_call("r2"), list_files_call("r3"), submit_call([VALID_FINDING]),
    ])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert Read("read_file", "internal/billing.go", 1, 1) in findings[0].reads


def test_calls_sent_together_with_the_accepted_verdict_were_never_seen(tools):
    provider = ScriptedProvider([
        list_files_call("r1"), list_files_call("r2"), list_files_call("r3"),
        ProviderResponse(
            tool_calls=(
                ToolCall(id="r4", name="read_file", arguments={"path": "internal/billing.go"}),
                ToolCall(id="s1", name="submit_findings",
                         arguments={"summary": "", "findings": [VALID_FINDING]}),
            ),
            text=None,
        ),
    ])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert all(r.tool == "list_files" for r in findings[0].reads)


def test_a_pass_that_read_nothing_attaches_an_empty_record_not_none(tools):
    provider = ScriptedProvider([
        submit_call([VALID_FINDING], "s1"),
        submit_call([VALID_FINDING], "s2"),
        submit_call([VALID_FINDING], "s3"),
    ])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert findings[0].reads == ()


def test_a_second_run_of_the_same_loop_does_not_inherit_the_first_runs_reads(tools):
    loop = ReviewLoop(ScriptedProvider([
        read_file_call("a1", "internal/billing.go"), list_files_call("a2"), list_files_call("a3"),
        submit_call([]),
        list_files_call("b1"), list_files_call("b2"), list_files_call("b3"),
        submit_call([VALID_FINDING]),
    ]), tools)
    loop.run("system", "user")
    findings = loop.run("system", "user")
    assert all(r.tool == "list_files" for r in findings[0].reads)


class ToolRecordingProvider(ScriptedProvider):
    """A scripted provider that also records which tools each call offered."""

    def __init__(self, responses):
        super().__init__(responses)
        self.offered: list[list[str]] = []

    def complete(self, messages, tools, required_tool=None):
        self.offered.append([tool["function"]["name"] for tool in tools])
        return super().complete(messages, tools, required_tool)


# go !340 (2026-10-08): two runs over a 40-file diff spent 25 iterations reading
# and never submitted -- $4.49 for "ревью не завершено". A bigger diff needs more
# steps; the budget grows with it, up to a ceiling.
def test_the_iteration_budget_grows_with_the_files_under_review():
    assert iteration_budget(0) == 25
    assert iteration_budget(10) == 25
    assert iteration_budget(11) == 26
    assert iteration_budget(20) == 35
    assert iteration_budget(30) == 45
    assert iteration_budget(400) == 45


def test_the_budget_follows_the_files_of_the_pass(tools):
    # Ten reads, then the verdict: with one file the budget is 25 and this fits;
    # what is checked is that the loop sizes itself from changed_files.
    provider = ToolRecordingProvider([list_files_call(f"i{n}") for n in range(30)] + [submit_call([VALID_FINDING])])
    findings = ReviewLoop(provider, tools).run("system", "user", changed_files=16)
    # 16 files -> 31 iterations: 30 reads, the 31st offers only the verdict.
    assert len(findings) == 1
    assert len(provider.offered) == 31
    assert provider.required[-1] == "submit_findings"


# The last iteration of the budget demands the verdict and says why, so a review
# that read a lot ends with the findings it has instead of nothing.
#
# It demands it through required_tool, NOT by narrowing the tool list. Until
# 2026-10-22 the last step offered only submit_findings; on Claude a change of
# tool definitions invalidates the whole prompt cache, so that one step was paid
# at full input price over a ~300K-token conversation. The list stays the same,
# the provider forces the tool where it can, and the reminder says it in words.
def test_the_last_iteration_demands_the_verdict_without_narrowing_the_tools(tools):
    provider = ToolRecordingProvider([
        list_files_call("i1"), list_files_call("i2"), list_files_call("i3"),
        submit_call([VALID_FINDING], call_id="s1"),
    ])
    findings = ReviewLoop(provider, tools, max_iterations=4).run("system", "user")

    assert len(findings) == 1
    assert provider.offered[-1] == provider.offered[0]
    assert len(provider.offered[-1]) > 1
    assert provider.required == [None, None, None, "submit_findings"]
    assert provider.calls[-1][-1] == {"role": "user", "content": BUDGET_EXHAUSTED_REMINDER}


# A provider that cannot force a tool (Opus/Sonnet 5.5 reject it) may still
# reach for a read tool on the last step. It gets one more step: the read is
# refused in its tool result, and the verdict is asked for again.
def test_a_read_on_the_last_step_is_refused_and_the_verdict_asked_once_more(tools):
    provider = ToolRecordingProvider([
        list_files_call("i1"),
        list_files_call("late-read"),
        submit_call([VALID_FINDING], call_id="s1"),
    ])
    findings = ReviewLoop(provider, tools, max_iterations=2).run("system", "user")

    assert len(findings) == 1
    assert provider.required == [None, "submit_findings", "submit_findings"]
    refused = [m for m in provider.calls[-1] if m.get("tool_call_id") == "late-read"]
    assert refused and "submit_findings" in refused[0]["content"]
    # The refused read was not served: nothing from the repository came back.
    assert "<tool_result>" not in refused[0]["content"]


def test_a_second_miss_after_the_extra_step_is_incomplete(tools):
    provider = ToolRecordingProvider([
        list_files_call("i1"), list_files_call("late-1"), list_files_call("late-2"),
    ])
    with pytest.raises(ReviewIncomplete):
        ReviewLoop(provider, tools, max_iterations=2).run("system", "user")


# On the last step the verdict is taken even below the investigation minimum:
# there is no next step to investigate in.
def test_the_last_iteration_takes_the_verdict_without_more_reading(tools):
    provider = ToolRecordingProvider([list_files_call("i1"), submit_call([VALID_FINDING], call_id="s1")])
    findings = ReviewLoop(provider, tools, max_iterations=2).run("system", "user")
    assert len(findings) == 1


def test_a_last_iteration_answered_with_text_is_still_incomplete(tools):
    # Text on the last step gets the same one extra step as a stray read; text
    # again after it is incomplete, as a text-only last step was before.
    provider = ToolRecordingProvider([
        list_files_call("i1"),
        ProviderResponse(tool_calls=(), text="later"),
        ProviderResponse(tool_calls=(), text="still later"),
    ])
    with pytest.raises(ReviewIncomplete):
        ReviewLoop(provider, tools, max_iterations=2).run("system", "user")


def test_text_on_the_last_step_then_the_verdict_is_accepted(tools):
    provider = ToolRecordingProvider([
        list_files_call("i1"),
        ProviderResponse(tool_calls=(), text="later"),
        submit_call([VALID_FINDING], call_id="s1"),
    ])
    assert len(ReviewLoop(provider, tools, max_iterations=2).run("system", "user")) == 1


# Called without changed_files and without max_iterations -- how a caller that
# knows nothing about the diff calls it -- the pass gets the base budget of 25,
# the fixed ceiling it had before the budget began to grow.
def test_without_a_file_count_the_pass_keeps_the_base_budget(tools):
    provider = ToolRecordingProvider([list_files_call(f"i{n}") for n in range(24)] + [submit_call([VALID_FINDING])])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert len(findings) == 1
    assert len(provider.offered) == 25
    assert provider.required[-1] == "submit_findings"
    assert all(required is None for required in provider.required[:-1])
