import json

import pytest

from reviewer.llm.base import ProviderResponse, ToolCall
from reviewer.llm.loop import ReviewIncomplete, ReviewLoop
from reviewer.tools import RepoTools


class ScriptedProvider:
    """Returns pre-programmed responses and records what it was asked."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def complete(self, messages, tools):
        self.calls.append(list(messages))
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
    provider = ScriptedProvider([submit_call([VALID_FINDING])])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert len(findings) == 1
    assert findings[0].title == "Charge не покрыт тестом"


def test_loop_serves_tool_calls_then_finishes(tools):
    provider = ScriptedProvider([
        ProviderResponse(
            tool_calls=(ToolCall(id="t1", name="list_files", arguments={"path_glob": "internal/*"}),),
            text=None,
        ),
        submit_call([]),
    ])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert findings == []
    tool_messages = [m for m in provider.calls[-1] if m.get("role") == "tool"]
    assert "internal/billing.go" in tool_messages[0]["content"]


def test_loop_reports_tool_errors_back_to_the_model(tools):
    provider = ScriptedProvider([
        ProviderResponse(
            tool_calls=(ToolCall(id="t1", name="read_file", arguments={"path": "../secret"}),),
            text=None,
        ),
        submit_call([]),
    ])
    ReviewLoop(provider, tools).run("system", "user")
    tool_messages = [m for m in provider.calls[-1] if m.get("role") == "tool"]
    assert "escapes the repository" in tool_messages[0]["content"]


def test_loop_retries_once_on_broken_schema(tools):
    broken = {"category": "bug"}
    provider = ScriptedProvider([submit_call([broken]), submit_call([VALID_FINDING])])
    findings = ReviewLoop(provider, tools).run("system", "user")
    assert len(findings) == 1
    assert not provider.responses


def test_loop_fails_when_schema_stays_broken(tools):
    broken = {"category": "bug"}
    provider = ScriptedProvider([submit_call([broken]), submit_call([broken]), submit_call([broken])])
    with pytest.raises(ReviewIncomplete):
        ReviewLoop(provider, tools).run("system", "user")


def test_loop_fails_when_model_never_submits(tools):
    responses = [
        ProviderResponse(
            tool_calls=(ToolCall(id=f"t{i}", name="list_files", arguments={}),), text=None
        )
        for i in range(3)
    ]
    provider = ScriptedProvider(responses)
    with pytest.raises(ReviewIncomplete):
        ReviewLoop(provider, tools, max_iterations=3).run("system", "user")


def test_loop_fails_when_model_answers_with_plain_text(tools):
    provider = ScriptedProvider([ProviderResponse(tool_calls=(), text="всё хорошо")])
    with pytest.raises(ReviewIncomplete):
        ReviewLoop(provider, tools, max_iterations=1).run("system", "user")


def test_arguments_are_parsed_from_json_string(tools):
    payload = json.dumps({"summary": "s", "findings": []})
    provider = ScriptedProvider([
        ProviderResponse(
            tool_calls=(ToolCall(id="c1", name="submit_findings", arguments=json.loads(payload)),),
            text=None,
        )
    ])
    assert ReviewLoop(provider, tools).run("system", "user") == []
