import pytest

from reviewer.adjudication import Verdict, adjudicate
from reviewer.gitlab_client import Discussion, Note
from reviewer.llm.base import ProviderResponse, ToolCall


class OneShotProvider:
    def __init__(self, response):
        self.response = response
        self.messages = None

    def complete(self, messages, tools):
        self.messages = messages
        return self.response


def verdict_response(accepted: bool, reason: str) -> ProviderResponse:
    return ProviderResponse(
        tool_calls=(ToolCall(id="a1", name="adjudicate",
                             arguments={"accepted": accepted, "reason": reason}),),
        text=None,
    )


DISCUSSION = Discussion(
    id="d1",
    notes=(
        Note(id=1, body="**[bug] Ошибка не проверяется**\n\n<!-- ai-review:fp=0123456789abcdef;file=a.go -->"),
        Note(id=2, body="Не баг: ошибка обрабатывается вызывающим кодом в handler.go:40"),
    ),
    resolved=False,
)


def test_accepted_argument_returns_positive_verdict():
    provider = OneShotProvider(verdict_response(True, "автор прав, ошибка обрабатывается выше"))
    result = adjudicate(provider, DISCUSSION)
    assert result == Verdict(accepted=True, reason="автор прав, ошибка обрабатывается выше")


def test_rejected_argument_returns_negative_verdict():
    provider = OneShotProvider(verdict_response(False, "в handler.go:40 ошибка только логируется"))
    result = adjudicate(provider, DISCUSSION)
    assert result.accepted is False
    assert "handler.go" in result.reason


def test_original_finding_and_reply_are_both_sent_to_the_model():
    provider = OneShotProvider(verdict_response(True, "ок"))
    adjudicate(provider, DISCUSSION)
    conversation = "\n".join(m["content"] for m in provider.messages if m.get("content"))
    assert "Ошибка не проверяется" in conversation
    assert "handler.go:40" in conversation


def test_missing_tool_call_is_treated_as_not_accepted():
    provider = OneShotProvider(ProviderResponse(tool_calls=(), text="ну наверное да"))
    result = adjudicate(provider, DISCUSSION)
    assert result.accepted is False


def test_discussion_without_human_reply_is_rejected_outright():
    only_bot = Discussion(id="d2", notes=(DISCUSSION.notes[0],), resolved=False)
    with pytest.raises(ValueError):
        adjudicate(OneShotProvider(verdict_response(True, "ок")), only_bot)


def test_string_accepted_value_is_treated_as_false():
    """Verify strict type checking: string "false" doesn't become True via truthiness."""
    provider = OneShotProvider(ProviderResponse(
        tool_calls=(ToolCall(id="a1", name="adjudicate",
                             arguments={"accepted": "false", "reason": "string not bool"}),),
        text=None,
    ))
    result = adjudicate(provider, DISCUSSION)
    assert result.accepted is False


def test_only_human_replies_without_marker_are_sent_to_model():
    """Verify bot follow-ups (notes with marker) are excluded from the prompt."""
    discussion_with_bot_followup = Discussion(
        id="d3",
        notes=(
            Note(id=1, body="**[bug] Issue**\n\n<!-- ai-review:fp=0123456789abcdef;file=a.go -->"),
            Note(id=2, body="First human reply: this is fine"),
            Note(id=3, body="Bot follow-up\n\n<!-- ai-review:fp=fedcba9876543210;file=a.go -->"),
            Note(id=4, body="Second human reply: actually no"),
        ),
        resolved=False,
    )
    provider = OneShotProvider(verdict_response(True, "ок"))
    adjudicate(provider, discussion_with_bot_followup)
    conversation = "\n".join(m["content"] for m in provider.messages if m.get("content"))
    # Only human replies should be in the prompt
    assert "First human reply" in conversation
    assert "Second human reply" in conversation
    # Bot follow-up should NOT be in the prompt
    assert "Bot follow-up" not in conversation
