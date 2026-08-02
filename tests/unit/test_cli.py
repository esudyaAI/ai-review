from pathlib import Path

import pytest

from reviewer.__main__ import (
    EXIT_CLEAN,
    EXIT_FINDINGS,
    EXIT_TECHNICAL,
    EXIT_TOO_LARGE,
    _adjudicator_for,
    main,
    review,
)
from reviewer.config import Config, Limits
from reviewer.diff import FileDiff, MergeRequestDiff
from reviewer.findings import Finding, render_comment
from reviewer.gitlab_client import Discussion, Note
from reviewer.llm.base import ProviderResponse, ToolCall
from reviewer.llm.loop import ReviewIncomplete


class FakeClient:
    def __init__(self, mr, discussions=(), fail_create_discussion_on_call=None):
        self.mr = mr
        self.discussions = list(discussions)
        self.created: list[Finding] = []
        self.resolved: list[str] = []
        self.replies: list[tuple[str, str]] = []
        self._fail_create_discussion_on_call = fail_create_discussion_on_call
        self._create_discussion_calls = 0

    def fetch_diff(self, mr_iid):
        return self.mr

    def list_discussions(self, mr_iid):
        return self.discussions

    def create_discussion(self, mr_iid, finding, mr):
        self._create_discussion_calls += 1
        if self._create_discussion_calls == self._fail_create_discussion_on_call:
            raise RuntimeError("GitLab API 500")
        self.created.append(finding)
        return f"d-{finding.fingerprint}"

    def resolve(self, mr_iid, discussion_id, body):
        self.resolved.append(discussion_id)

    def reply(self, mr_iid, discussion_id, body):
        self.replies.append((discussion_id, body))


class FakeReviewer:
    def __init__(self, findings=(), error=None):
        self.findings = list(findings)
        self.error = error

    def run(self, system_prompt, user_prompt):
        if self.error:
            raise self.error
        return self.findings


def make_mr():
    return MergeRequestDiff(
        base_sha="b" * 40, head_sha="h" * 40, start_sha="s" * 40,
        files=(FileDiff("a.go", "a.go", "@@ -1,2 +1,3 @@\n x\n+y\n", (2,)),),
    )


def make_config(**overrides):
    data = dict(
        test_command="", provider="openai",
        base_url="https://api.openai.com/v1", model="m", model_high_risk="m",
        high_risk_paths=(), limits=Limits(),
    )
    data.update(overrides)
    return Config(**data)


def finding():
    return Finding(
        category="bug", file="a.go", line_start=2, line_end=2, title="Ошибка",
        problem="p", evidence="e", suggested_fix="s", confidence="high",
    )


def test_clean_review_exits_zero(tmp_path):
    client = FakeClient(make_mr())
    code = review(make_config(), tmp_path, client, lambda model: FakeReviewer([]), lambda d: None, mr_iid=7)
    assert code == EXIT_CLEAN
    assert client.created == []


def test_findings_are_posted_and_block(tmp_path):
    client = FakeClient(make_mr())
    code = review(make_config(), tmp_path, client, lambda model: FakeReviewer([finding()]), lambda d: None, mr_iid=7)
    assert code == EXIT_FINDINGS
    assert [f.title for f in client.created] == ["Ошибка"]


def test_low_confidence_findings_are_dropped(tmp_path):
    weak = Finding(
        category="architecture", file="a.go", line_start=2, line_end=2,
        title="Можно красивее", problem="p", evidence="e", suggested_fix="s",
        confidence="medium",
    )
    client = FakeClient(make_mr())
    code = review(make_config(), tmp_path, client, lambda model: FakeReviewer([weak]), lambda d: None, mr_iid=7)
    assert code == EXIT_CLEAN
    assert client.created == []


def test_fixed_finding_is_resolved(tmp_path):
    old = finding()
    thread = Discussion(id=f"d-{old.fingerprint}", notes=(Note(id=1, body=render_comment(old)),), resolved=False)
    client = FakeClient(make_mr(), [thread])
    code = review(make_config(), tmp_path, client, lambda model: FakeReviewer([]), lambda d: None, mr_iid=7)
    assert client.resolved == [thread.id]
    assert code == EXIT_CLEAN


def test_accepted_author_argument_resolves_thread(tmp_path):
    old = finding()
    thread = Discussion(
        id=f"d-{old.fingerprint}",
        notes=(Note(id=1, body=render_comment(old)), Note(id=2, body="не баг, проверяется выше")),
        resolved=False,
    )
    client = FakeClient(make_mr(), [thread])

    class AcceptingVerdict:
        accepted = True
        reason = "автор прав"

    code = review(make_config(), tmp_path, client, lambda model: FakeReviewer([old]),
                  lambda d: AcceptingVerdict(), mr_iid=7)
    assert client.resolved == [thread.id]
    assert code == EXIT_CLEAN


def test_rejected_author_argument_keeps_thread_open(tmp_path):
    old = finding()
    thread = Discussion(
        id=f"d-{old.fingerprint}",
        notes=(Note(id=1, body=render_comment(old)), Note(id=2, body="не баг")),
        resolved=False,
    )
    client = FakeClient(make_mr(), [thread])

    class RejectingVerdict:
        accepted = False
        reason = "аргумент не подтверждается кодом"

    code = review(make_config(), tmp_path, client, lambda model: FakeReviewer([old]),
                  lambda d: RejectingVerdict(), mr_iid=7)
    assert client.resolved == []
    assert client.replies and "аргумент не подтверждается кодом" in client.replies[0][1]
    assert code == EXIT_FINDINGS


def test_oversized_diff_exits_with_its_own_code(tmp_path):
    client = FakeClient(make_mr())
    config = make_config(limits=Limits(max_files=0, max_changed_lines=0))
    code = review(config, tmp_path, client, lambda model: FakeReviewer([]), lambda d: None, mr_iid=7)
    assert code == EXIT_TOO_LARGE


def test_incomplete_review_is_a_technical_error(tmp_path):
    client = FakeClient(make_mr())
    reviewer = FakeReviewer(error=ReviewIncomplete("model never called submit_findings"))
    code = review(make_config(), tmp_path, client, lambda model: reviewer, lambda d: None, mr_iid=7)
    assert code == EXIT_TECHNICAL


def test_create_discussion_failure_does_not_abort_the_run(tmp_path, capsys):
    """A GitLab failure (network blip, 500, etc.) publishing one finding must not hide
    every other finding or crash review() - each mutating call is isolated, and the exit
    code still reflects the real blocking findings regardless of publishing success."""
    # Different line buckets (10-line windows) so each finding gets a distinct
    # fingerprint - fingerprint no longer depends on title.
    findings = [
        Finding(
            category="bug", file="a.go", line_start=2 + i * 10, line_end=2 + i * 10,
            title=f"Ошибка {i}", problem="p", evidence="e", suggested_fix="s",
            confidence="high",
        )
        for i in range(3)
    ]
    # create_discussion blows up on the 2nd of 3 calls, simulating a GitLab API failure
    # mid-loop (e.g. after successfully posting the first discussion).
    client = FakeClient(make_mr(), fail_create_discussion_on_call=2)
    code = review(make_config(), tmp_path, client, lambda model: FakeReviewer(findings), lambda d: None, mr_iid=7)
    # There are still 3 blocking findings, so the run must still block the merge even
    # though publishing the 2nd one failed.
    assert code == EXIT_FINDINGS
    # The 1st and 3rd discussions were posted; only the 2nd (which raised) is missing.
    assert len(client.created) == 2
    captured = capsys.readouterr()
    assert "не удалось опубликовать находку" in captured.err
    # The summary line must report the actually-succeeded count (2), not the planned
    # count (3) - otherwise it lies about what a reviewer will actually see in the MR.
    assert "новых замечаний 2" in captured.out


def test_high_risk_path_selects_the_stronger_model(tmp_path):
    seen: list[str] = []

    def make_reviewer(model):
        seen.append(model)
        return FakeReviewer([])

    mr = MergeRequestDiff(
        base_sha="b" * 40, head_sha="h" * 40, start_sha="s" * 40,
        files=(FileDiff("internal/billing/x.go", "internal/billing/x.go",
                        "@@ -1,2 +1,3 @@\n x\n+y\n", (2,)),),
    )
    config = make_config(model="weak", model_high_risk="strong",
                         high_risk_paths=("internal/billing/**",))
    review(config, tmp_path, FakeClient(mr), make_reviewer, lambda d: None, mr_iid=7)
    assert seen == ["strong"]


class FakeAdjudicationProvider:
    def __init__(self, response):
        self.response = response

    def complete(self, messages, tools):
        return self.response


def test_high_risk_discussion_selects_the_stronger_model_for_adjudication():
    """A dispute over a finding on a high-risk path (e.g. billing) must be adjudicated
    by the stronger model, not always the base config.model - otherwise a weaker model
    resolves a dispute over a finding the stronger model produced."""
    seen: list[str] = []

    def make_provider(model):
        seen.append(model)
        return FakeAdjudicationProvider(ProviderResponse(
            tool_calls=(ToolCall(id="a1", name="adjudicate",
                                 arguments={"accepted": True, "reason": "ок"}),),
            text=None,
        ))

    config = make_config(model="weak", model_high_risk="strong",
                         high_risk_paths=("internal/billing/**",))
    discussion = Discussion(
        id="d1",
        notes=(
            Note(id=1, body="**[bug] x**\n\n<!-- ai-review:fp=0123456789abcdef;"
                             "file=internal/billing/x.go -->"),
            Note(id=2, body="человеческий ответ автора"),
        ),
        resolved=False,
    )

    adjudicator = _adjudicator_for(config, make_provider)
    adjudicator(discussion)
    assert seen == ["strong"]


def test_main_returns_technical_error_when_env_var_is_missing(tmp_path, monkeypatch):
    """A missing required env var (OPENAI_API_KEY etc.) must be classified as a technical
    error (EXIT_TECHNICAL), not an uncaught exception that crashes with Python's exit
    code 1 - which, per contract, means 'there are blocking findings'."""
    (tmp_path / ".ai-review.yml").write_text("model: m\n", encoding="utf-8")
    monkeypatch.setenv("CI_PROJECT_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = main(["review", "--mr", "7"])
    assert code == EXIT_TECHNICAL


def test_main_returns_technical_error_on_broken_config(tmp_path, monkeypatch):
    """A broken .ai-review.yml (e.g. bad YAML, or one caught by Config.validate()) must
    not crash main() with an uncaught ConfigError."""
    (tmp_path / ".ai-review.yml").write_text("test_command: x\n", encoding="utf-8")
    monkeypatch.setenv("CI_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    code = main(["review", "--mr", "7"])
    assert code == EXIT_TECHNICAL
