from pathlib import Path

import pytest

from reviewer.__main__ import EXIT_CLEAN, EXIT_FINDINGS, EXIT_TECHNICAL, EXIT_TOO_LARGE, review
from reviewer.config import Config, Limits
from reviewer.diff import FileDiff, MergeRequestDiff
from reviewer.findings import Finding, render_comment
from reviewer.gitlab_client import Discussion, Note
from reviewer.llm.loop import ReviewIncomplete


class FakeClient:
    def __init__(self, mr, discussions=()):
        self.mr = mr
        self.discussions = list(discussions)
        self.created: list[Finding] = []
        self.resolved: list[str] = []
        self.replies: list[tuple[str, str]] = []

    def fetch_diff(self, mr_iid):
        return self.mr

    def list_discussions(self, mr_iid):
        return self.discussions

    def create_discussion(self, mr_iid, finding, mr):
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
        linter="none", test_command="", provider="openai",
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
