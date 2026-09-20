from pathlib import Path

import pytest

from reviewer.llm.usage import ADJUDICATION_PHASE
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
    # All three lines have to be in the diff: the gate drops findings on lines
    # the merge request did not add, and make_mr()'s single added line would
    # leave one finding to publish - the isolation this test is about would
    # never be exercised.
    wide_mr = MergeRequestDiff(
        base_sha="b" * 40, head_sha="h" * 40, start_sha="s" * 40,
        files=(FileDiff("a.go", "a.go", "@@ -1,30 +1,30 @@\n x\n", (2, 12, 22)),),
    )
    # create_discussion blows up on the 2nd of 3 calls, simulating a GitLab API failure
    # mid-loop (e.g. after successfully posting the first discussion).
    client = FakeClient(wide_mr, fail_create_discussion_on_call=2)
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
    phases: list[str] = []

    def make_provider(model, phase="review"):
        seen.append(model)
        phases.append(phase)
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
    # The adjudication pass must announce itself as such, or its tokens land in the
    # review bucket and the breakdown stops meaning anything.
    assert phases == [ADJUDICATION_PHASE]

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


def make_mixed_mr():
    return MergeRequestDiff(
        base_sha="b" * 40, head_sha="h" * 40, start_sha="s" * 40,
        files=(
            FileDiff("internal/service/auth_service.go", "internal/service/auth_service.go",
                     "@@ -1,1 +1,2 @@\n x\n+y\n", (2,)),
            FileDiff("internal/stats/pipeline_stats.go", "internal/stats/pipeline_stats.go",
                     "@@ -1,1 +1,2 @@\n x\n+z\n", (2,)),
        ),
    )


class RecordingReviewerFactory:
    """Remembers which model was asked to review which files."""

    def __init__(self, findings_by_model=None):
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self._findings = findings_by_model or {}

    def __call__(self, model):
        factory = self

        class _Reviewer:
            def run(self_inner, system_prompt, user_prompt):
                files = tuple(
                    line[2:] for line in user_prompt.splitlines() if line.startswith("- ")
                )
                factory.calls.append((model, files))
                return list(factory._findings.get(model, []))

        return _Reviewer()


# The cost driver, measured on jobbytime-go !29: one auth file made the strong
# model review all nine files in the diff. Each half must now go to its own model.
def test_high_risk_files_are_reviewed_separately_from_the_rest(tmp_path):
    config = make_config(model="mini", model_high_risk="strong",
                         high_risk_paths=("internal/service/auth_*",))
    factory = RecordingReviewerFactory()
    client = FakeClient(make_mixed_mr())

    review(config, tmp_path, client, factory, lambda d: None, mr_iid=7)

    assert dict(factory.calls) == {
        "strong": ("internal/service/auth_service.go",),
        "mini": ("internal/stats/pipeline_stats.go",),
    }


def test_findings_from_both_passes_are_reported(tmp_path):
    high = Finding(category="security", file="internal/service/auth_service.go", line_start=2,
                   line_end=2, title="Дыра", problem="p", evidence="e", suggested_fix="s",
                   confidence="high")
    low = Finding(category="bug", file="internal/stats/pipeline_stats.go", line_start=2,
                  line_end=2, title="Течёт", problem="p", evidence="e", suggested_fix="s",
                  confidence="high")
    config = make_config(model="mini", model_high_risk="strong",
                         high_risk_paths=("internal/service/auth_*",))
    client = FakeClient(make_mixed_mr())

    code = review(config, tmp_path, client,
                  RecordingReviewerFactory({"strong": [high], "mini": [low]}),
                  lambda d: None, mr_iid=7)

    assert code == EXIT_FINDINGS
    assert sorted(f.title for f in client.created) == ["Дыра", "Течёт"]


# A diff that is entirely high-risk, or entirely not, must stay a single pass:
# an empty half would otherwise pay a whole review for no files.
def test_a_single_sided_diff_is_one_pass(tmp_path):
    config = make_config(model="mini", model_high_risk="strong",
                         high_risk_paths=("internal/service/auth_*",))
    factory = RecordingReviewerFactory()

    review(config, tmp_path, FakeClient(make_mr()), factory, lambda d: None, mr_iid=7)
    assert factory.calls == [("mini", ("a.go",))]


def test_without_a_distinct_strong_model_there_is_one_pass(tmp_path):
    config = make_config(model="mini", model_high_risk="mini",
                         high_risk_paths=("internal/service/auth_*",))
    factory = RecordingReviewerFactory()

    review(config, tmp_path, FakeClient(make_mixed_mr()), factory, lambda d: None, mr_iid=7)
    assert len(factory.calls) == 1
    assert factory.calls[0][0] == "mini"


# An empty model_high_risk must never reach the provider as a model name. The
# guard that prevents it is the empty high-risk half plus the `if not files:
# continue` skip, which is easy to read as "the first pass runs anyway".
def test_empty_strong_model_never_reaches_the_provider(tmp_path):
    config = make_config(model="mini", model_high_risk="",
                         high_risk_paths=("internal/service/auth_*",))
    factory = RecordingReviewerFactory()

    code = review(config, tmp_path, FakeClient(make_mixed_mr()), factory,
                  lambda d: None, mr_iid=7)

    assert code == EXIT_CLEAN
    assert [model for model, _ in factory.calls] == ["mini"]
    assert factory.calls[0][1] == (
        "internal/service/auth_service.go", "internal/stats/pipeline_stats.go",
    ), "an all-high-risk diff must still be reviewed, by the base model"


def test_all_high_risk_diff_with_empty_strong_model_is_still_reviewed(tmp_path):
    mr = MergeRequestDiff(
        base_sha="b" * 40, head_sha="h" * 40, start_sha="s" * 40,
        files=(FileDiff("internal/service/auth_a.go", "internal/service/auth_a.go",
                        "@@ -1,1 +1,2 @@\n x\n+y\n", (2,)),),
    )
    config = make_config(model="mini", model_high_risk="",
                         high_risk_paths=("internal/service/auth_*",))
    factory = RecordingReviewerFactory()

    review(config, tmp_path, FakeClient(mr), factory, lambda d: None, mr_iid=7)

    assert factory.calls == [("mini", ("internal/service/auth_a.go",))]


# A diff with nothing high-risk must skip the strong pass and still carry every
# file into the base one. test_a_single_sided_diff_is_one_pass shows the skip on
# a one-file diff; this one shows that subset() does not lose files on the way.
def test_diff_without_high_risk_files_reviews_all_of_them_in_one_pass(tmp_path):
    mr = MergeRequestDiff(
        base_sha="b" * 40, head_sha="h" * 40, start_sha="s" * 40,
        files=(
            FileDiff("internal/stats/pipeline_stats.go", "internal/stats/pipeline_stats.go",
                     "@@ -1,1 +1,2 @@\n x\n+y\n", (2,)),
            FileDiff("internal/vacancies/updater.go", "internal/vacancies/updater.go",
                     "@@ -1,1 +1,2 @@\n x\n+z\n", (2,)),
            FileDiff("README.md", "README.md", "@@ -1,1 +1,2 @@\n x\n+w\n", (2,)),
        ),
    )
    config = make_config(model="mini", model_high_risk="strong",
                         high_risk_paths=("internal/service/auth_*", "*stripe*"))
    factory = RecordingReviewerFactory()

    review(config, tmp_path, FakeClient(mr), factory, lambda d: None, mr_iid=7)

    assert factory.calls == [("mini", (
        "internal/stats/pipeline_stats.go", "internal/vacancies/updater.go", "README.md",
    ))]


# End to end: review() has to hand the gate the diff's added lines, or the rule
# above is dead code in production.
def test_finding_outside_the_diff_is_not_published_and_does_not_block(tmp_path, capsys):
    off_diff = Finding(
        category="rules", file="a.go", line_start=132, line_end=132,
        title="Русский комментарий", problem="p", evidence="e", suggested_fix="s",
        confidence="high",
    )
    client = FakeClient(make_mr())
    code = review(make_config(), tmp_path, client, lambda model: FakeReviewer([off_diff]),
                  lambda d: None, mr_iid=7)
    assert client.created == []
    assert code == EXIT_CLEAN
    captured = capsys.readouterr()
    assert "a.go:132" in captured.out
    assert "вне изменённого диффа" in captured.out


# --- Outcome classification and readable technical errors -------------------
#
# A technical failure and "there are findings" were already told apart by the exit
# code (2 against 1), but a person opening a failed job saw only the exception
# text - and httpx stringifies timeouts and dropped connections to the empty
# string. Measured on httpx 0.28: str(ReadTimeout("")), str(ConnectTimeout("")),
# str(ConnectError("")) and str(RemoteProtocolError("")) all return ''. So the log
# printed the Russian prefix and stopped at the colon, saying nothing.
#
# The Russian string literals below are the reviewer's own output, i.e. data.


class TimingOutClient(FakeClient):
    """A client that fails the way the network fails: an exception with no text."""

    def __init__(self, mr, error):
        super().__init__(mr)
        self.error = error

    def fetch_diff(self, mr_iid):
        raise self.error


def test_technical_error_names_the_exception_type_when_it_has_no_text(tmp_path, capsys):
    import httpx

    client = TimingOutClient(make_mr(), httpx.ReadTimeout(""))
    code = review(make_config(), tmp_path, client, lambda model: FakeReviewer([]),
                  lambda d: None, mr_iid=7)

    assert code == EXIT_TECHNICAL
    err = capsys.readouterr().err
    assert "ReadTimeout" in err, err
    # The line must not end at a colon - that is exactly how the old output looked.
    assert not any(line.rstrip().endswith(":") for line in err.splitlines()), err


def test_technical_error_keeps_the_exception_text_when_there_is_one(tmp_path, capsys):
    client = TimingOutClient(make_mr(), RuntimeError("GitLab returned 502"))
    review(make_config(), tmp_path, client, lambda model: FakeReviewer([]),
           lambda d: None, mr_iid=7)

    err = capsys.readouterr().err
    assert "RuntimeError" in err, err
    assert "GitLab returned 502" in err, err


def test_outcome_line_names_every_exit_code():
    """The last line of the log must name the class of outcome in words, not only by
    code: in GitLab's list of jobs, "failed because it found something" and "failed
    because it could not review" look identical."""
    from reviewer.__main__ import outcome_line

    for code in (EXIT_CLEAN, EXIT_FINDINGS, EXIT_TECHNICAL, EXIT_TOO_LARGE):
        line = outcome_line(code)
        assert str(code) in line, line
        assert line.startswith("AI review: итог"), line

    assert "замечани" in outcome_line(EXIT_FINDINGS)
    assert "не состоялось" in outcome_line(EXIT_TECHNICAL)
    assert "не состоялось" in outcome_line(EXIT_TOO_LARGE)
    assert "не состоялось" not in outcome_line(EXIT_FINDINGS)
    assert "не состоялось" not in outcome_line(EXIT_CLEAN)


def test_outcome_line_distinguishes_the_two_reasons_a_review_did_not_happen():
    """Both codes mean the review did not happen, but they are fixed differently: 2
    by a retry or by fixing infrastructure, 3 by splitting the merge request."""
    from reviewer.__main__ import outcome_line

    assert outcome_line(EXIT_TECHNICAL) != outcome_line(EXIT_TOO_LARGE)


def test_main_prints_the_outcome_line_on_a_startup_failure(tmp_path, monkeypatch, capsys):
    (tmp_path / ".ai-review.yml").write_text("model: m\n", encoding="utf-8")
    monkeypatch.setenv("CI_PROJECT_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = main(["review", "--mr", "7"])

    assert code == EXIT_TECHNICAL
    captured = capsys.readouterr()
    assert "AI review: итог" in (captured.out + captured.err)


def _migration_mr():
    return MergeRequestDiff(
        base_sha="b" * 40, head_sha="h" * 40, start_sha="s" * 40,
        files=(
            FileDiff("migrations/new/1_add.sql", "migrations/new/1_add.sql",
                     "@@ -0,0 +1,1 @@\n+ALTER TABLE t ADD COLUMN IF NOT EXISTS c text;\n", (1,)),
            FileDiff("migrations/new/2_drop.sql", "migrations/new/2_drop.sql",
                     "@@ -0,0 +1,1 @@\n+ALTER TABLE t DROP COLUMN c;\n", (1,)),
        ),
    )


def _docs_mr():
    return MergeRequestDiff(
        base_sha="b" * 40, head_sha="h" * 40, start_sha="s" * 40,
        files=(
            FileDiff("README.md", "README.md", "@@ -1,1 +1,2 @@\n x\n+y\n", (2,)),
            FileDiff("docs/api.md", "docs/api.md", "@@ -1,1 +1,2 @@\n x\n+y\n", (2,)),
        ),
    )


# Деление по содержимому должно быть ПРОВЕДЕНО до модели, а не просто существовать
# в Config. Тесты config.py проверяют правило; здесь проверяется, что review()
# действительно читает файлы и передаёт читателя в split_by_risk.
#
# Написан после мутации: убрать read_content из вызова в __main__ — и все 282
# теста оставались зелёными. Ровно тот же промах, что был у раннера матчера,
# где схлопывание городов перестало вызываться, а тест звал функцию напрямую.
def test_migration_content_decides_the_model_not_just_the_path(tmp_path):
    (tmp_path / "migrations" / "new").mkdir(parents=True)
    (tmp_path / "migrations/new/1_add.sql").write_text(
        "ALTER TABLE t ADD COLUMN IF NOT EXISTS c text;\n", encoding="utf-8")
    (tmp_path / "migrations/new/2_drop.sql").write_text(
        "ALTER TABLE t DROP COLUMN c;\n", encoding="utf-8")

    config = make_config(model="mini", model_high_risk="strong",
                         high_risk_paths=(), high_risk_content=(r"DROP\s+COLUMN",))
    factory = RecordingReviewerFactory()

    review(config, tmp_path, FakeClient(_migration_mr()), factory, lambda d: None, mr_iid=7)

    assert dict(factory.calls) == {
        "strong": ("migrations/new/2_drop.sql",),
        "mini": ("migrations/new/1_add.sql",),
    }


# Файл из диффа может отсутствовать в дереве — удалён этим же MR. Читать нечего,
# и это не повод считать его безопасным: решает путь, как раньше.
def test_a_deleted_file_falls_back_to_the_path_rule(tmp_path):
    config = make_config(model="mini", model_high_risk="strong",
                         high_risk_paths=("migrations/*",),
                         high_risk_content=(r"DROP\s+COLUMN",))
    factory = RecordingReviewerFactory()

    review(config, tmp_path, FakeClient(_migration_mr()), factory, lambda d: None, mr_iid=7)

    assert dict(factory.calls) == {
        "strong": ("migrations/new/1_add.sql", "migrations/new/2_drop.sql"),
    }
# Проверяется ПРОВОДКА, а не правило: модель не должна быть вызвана ни разу.
# Признак отбора — пустой список вызовов фабрики, и его подмена изменить не может.
def test_a_docs_only_diff_never_reaches_the_model(tmp_path):
    config = make_config(skip_paths=("*.md", "docs/**"))
    factory = RecordingReviewerFactory()

    code = review(config, tmp_path, FakeClient(_docs_mr()), factory, lambda d: None, mr_iid=7)

    assert factory.calls == []
    assert code == 0


def test_one_code_file_brings_the_model_back(tmp_path):
    mr = MergeRequestDiff(
        base_sha="b" * 40, head_sha="h" * 40, start_sha="s" * 40,
        files=(
            FileDiff("README.md", "README.md", "@@ -1,1 +1,2 @@\n x\n+y\n", (2,)),
            FileDiff("a.go", "a.go", "@@ -1,1 +1,2 @@\n x\n+y\n", (2,)),
        ),
    )
    config = make_config(skip_paths=("*.md",))
    factory = RecordingReviewerFactory()

    review(config, tmp_path, FakeClient(mr), factory, lambda d: None, mr_iid=7)

    # Оба файла: пропускается прогон целиком или не пропускается вовсе.
    # Выбрасывать .md из диффа с кодом нельзя — README нередко и есть то место,
    # где видно, что изменение описано неверно.
    assert factory.calls == [("m", ("README.md", "a.go"))]


# --- Read record in the job log -------------------------------------------------
# The thread keeps the record of a published finding. A finding that is dropped
# before publication (outside the diff, low confidence) reaches no thread, so the
# log is its only trace; one line per finding, keyed by fingerprint.
from reviewer.reads import Read  # noqa: E402


def test_every_llm_finding_logs_its_read_record_even_when_dropped(tmp_path, capsys):
    import dataclasses
    kept = dataclasses.replace(finding(), reads=(Read("read_file", "a.go", 1, 3),))
    low = dataclasses.replace(finding(), file="b.go", confidence="low", reads=())
    review(make_config(), tmp_path, FakeClient(make_mr()), lambda model: FakeReviewer([kept, low]),
           lambda d: None, mr_iid=7)
    out = capsys.readouterr().out
    assert f"AI review: чтения находки fp={kept.fingerprint} [bug] a.go:2 — v=1;read=a.go:1-3;grep=;list=" in out
    assert f"AI review: чтения находки fp={low.fingerprint} [bug] b.go:2 — v=1;read=;grep=;list=" in out


def test_finding_without_a_record_logs_no_read_line(tmp_path, capsys):
    review(make_config(), tmp_path, FakeClient(make_mr()), lambda model: FakeReviewer([finding()]),
           lambda d: None, mr_iid=7)
    assert "чтения находки" not in capsys.readouterr().out


def test_log_keeps_the_full_record_even_when_the_thread_copy_is_bounded(tmp_path, capsys):
    import dataclasses
    reads = tuple(Read("grep", f"pkg{i:04d}/file_with_a_long_name_{i:04d}.go", 7, 7) for i in range(3000))
    big = dataclasses.replace(finding(), reads=reads)
    review(make_config(), tmp_path, FakeClient(make_mr()), lambda model: FakeReviewer([big]),
           lambda d: None, mr_iid=7)
    out = capsys.readouterr().out
    assert "pkg2999/file_with_a_long_name_2999.go:7" in out
    assert "grep_dropped" not in out


# --- Generated files: not counted against the limit, and never sent to the model -
# Measured on jobbytime-matcher !133, 2026-09-21: a merge request of a builder,
# its tests and the dictionary that builder produces was refused outright --
# "изменено 8924 строк, лимит 3000", exit 3 -- so nothing was reviewed at all,
# code included. 8000 of those lines were the generated file. skip_paths could
# not help: check_limits runs before it, and changed_line_count counts every
# file in the diff whether or not anyone will read it.
def _generated_file(path, added):
    body = "@@ -0,0 +1,%d @@\n" % added + "".join("+x\n" for _ in range(added))
    return FileDiff(path, path, body, tuple(range(1, added + 1)))


def _generated_and_code_mr():
    return MergeRequestDiff(
        base_sha="b" * 40, head_sha="h" * 40, start_sha="s" * 40,
        files=(
            _generated_file("docs/shared/roles.json", 5000),
            FileDiff("scripts/build.py", "scripts/build.py",
                     "@@ -1,1 +1,2 @@\n x\n+y\n", (2,)),
        ),
    )


def test_a_generated_file_does_not_count_against_the_size_limit(tmp_path):
    config = make_config(generated_paths=("docs/shared/roles.json",),
                         limits=Limits(max_changed_lines=100))
    factory = RecordingReviewerFactory()

    code = review(config, tmp_path, FakeClient(_generated_and_code_mr()), factory,
                  lambda d: None, mr_iid=7)

    assert code != EXIT_TOO_LARGE
    assert factory.calls == [("m", ("scripts/build.py",))]


def test_a_generated_file_is_named_in_the_log_rather_than_dropped_quietly(tmp_path, capsys):
    config = make_config(generated_paths=("docs/shared/roles.json",))
    review(config, tmp_path, FakeClient(_generated_and_code_mr()),
           RecordingReviewerFactory(), lambda d: None, mr_iid=7)

    assert "docs/shared/roles.json" in capsys.readouterr().err


def test_a_diff_of_generated_files_only_needs_no_model(tmp_path):
    mr = MergeRequestDiff(
        base_sha="b" * 40, head_sha="h" * 40, start_sha="s" * 40,
        files=(_generated_file("docs/shared/roles.json", 5000),),
    )
    config = make_config(generated_paths=("docs/shared/roles.json",),
                         limits=Limits(max_changed_lines=100))
    factory = RecordingReviewerFactory()

    code = review(config, tmp_path, FakeClient(mr), factory, lambda d: None, mr_iid=7)

    assert code == EXIT_CLEAN
    assert factory.calls == []


def test_an_empty_diff_is_not_mistaken_for_a_generated_only_one(tmp_path, capsys):
    """An empty diff is a strange state, not a merge request of generated files.

    `is_review_pointless` guards the same thing for skip_paths and says why:
    exiting zero on it would hide the strangeness.
    """
    mr = MergeRequestDiff(base_sha="b" * 40, head_sha="h" * 40, start_sha="s" * 40,
                          files=())
    config = make_config(generated_paths=("docs/shared/roles.json",))

    review(config, tmp_path, FakeClient(mr), RecordingReviewerFactory(),
           lambda d: None, mr_iid=7)

    assert "не ревьюируются" not in capsys.readouterr().err


def test_a_generated_file_alongside_a_docs_only_diff_still_skips_the_run(tmp_path):
    """The two lists do not get in each other's way.

    A diff of one .md and one generated file is still "nothing here needs a
    review": dropping the generated file leaves only the .md, and skip_paths
    cancels the run as it did before.
    """
    mr = MergeRequestDiff(
        base_sha="b" * 40, head_sha="h" * 40, start_sha="s" * 40,
        files=(
            FileDiff("README.md", "README.md", "@@ -1,1 +1,2 @@\n x\n+y\n", (2,)),
            _generated_file("docs/shared/roles.json", 5000),
        ),
    )
    config = make_config(skip_paths=("*.md",),
                         generated_paths=("docs/shared/roles.json",),
                         limits=Limits(max_changed_lines=100))
    factory = RecordingReviewerFactory()

    code = review(config, tmp_path, FakeClient(mr), factory, lambda d: None, mr_iid=7)

    assert code == EXIT_CLEAN
    assert factory.calls == []
