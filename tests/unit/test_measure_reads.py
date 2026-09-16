"""Measurement over accumulated threads: did a finding's claims rest on files its
pass actually read, and how did the thread end.

This is the second step of three (record -> measure -> gate). It decides nothing and
blocks nothing; it turns the hidden read records into a table."""

import re

from reviewer.findings import Finding, render_comment
from reviewer.gitlab_client import Discussion, Note
from reviewer.measure_reads import (
    Mention,
    classify_thread,
    mentioned_paths,
    outcome_of,
    summarize,
)
from reviewer.reads import Read

BOT = "\n\n<!-- ai-review:bot -->"


def finding(evidence="e", problem="p", file="ci/deploy.go", reads=None, line=10):
    return Finding(category="bug", file=file, line_start=line, line_end=line, title="t",
                   problem=problem, evidence=evidence, suggested_fix="s", confidence="high",
                   reads=reads)


def with_record(body, record):
    """Replace the read record of a rendered comment with a hand-written one."""
    return re.sub(r"<!-- ai-review:reads [^>]*-->", f"<!-- ai-review:reads {record} -->", body)


def thread(first_body, *later, resolved=False):
    notes = (Note(1, first_body),) + tuple(Note(i + 2, body) for i, body in enumerate(later))
    return Discussion(id="d1", notes=notes, resolved=resolved)


# --- mentions -------------------------------------------------------------------

def test_mentions_are_paths_with_optional_lines_from_problem_and_evidence():
    body = render_comment(finding(
        problem="Множество в `ci/sets.go` перестало покрывать джобу",
        evidence="в .gitlab-ci.yml:1100 нет scp, см. internal/x_test.go:40-52",
    ))
    assert mentioned_paths(body) == [
        Mention("ci/sets.go", None, None),
        Mention(".gitlab-ci.yml", 1100, 1100),
        Mention("internal/x_test.go", 40, 52),
    ]


def test_markers_and_footer_are_not_mentions():
    body = render_comment(finding(evidence="без путей", problem="без путей",
                                  reads=(Read("read_file", "z.go", 1, 2),)))
    assert mentioned_paths(body) == []


def test_suggested_fix_is_a_mention():
    # front !125: the only path the finding named was in the fix.
    body = render_comment(finding(evidence="без путей", problem="без путей"))
    body = body.replace("_Предлагаемая правка:_ s", "_Предлагаемая правка:_ привести к `.gitlab-ci.yml`")
    assert mentioned_paths(body) == [Mention(".gitlab-ci.yml", None, None)]


def test_words_with_dots_are_not_paths():
    body = render_comment(finding(evidence="err.Error() и т.д., версия 1.2, e.g. fmt.Sprintf"))
    assert mentioned_paths(body) == []


# --- outcome --------------------------------------------------------------------

def test_outcome_refuted_when_the_authors_argument_was_accepted():
    t = thread("x", "не баг", "Аргумент автора принят: верно" + BOT, resolved=True)
    assert outcome_of(t) == "refuted"


def test_outcome_upheld_when_the_argument_was_rejected_and_nothing_later():
    assert outcome_of(thread("x", "не баг", "Замечание остаётся: нет" + BOT)) == "upheld"


def test_later_acceptance_overrides_an_earlier_rejection():
    t = thread("x", "не баг", "Замечание остаётся: нет" + BOT, "вот факт",
               "Аргумент автора принят: да" + BOT, resolved=True)
    assert outcome_of(t) == "refuted"


def test_outcome_fixed_when_closed_because_the_finding_stopped_reproducing():
    t = thread("x", "Замечание исправлено, тред закрыт автоматически." + BOT, resolved=True)
    assert outcome_of(t) == "fixed"


def test_auto_close_after_a_human_argument_is_not_a_fix():
    t = thread("x", "Ложное срабатывание: вот строки 57-62",
               "Замечание исправлено, тред закрыт автоматически." + BOT, resolved=True)
    assert outcome_of(t) == "vanished_after_dispute"


def test_outcome_resolved_by_hand_and_open():
    assert outcome_of(thread("x", "ок", resolved=True)) == "resolved_by_human"
    assert outcome_of(thread("x")) == "open"


# --- classification -------------------------------------------------------------

def test_thread_without_a_record_is_no_data_not_unread():
    row = classify_thread(thread(render_comment(finding(evidence="см. ci/sets.go"))), changed_files=set())
    assert row.has_record is False
    assert row.unread == []


def test_mention_is_read_when_a_read_file_range_covers_its_line():
    body = render_comment(finding(evidence=".gitlab-ci.yml:1100",
                                  reads=(Read("read_file", ".gitlab-ci.yml", 1001, 1200),)))
    row = classify_thread(thread(body), changed_files=set())
    assert row.statuses == {".gitlab-ci.yml:1100": "read"}
    assert row.unread == []


def test_file_read_elsewhere_does_not_cover_a_claim_about_another_line():
    body = render_comment(finding(evidence=".gitlab-ci.yml:1100",
                                  reads=(Read("read_file", ".gitlab-ci.yml", 1, 200),)))
    row = classify_thread(thread(body), changed_files=set())
    assert row.statuses == {".gitlab-ci.yml:1100": "other_lines"}
    assert row.unread == [".gitlab-ci.yml:1100"]


def test_grep_hit_on_the_line_counts_as_seen():
    body = render_comment(finding(evidence="ci/sets.go:40", reads=(Read("grep", "ci/sets.go", 40, 40),)))
    assert classify_thread(thread(body), changed_files=set()).statuses == {"ci/sets.go:40": "grep"}


def test_mention_by_basename_matches_a_unique_read_path():
    body = render_comment(finding(evidence="в sets_test.go нет кейса",
                                  reads=(Read("read_file", "ci/internal/sets_test.go", 1, 90),)))
    assert classify_thread(thread(body), changed_files=set()).statuses == {"sets_test.go": "read"}


def test_changed_file_that_was_not_read_is_marked_as_seen_only_in_the_diff():
    body = render_comment(finding(evidence="docs/ci.md описывает не те джобы", reads=()))
    row = classify_thread(thread(body), changed_files={"docs/ci.md"})
    assert row.statuses == {"docs/ci.md": "diff_only"}
    assert row.unread == ["docs/ci.md"]


def test_unread_file_outside_the_diff_is_unread():
    body = render_comment(finding(evidence="ci/sets.go перестал включать job", reads=()))
    row = classify_thread(thread(body), changed_files={"ci/deploy.go"})
    assert row.statuses == {"ci/sets.go": "unread"}


def test_degraded_grep_without_lines_is_seen_but_not_line_exact():
    body = with_record(render_comment(finding(evidence="ci/sets.go:40", reads=())),
                       "v=1;read=;grep=ci/sets.go;list=;grep_lines=0;grep_dropped=0")
    assert classify_thread(thread(body), changed_files=set()).statuses == {"ci/sets.go:40": "grep_no_lines"}


# --- summary --------------------------------------------------------------------

def test_summary_crosses_outcome_with_whether_claims_rested_on_reads():
    read_ok = render_comment(finding(evidence="a.go:3", reads=(Read("read_file", "a.go", 1, 9),)))
    unread = render_comment(finding(evidence="b.go:3", reads=()))
    no_record = render_comment(finding(evidence="c.go:3"))
    rows = [
        classify_thread(thread(read_ok, "x", "Замечание остаётся: y" + BOT), changed_files=set()),
        classify_thread(thread(unread, "x", "Аргумент автора принят: y" + BOT), changed_files=set()),
        classify_thread(thread(no_record), changed_files=set()),
    ]
    assert summarize(rows) == {
        ("upheld", "all_read"): 1,
        ("refuted", "some_unread"): 1,
        ("open", "no_record"): 1,
    }


def test_finding_without_mentions_is_its_own_bucket():
    body = render_comment(finding(evidence="без путей", reads=()))
    assert summarize([classify_thread(thread(body), changed_files=set())]) == {("open", "no_mentions"): 1}


def test_file_that_may_be_among_dropped_grep_hits_is_not_called_unread():
    body = with_record(render_comment(finding(evidence="ci/sets.go:40", reads=())),
                       "v=1;read=;grep=other.go;list=;grep_lines=0;grep_dropped=5")
    row = classify_thread(thread(body), changed_files=set())
    assert row.fingerprint is not None
    assert row.statuses == {"ci/sets.go:40": "maybe_dropped"}
    assert row.unread == []


def test_run_collects_only_bot_threads_across_merge_requests():
    from reviewer.diff import FileDiff, MergeRequestDiff
    from reviewer.measure_reads import collect

    class Client:
        def fetch_diff(self, mr_iid):
            return MergeRequestDiff("b", "h", "s", (FileDiff("docs/ci.md", "docs/ci.md", "", ()),))

        def list_discussions(self, mr_iid):
            bot = render_comment(finding(evidence="docs/ci.md не соответствует", reads=()))
            return [thread(bot), Discussion(id="human", notes=(Note(9, "просто вопрос"),), resolved=False)]

    rows = collect(Client(), [125, 181])
    assert [(r.mr, r.statuses) for r in rows] == [
        (125, {"docs/ci.md": "diff_only"}),
        (181, {"docs/ci.md": "diff_only"}),
    ]


def test_grep_hit_on_another_line_does_not_cover_the_claimed_line():
    body = render_comment(finding(evidence="ci/sets.go:40", reads=(Read("grep", "ci/sets.go", 12, 12),)))
    row = classify_thread(thread(body), changed_files=set())
    assert row.statuses == {"ci/sets.go:40": "other_lines"}


def test_ambiguous_basename_is_not_guessed():
    # Two files share the name: picking one would call a claim "read" on a coin toss.
    body = render_comment(finding(evidence="sets_test.go:5", reads=(
        Read("read_file", "a/sets_test.go", 1, 90), Read("read_file", "b/sets_test.go", 1, 90))))
    assert classify_thread(thread(body), changed_files=set()).statuses == {"sets_test.go:5": "unread"}


def test_unpositioned_thread_note_after_the_footer_is_not_a_mention():
    # gitlab_client appends "(файл X, строка N вне изменённого диффа ...)" after the
    # footer when the line is outside the diff; that repeats the file, it claims nothing.
    body = render_comment(finding(evidence="без путей", problem="без путей", file="ci/deploy.go"))
    body += "\n\n_(файл ci/deploy.go, строка 10 вне изменённого диффа, замечание не привязано к строке)_"
    assert mentioned_paths(body) == []
