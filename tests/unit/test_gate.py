from reviewer.findings import Finding, render_comment
from reviewer.gate import decide
from reviewer.gitlab_client import Discussion, Note


def finding(file="a.go", title="Ошибка не проверяется", category="bug") -> Finding:
    return Finding(
        category=category, file=file, line_start=1, line_end=1, title=title,
        problem="p", evidence="e", suggested_fix="s", confidence="high",
    )


def discussion_for(f: Finding, resolved=False, human_reply=False) -> Discussion:
    notes = [Note(id=1, body=render_comment(f))]
    if human_reply:
        notes.append(Note(id=2, body="Это не баг: ошибка проверяется выше по стеку"))
    return Discussion(id=f"d-{f.fingerprint}", notes=tuple(notes), resolved=resolved)


def test_new_finding_is_created_and_blocks():
    result = decide([finding()], [], ["a.go"])
    assert [f.title for f in result.to_create] == ["Ошибка не проверяется"]
    assert result.blocking_count == 1
    assert result.exit_code == 1


def test_known_finding_is_not_duplicated():
    f = finding()
    result = decide([f], [discussion_for(f)], ["a.go"])
    assert result.to_create == ()
    assert result.blocking_count == 1


def test_disappeared_finding_in_changed_file_is_resolved():
    f = finding()
    result = decide([], [discussion_for(f)], ["a.go"])
    assert result.to_resolve == (f"d-{f.fingerprint}",)
    assert result.blocking_count == 0
    assert result.exit_code == 0


def test_disappeared_finding_in_untouched_file_stays_open():
    f = finding(file="b.go")
    result = decide([], [discussion_for(f)], ["a.go"])
    assert result.to_resolve == ()
    assert result.kept_open == (f"d-{f.fingerprint}",)
    assert result.exit_code == 1


def test_human_reply_goes_to_adjudication():
    f = finding()
    result = decide([f], [discussion_for(f, human_reply=True)], ["a.go"])
    assert result.to_adjudicate == (f"d-{f.fingerprint}",)


def test_already_resolved_thread_does_not_block():
    f = finding()
    result = decide([], [discussion_for(f, resolved=True)], ["a.go"])
    assert result.to_resolve == ()
    assert result.blocking_count == 0


def test_auto_resolved_thread_with_reproducing_finding_blocks_again():
    # A thread that was auto-resolved (no human ever replied - it was closed only
    # because the finding stopped reproducing on a prior run) carries no human decision
    # to respect. If the same finding comes back (revert, bad rebase, regression), it
    # must be treated as new and block again - NOT silently suppressed because a
    # fingerprint with that value once existed. This is the true-positive regression
    # case rule 5.5 exists to catch.
    f = finding()
    result = decide([f], [discussion_for(f, resolved=True)], ["a.go"])
    assert result.to_create == (f,)
    assert result.blocking_count == 1


def test_disputed_and_resolved_thread_with_reproducing_finding_is_not_duplicated():
    # A deterministic finding (gitleaks) reproduces on every run. If its thread
    # was resolved BECAUSE a human argument was accepted (there was a human reply in the
    # thread at some point), the same finding showing up again must NOT spawn a new
    # duplicate thread - we respect the human's accepted decision going forward.
    f = finding()
    result = decide([f], [discussion_for(f, resolved=True, human_reply=True)], ["a.go"])
    assert result.to_create == ()
    assert result.blocking_count == 0


def test_foreign_discussions_are_ignored():
    human_thread = Discussion(id="human", notes=(Note(id=1, body="а давай ещё поле добавим"),), resolved=False)
    result = decide([], [human_thread], ["a.go"])
    assert result.blocking_count == 0
    assert result.exit_code == 0


def test_clean_review_returns_zero():
    assert decide([], [], ["a.go"]).exit_code == 0


# Retracted 2026-08-17. This used to assert the opposite: a disappeared finding
# with a human reply went to adjudication and was kept open even when its file had
# been changed. The intent was sound - a disappearance is not proof of a fix - but
# in practice it meant an author who fixed a finding and said so could never close
# the thread, because the adjudicator judges arguments and "fixed in commit abc"
# refutes nothing. That is what dragged every merge request on 2026-08-17.
#
# The distinction that keeps the original intent is the file: gone from a file
# this merge request changed is a fix; gone from a file nobody touched is not,
# and that case still goes to adjudication (the test below).
def test_disappeared_finding_in_a_changed_file_is_resolved_not_adjudicated():
    f = finding()
    result = decide([], [discussion_for(f, human_reply=True)], ["a.go"])
    assert result.to_resolve == (f"d-{f.fingerprint}",)
    assert result.to_adjudicate == ()
    assert result.kept_open == ()


def test_human_reply_with_disappeared_finding_in_untouched_file_goes_to_adjudication():
    f = finding(file="b.go")
    result = decide([], [discussion_for(f, human_reply=True)], ["a.go"])
    assert result.to_adjudicate == (f"d-{f.fingerprint}",)
    assert result.to_resolve == ()
    assert result.kept_open == (f"d-{f.fingerprint}",)


def _at_line(f: Finding, line: int) -> Finding:
    return Finding(
        category=f.category, file=f.file, line_start=line, line_end=line, title=f.title,
        problem=f.problem, evidence=f.evidence, suggested_fix=f.suggested_fix,
        confidence=f.confidence,
    )


# Seen on jobbytime-go MR !29: the author closed a finding about an untested
# Redis repository method with an argument, and the next run found it again and
# opened a second thread. The fingerprint is built from a line window
# (line_start // 10) and the push shifted the file, so the accepted argument
# stopped being recognised. The two-rounds rule cannot break that loop - every
# run is formally a new finding.
def test_accepted_argument_survives_line_shift():
    original = finding(file="internal/repository/redis/redis_repository.go", category="tests")
    closed = discussion_for(original, resolved=True, human_reply=True)

    shifted = _at_line(original, 89)
    assert shifted.fingerprint != original.fingerprint, "otherwise the test asserts nothing"

    result = decide([shifted], [closed], ["internal/repository/redis/redis_repository.go"])
    assert result.to_create == ()
    assert result.suppressed == (("tests", "internal/repository/redis/redis_repository.go"),)


def test_accepted_argument_does_not_suppress_other_categories():
    original = finding(file="a.go", category="tests")
    closed = discussion_for(original, resolved=True, human_reply=True)

    other = _at_line(finding(file="a.go", category="bug"), 89)
    result = decide([other], [closed], ["a.go"])
    assert [f.category for f in result.to_create] == ["bug"]


def test_accepted_argument_does_not_suppress_other_files():
    original = finding(file="a.go", category="tests")
    closed = discussion_for(original, resolved=True, human_reply=True)

    other = _at_line(finding(file="b.go", category="tests"), 89)
    result = decide([other], [closed], ["a.go", "b.go"])
    assert [f.file for f in result.to_create] == ["b.go"]


# A thread the reviewer closed itself, because the finding stopped reproducing,
# carries no human decision: a regression that brings it back must block again.
def test_auto_resolved_thread_does_not_suppress_after_shift():
    original = finding(file="a.go", category="tests")
    auto_closed = discussion_for(original, resolved=True, human_reply=False)

    shifted = _at_line(original, 89)
    result = decide([shifted], [auto_closed], ["a.go"])
    assert [f.file for f in result.to_create] == ["a.go"]


# An argument the reviewer REJECTED is not an accepted argument. Suppressing by
# (category, file) on any disputed thread made a rejected finding vanish: the
# shifted fingerprint was suppressed, and the old open thread was auto-resolved
# because its own fingerprint no longer reproduced in a changed file. Net effect
# - a finding the reviewer stood behind stopped blocking the merge.
def test_rejected_dispute_does_not_suppress_the_finding():
    original = finding(file="a.go", category="tests")
    rejected = Discussion(
        id="d-rejected",
        notes=(
            Note(id=1, body=render_comment(original)),
            Note(id=2, body="это не проблема"),
            Note(id=3, body="Замечание остаётся: аргумент не по существу <!-- ai-review:bot -->"),
        ),
        resolved=False,
    )

    shifted = _at_line(original, 89)
    result = decide([shifted], [rejected], ["a.go"])

    # The drifted finding belongs to the thread that is already arguing about it:
    # no new thread, the old one stays open and keeps blocking, and the argument
    # history stays in one place.
    assert result.to_create == ()
    assert result.suppressed == ()
    assert rejected.id in result.kept_open
    assert rejected.id not in result.to_resolve
    assert result.blocking_count >= 1


# The same drift must not spawn a duplicate next to a thread nobody has replied
# to yet: before the scope match, a shifted fingerprint minted a second thread
# about the same defect on every push.
def test_drifted_finding_reuses_the_open_thread():
    original = finding(file="a.go", category="tests")
    open_thread = discussion_for(original)

    shifted = _at_line(original, 89)
    result = decide([shifted], [open_thread], ["a.go"])

    assert result.to_create == ()
    assert open_thread.id in result.kept_open
    assert result.to_resolve == ()


# A finding that really is gone still gets its thread closed - the scope match
# must not turn auto-resolution off.
def test_thread_is_still_resolved_when_nothing_reproduces():
    original = finding(file="a.go", category="tests")
    open_thread = discussion_for(original)

    result = decide([], [open_thread], ["a.go"])

    assert open_thread.id in result.to_resolve


# Scope matching cannot tell "the same finding, shifted" from "a different
# finding of the same category in the same file" - the fingerprint is all we
# have. The second one is therefore folded into the open thread instead of
# getting its own. That is a deliberate cost, but it must not be silent: a
# finding nobody ever sees is worse than a duplicate thread.
def test_different_finding_in_the_same_scope_is_folded_and_reported():
    first = finding(file="a.go", category="bug", title="Ошибка не проверяется")
    open_thread = discussion_for(first)

    second = Finding(
        category="bug", file="a.go", line_start=89, line_end=89,
        title="Индекс за границей среза", problem="другая проблема",
        evidence="другое основание", suggested_fix="другая правка", confidence="high",
    )

    result = decide([second], [open_thread], ["a.go"])

    assert result.to_create == ()
    assert open_thread.id in result.kept_open
    assert result.folded == (("bug", "a.go"),)


def test_exact_match_is_not_reported_as_folded():
    f = finding(file="a.go", category="bug")
    result = decide([f], [discussion_for(f)], ["a.go"])

    assert result.folded == ()


# A .gitlab-ci.yml cannot hold a unit test: there is nothing in it to call. The
# reviewer produced three such findings on one YAML file in !12, each one asking
# for a test of GitLab's own configuration semantics. A category that cannot
# apply must be dropped before it reaches the merge request - and said out loud,
# because a silently dropped finding is how real ones get lost.
def test_tests_findings_on_non_code_files_are_dropped():
    yaml_finding = Finding(
        category="tests", file=".gitlab-ci.yml", line_start=3, line_end=3,
        title="Нет теста на workflow", problem="p", evidence="e",
        suggested_fix="s", confidence="high",
    )
    result = decide([yaml_finding], [], [".gitlab-ci.yml"])

    assert result.to_create == ()
    assert result.blocking_count == 0
    assert result.not_applicable == (("tests", ".gitlab-ci.yml"),)


def test_other_categories_on_non_code_files_are_kept():
    secret = Finding(
        category="security", file=".gitlab-ci.yml", line_start=3, line_end=3,
        title="Секрет в конфиге", problem="p", evidence="e",
        suggested_fix="s", confidence="high",
    )
    result = decide([secret], [], [".gitlab-ci.yml"])

    assert [f.title for f in result.to_create] == ["Секрет в конфиге"]
    assert result.not_applicable == ()


def test_tests_findings_on_code_files_are_kept():
    real = finding(file="internal/service/auth.go", category="tests")
    result = decide([real], [], ["internal/service/auth.go"])

    assert [f.file for f in result.to_create] == ["internal/service/auth.go"]
    assert result.not_applicable == ()


# The `tests` category asked "is this function covered?" instead of "did this
# merge request make coverage worse". On jobbytime-go !33 that produced seven
# findings across seven rounds - one per touched file - including one about
# ProcessSingleApplication whose tests were in the very same diff. A gap the
# merge request did not create is not a finding about the merge request.
def test_tests_finding_is_dropped_when_the_mr_adds_the_matching_test():
    f = finding(file="internal/service/auto_apply_main.go", category="tests")

    result = decide(
        [f], [],
        ["internal/service/auto_apply_main.go", "internal/service/auto_apply_main_test.go"],
        test_globs=("*_test.go",),
    )

    assert result.to_create == ()
    assert result.covered_by_new_tests == (("tests", "internal/service/auto_apply_main.go"),)


# Caught by the reviewer on !14: a directory match excused findings the changed
# test had nothing to do with.
def test_an_unrelated_test_in_the_same_directory_excuses_nothing():
    f = finding(file="internal/service/auth.go", category="tests")

    result = decide(
        [f], [],
        ["internal/service/auth.go", "internal/service/metrics_test.go"],
        test_globs=("*_test.go",),
    )

    assert [x.file for x in result.to_create] == ["internal/service/auth.go"]


# Also caught on !14: in this repository tests live in tests/unit/ and code in
# reviewer/, so a directory match never fired where it was needed most.
def test_python_layout_is_matched_across_directories():
    f = finding(file="reviewer/gate.py", category="tests")

    result = decide(
        [f], [], ["reviewer/gate.py", "tests/unit/test_gate.py"], test_globs=("test_*.py",),
    )

    assert result.to_create == ()
    assert result.covered_by_new_tests == (("tests", "reviewer/gate.py"),)


def test_tests_finding_survives_when_the_mr_adds_no_tests_there():
    f = finding(file="internal/service/auto_apply_main.go", category="tests")

    result = decide([f], [], ["internal/service/auto_apply_main.go"], test_globs=("*_test.go",))

    assert [x.file for x in result.to_create] == ["internal/service/auto_apply_main.go"]
    assert result.covered_by_new_tests == ()


# Tests added in a different package say nothing about this one.
def test_tests_added_elsewhere_do_not_excuse_this_file():
    f = finding(file="internal/service/auto_apply_main.go", category="tests")

    result = decide(
        [f], [],
        ["internal/service/auto_apply_main.go", "internal/stats/pipeline_stats_test.go"],
        test_globs=("*_test.go",),
    )

    assert [x.file for x in result.to_create] == ["internal/service/auto_apply_main.go"]


# Only the `tests` category is excused - a bug is a bug whatever the diff adds.
def test_other_categories_are_not_excused_by_new_tests():
    f = finding(file="internal/service/auto_apply_main.go", category="bug")

    result = decide(
        [f], [],
        ["internal/service/auto_apply_main.go", "internal/service/auto_apply_main_test.go"],
        test_globs=("*_test.go",),
    )

    assert [x.file for x in result.to_create] == ["internal/service/auto_apply_main.go"]


def test_without_known_test_globs_nothing_is_excused():
    f = finding(file="internal/service/auto_apply_main.go", category="tests")

    result = decide(
        [f], [],
        ["internal/service/auto_apply_main.go", "internal/service/auto_apply_main_test.go"],
        test_globs=(),
    )

    assert [x.file for x in result.to_create] == ["internal/service/auto_apply_main.go"]


# The loop that made every merge request today drag on: a thread with a human
# reply went to adjudication and was kept open no matter what, even when the
# finding had stopped reproducing because the author fixed it. The adjudicator
# only understands arguments - asked to judge "confirmed, fixed in commit abc" it
# answers "this does not refute the finding", which is true and useless.
#
# A finding that no longer reproduces in a file this merge request changed is
# fixed. That is not a matter of judgement and must not be sent to one.
def test_fixed_finding_is_resolved_even_after_the_author_replied():
    f = finding(file="a.go")
    fixed = discussion_for(f, human_reply=True)

    result = decide([], [fixed], ["a.go"])

    assert fixed.id in result.to_resolve
    assert fixed.id not in result.to_adjudicate
    assert result.blocking_count == 0


# Still reproducing plus a human argument is exactly what adjudication is for.
def test_reproducing_finding_with_a_reply_is_still_adjudicated():
    f = finding(file="a.go")
    disputed = discussion_for(f, human_reply=True)

    result = decide([f], [disputed], ["a.go"])

    assert disputed.id in result.to_adjudicate
    assert disputed.id in result.kept_open


# A finding that vanished from a file nobody touched proves nothing: the model is
# not deterministic, and silence is not a fix.
def test_vanished_finding_in_an_untouched_file_stays_open():
    f = finding(file="a.go")
    quiet = discussion_for(f, human_reply=True)

    result = decide([], [quiet], ["other.go"])

    assert quiet.id in result.kept_open
    assert quiet.id not in result.to_resolve


# On jobbytime-front !20 a finding arrived carrying its own admission - the note
# said in so many words that CLAUDE.md line 132 was outside the diff and that the
# comment could not be anchored to a line - and was published anyway and counted
# as open. The file was in the diff; line 132 was a context line of the hunk at
# +130,9 (checked against the merge request diff). Such a finding cannot be fixed
# inside the merge request (it did not touch that line) and cannot be argued
# with on the merits (it is about somebody else's code), so it can only be closed
# by hand - exactly the work the two-rounds rule exists to avoid.
def _finding_at(line: int, file="a.go", source="llm", category="bug") -> Finding:
    return Finding(
        category=category, file=file, line_start=line, line_end=line,
        title=f"Находка на {line}", problem="p", evidence="e", suggested_fix="s",
        confidence="high", source=source,
    )


ADDED = {"a.go": frozenset({10, 11, 12})}


def test_finding_on_a_line_the_merge_request_did_not_add_is_dropped():
    result = decide([_finding_at(132)], [], ["a.go"], (), ADDED)
    assert result.to_create == ()
    assert result.blocking_count == 0
    assert result.exit_code == 0
    assert result.outside_diff == (("bug", "a.go", 132),)


# The drop is a judgement about where a finding points, not about whether it is
# right, and the line a model names is the least reliable thing it says: on
# jobbytime-go !323 a [bug] was anchored to `if exists {`, an untouched line
# with added lines directly above and below it, and on jobbytime-front !236 to
# the closing brace one line above an added block. Both were dropped, both
# correctly by the letter of the rule -- and nobody could say what they were,
# because only the category and the line survived. A finding that is dropped
# has to stay readable, or the rule cannot be judged against its own cost.
def test_a_dropped_finding_is_handed_back_whole():
    lost = _finding_at(132)
    result = decide([lost, _finding_at(11)], [], ["a.go"], (), ADDED)
    assert result.dropped_outside_diff == (lost,)


def test_nothing_is_handed_back_when_nothing_is_dropped():
    result = decide([_finding_at(11)], [], ["a.go"], (), ADDED)
    assert result.dropped_outside_diff == ()


def test_finding_on_an_added_line_is_kept():
    result = decide([_finding_at(11)], [], ["a.go"], (), ADDED)
    assert [f.line_start for f in result.to_create] == [11]
    assert result.outside_diff == ()


def test_finding_in_a_file_the_merge_request_never_touched_is_dropped():
    result = decide([_finding_at(11, file="other.go")], [], ["a.go"], (), ADDED)
    assert result.to_create == ()
    assert result.outside_diff == (("bug", "other.go", 11),)


# gitleaks reports the real line of the secret, and a secret sitting on an
# untouched line of a file this merge request edits is a real finding, not an
# artefact of where the model happened to look. Deterministic findings are
# already filtered by file in deterministic.py; the line rule is for the model.
def test_a_deterministic_finding_survives_on_an_untouched_line():
    result = decide([_finding_at(500, source="gitleaks", category="security")], [], ["a.go"], (), ADDED)
    assert [f.line_start for f in result.to_create] == [500]
    assert result.outside_diff == ()


# Without diff line information there is nothing to compare against, and a filter
# that drops everything would be worse than no filter at all. Only tests that
# predate this rule call decide() that way; review() always passes the mapping.
def test_without_line_information_nothing_is_dropped():
    result = decide([_finding_at(132)], [], ["a.go"])
    assert [f.line_start for f in result.to_create] == [132]
    assert result.outside_diff == ()


# The drop must not swallow the history of an argument: a thread already open on
# a finding that is now dropped is left alone, not resolved as "fixed".
def test_a_dropped_finding_does_not_resolve_its_open_thread():
    f = _finding_at(132)
    result = decide([f], [discussion_for(f)], ["a.go"], (), ADDED)
    assert result.to_resolve == ()


# The first version of the drop skipped every open thread sharing a
# (category, file) with a dropped finding, and the reviewer caught it on its own
# !17: a finding that survived the drop is folded into its already-open thread
# instead of being re-created, so it blocks only through kept_open - skipping
# that thread took it out of the gate entirely and turned the merge green with a
# valid open finding on it.
def test_a_surviving_finding_still_blocks_when_another_in_its_file_is_dropped():
    kept = _finding_at(11)
    dropped = _finding_at(132)
    thread = discussion_for(kept)
    result = decide([kept, dropped], [thread], ["a.go"], (), ADDED)
    assert result.kept_open == (thread.id,)
    assert result.blocking_count == 1
    assert result.exit_code == 1
    assert result.outside_diff == (("bug", "a.go", 132),)


def _relabel(f: Finding, category: str) -> Finding:
    return Finding(
        category=category, file=f.file, line_start=f.line_start, line_end=f.line_end,
        title=f.title, problem=f.problem, evidence=f.evidence,
        suggested_fix=f.suggested_fix, confidence=f.confidence,
    )


# Seen on jobbytime-front MR !28: the reviewer objected to an include taken from
# a branch, the author answered in the thread and closed it, and the next run
# raised the same objection again - as "architecture" where it had been "rules".
# Same file, same line, same argument, nothing new said.
#
# The category is picked by the model, and both the fingerprint and the accepted
# scope were built on it, so relabelling alone made an answered claim look new.
# The two-rounds rule cannot converge against a loop the model can restart by
# renaming its own label.
def test_accepted_argument_survives_a_relabelled_category():
    original = finding(file=".gitlab-ci.yml", category="rules")
    closed = discussion_for(original, resolved=True, human_reply=True)

    relabelled = _relabel(original, "architecture")
    assert relabelled.fingerprint != original.fingerprint, "otherwise the test asserts nothing"

    result = decide([relabelled], [closed], [".gitlab-ci.yml"])
    assert result.to_create == ()
    assert result.suppressed == (("architecture", ".gitlab-ci.yml"),)


def test_relabelled_finding_folds_into_the_thread_already_arguing_about_it():
    # Same defect, seen from the other side: while the thread is still open, a
    # relabelled finding used to open a second one next to it, splitting the
    # argument in two and blocking twice for one claim.
    original = finding(file=".gitlab-ci.yml", category="rules")
    open_thread = discussion_for(original)

    result = decide([_relabel(original, "architecture")], [open_thread], [".gitlab-ci.yml"])
    assert result.to_create == ()
    assert result.folded == (("architecture", ".gitlab-ci.yml"),)
    assert result.kept_open == (f"d-{original.fingerprint}",)


def test_a_different_claim_in_the_same_file_is_still_created():
    # The guard on the fix: matching on the line as well as the category must
    # not swallow a finding that is genuinely different. Different category AND
    # different lines is a different claim.
    original = finding(file="a.go", category="tests")
    closed = discussion_for(original, resolved=True, human_reply=True)

    other = _at_line(finding(file="a.go", category="security"), 240)
    result = decide([other], [closed], ["a.go"])
    assert [f.category for f in result.to_create] == ["security"]


# Raised by the reviewer on jobbytime-ai-review !22, against this very change,
# and it reproduces: the outside-diff filter still remembered what it dropped as
# (category, file), while everything else had moved to claim matching. A dropped
# relabelled finding therefore failed to shield its own open thread, and the
# thread was resolved as if the finding had been fixed - the exact silent
# green-out that dropped_scopes exists to prevent.
def test_relabelled_finding_dropped_outside_the_diff_does_not_resolve_its_thread():
    original = _at_line(finding(file="a.py", category="rules"), 132)
    open_thread = discussion_for(original)

    relabelled = _relabel(original, "architecture")
    result = decide(
        [relabelled], [open_thread], ["a.py"],
        added_lines={"a.py": {10}},
    )

    assert result.to_resolve == (), "тред закрыт как исправленный, хотя находка только отброшена"
    assert result.kept_open == ()
