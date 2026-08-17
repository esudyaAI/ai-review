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


def test_human_reply_with_disappeared_finding_in_changed_file_goes_to_adjudication_not_resolve():
    f = finding()
    result = decide([], [discussion_for(f, human_reply=True)], ["a.go"])
    assert result.to_adjudicate == (f"d-{f.fingerprint}",)
    assert result.to_resolve == ()
    assert result.kept_open == (f"d-{f.fingerprint}",)


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
