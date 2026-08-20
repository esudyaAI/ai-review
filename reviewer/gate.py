"""Deterministic decision layer: what to post, what to resolve, whether to block."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Sequence

from reviewer.findings import Finding
from reviewer.gitlab_client import Discussion
from reviewer.testlayout import can_hold_a_test, covers, is_test_file


@dataclass(frozen=True)
class GateDecision:
    to_create: tuple[Finding, ...]
    to_resolve: tuple[str, ...]
    to_adjudicate: tuple[str, ...]
    kept_open: tuple[str, ...]
    suppressed: tuple[tuple[str, str], ...] = ()
    folded: tuple[tuple[str, str], ...] = ()
    not_applicable: tuple[tuple[str, str], ...] = ()
    covered_by_new_tests: tuple[tuple[str, str], ...] = ()
    outside_diff: tuple[tuple[str, str, int], ...] = ()

    @property
    def blocking_count(self) -> int:
        return len(self.to_create) + len(self.kept_open)

    @property
    def exit_code(self) -> int:
        return 1 if self.blocking_count else 0


def decide(
    findings: Sequence[Finding],
    discussions: Sequence[Discussion],
    changed_files: Sequence[str],
    test_globs: Sequence[str] = (),
    added_lines: Mapping[str, Collection[int]] | None = None,
) -> GateDecision:
    # A finding on a line the merge request did not add is not a finding about
    # this merge request. On jobbytime-front !20 one was published carrying its
    # own admission that its line was outside the diff, and counted as open. Its
    # file was in the diff; the line it named was a context line of a hunk. The
    # author can neither fix such a finding (the line is not his) nor argue it
    # away on the merits (it is about somebody else's code), so it can only be
    # closed by hand - the work the two-rounds rule exists to avoid.
    #
    # By line and not by file, because the file-level version does not catch
    # that case at all: the file was changed, just not on the line reported.
    #
    # Deterministic findings are exempt: gitleaks reports the real line of a
    # secret, and a secret on an untouched line of a file this merge request
    # edits is a real finding. deterministic.py already drops the ones whose
    # file is outside the diff.
    #
    # `added_lines` is None only for callers with no diff at hand - the tests
    # written before this rule. review() always passes the mapping; a filter
    # that dropped everything when it is missing would be worse than no filter.
    outside_diff: tuple[tuple[str, str, int], ...] = ()
    dropped_fps: set[str] = set()
    dropped_scopes: set[tuple[str, str]] = set()
    if added_lines is not None:
        kept: list[Finding] = []
        dropped: list[Finding] = []
        for finding in findings:
            inside = (finding.source != "llm"
                      or finding.line_start in added_lines.get(finding.file, ()))
            (kept if inside else dropped).append(finding)
        findings = kept
        outside_diff = tuple(sorted(
            {(f.category, f.file, f.line_start) for f in dropped}
        ))
        # A thread already open on a dropped finding is left exactly as it is:
        # resolving it would claim a fix that never happened, and keeping it in
        # kept_open would block the merge on the very finding just dropped.
        dropped_fps = {f.fingerprint for f in dropped}
        dropped_scopes = {(f.category, f.file) for f in dropped}
    # The `tests` category has to answer "did this merge request make coverage
    # worse", not "is this function covered". Asked the second way it reports gaps
    # the merge request did not create, and on jobbytime-go !33 that produced a
    # finding per touched file across seven review rounds - one of them about a
    # function whose tests were in the same diff.
    #
    # The deterministic half of the answer: if this merge request adds or changes
    # the test that belongs to the file by name, a "no coverage" claim about that
    # file is not a finding about this merge request. Matching is by name, not by
    # directory - a directory match excused an unrelated metrics_test.go for
    # auth.go, and never fired at all in a tests/unit/ layout.
    #
    # It does not cover every case: a package-wide test file (auth_service_test.go
    # covering auth_password.go) is not matched by name, and there is no way to
    # tell from the file list alone whether it covers the finding. That half is a
    # matter of judgement and lives in the prompt.
    changed_tests = [p for p in changed_files if is_test_file(p, test_globs)]
    covered_by_new_tests = tuple(sorted(
        {(f.category, f.file) for f in findings
         if f.category == "tests" and any(covers(f.file, t) for t in changed_tests)}
    ))
    findings = [f for f in findings
                if not (f.category == "tests"
                        and any(covers(f.file, t) for t in changed_tests))]

    # A `tests` finding on a file with nothing to call asks for a unit test of
    # somebody else's configuration format. Three of them landed on one
    # .gitlab-ci.yml in !12. The category cannot apply, so the finding is dropped
    # before it reaches the merge request - and reported, because a silently
    # dropped finding is how real ones get lost.
    not_applicable = tuple(sorted(
        {(f.category, f.file) for f in findings
         if f.category == "tests" and not can_hold_a_test(f.file)}
    ))
    findings = [f for f in findings
                if not (f.category == "tests" and not can_hold_a_test(f.file))]

    ours = [d for d in discussions if d.fingerprint is not None and not d.resolved]
    open_by_fp = {d.fingerprint: d for d in ours}
    found_by_fp = {f.fingerprint: f for f in findings}
    changed = set(changed_files)

    # A finding whose thread was resolved because a human argument was accepted must not
    # be re-created just because a deterministic finding (gitleaks) reproduces on
    # every run - that would spawn a duplicate thread contradicting a decision a human
    # already made. But a thread that was auto-resolved (no human ever replied - it was
    # closed only because the finding stopped reproducing) carries no such decision: if
    # the same finding comes back (revert, bad rebase, regression), it must be treated
    # as new and block again, not silently suppressed by the old fingerprint.
    known_fps = {
        d.fingerprint for d in discussions
        if d.fingerprint is not None and (not d.resolved or d.was_ever_disputed)
    }
    # The fingerprint is positional (category + file + a 10-line window), so any
    # push that shifts the file mints a new one for the same finding. Observed on
    # jobbytime-go MR !29: a thread the author closed with an argument came back
    # under a fresh fingerprint on the next run, and the two-rounds rule cannot
    # break that loop - every round is a "new" finding.
    #
    # So an argument a human actually made and that was ACCEPTED - the thread is
    # closed - is remembered by (category, file), not by line. An argument the
    # reviewer rejected is not an accepted argument: its thread stays open, and
    # suppressing on it would make the finding vanish (the shifted fingerprint
    # suppressed, the old thread auto-resolved because its own fingerprint no
    # longer reproduces in a changed file), so a finding the reviewer stands
    # behind would stop blocking the merge.
    #
    # The cost is a real false negative: a genuinely different finding of the
    # same category in the same file is suppressed too. It is logged for that
    # reason - silence here would read as "nothing found".
    accepted_scopes = {
        (d.category, d.file) for d in discussions
        if d.resolved and d.was_ever_disputed and d.category is not None and d.file is not None
    }

    # The same drift also splits a finding away from the thread already arguing
    # about it. Match by scope as well as by exact fingerprint, so a shifted
    # finding lands back in its own thread instead of minting a duplicate next
    # to it, and the argument history stays in one place.
    open_scopes = {(d.category, d.file) for d in ours if d.category is not None and d.file is not None}
    found_scopes = {(f.category, f.file) for f in findings}

    not_yet_known = [f for fp, f in found_by_fp.items() if fp not in known_fps]
    to_create_all = [f for f in not_yet_known if (f.category, f.file) not in open_scopes]

    # Scope matching cannot tell "the same finding, shifted" from "a different
    # finding in the same file" - the fingerprint is all we have, and it moved.
    # The second one is folded into the open thread rather than getting its own,
    # so it has to be reported: a finding nobody ever sees is worse than a
    # duplicate thread.
    folded = tuple(sorted(
        {(f.category, f.file) for f in not_yet_known if (f.category, f.file) in open_scopes}
    ))
    to_create = tuple(f for f in to_create_all if (f.category, f.file) not in accepted_scopes)
    suppressed = tuple(sorted(
        {(f.category, f.file) for f in to_create_all if (f.category, f.file) in accepted_scopes}
    ))
    to_resolve: list[str] = []
    kept_open: list[str] = []
    to_adjudicate: list[str] = []

    for fp, discussion in open_by_fp.items():
        if fp in dropped_fps or (discussion.category, discussion.file) in dropped_scopes:
            continue
        # Exact fingerprint, or the same (category, file) under a fingerprint the
        # line drift moved - both mean the finding still reproduces.
        reproduces = fp in found_by_fp or (discussion.category, discussion.file) in found_scopes

        # Adjudication is for arguments, and only a reproducing finding can be
        # argued about. Sending a fixed one there is what made every merge request
        # drag: the author writes "confirmed, fixed in commit abc", the adjudicator
        # answers "this does not refute the finding" - true, useless - and the
        # thread stays open forever. A finding that stopped reproducing in a file
        # this merge request changed is fixed, and that is not a judgement call.
        if reproduces:
            if discussion.has_human_reply:
                to_adjudicate.append(discussion.id)
            kept_open.append(discussion.id)
            continue

        if discussion.file in changed:
            # Gone from a file this merge request changed: fixed.
            to_resolve.append(discussion.id)
        elif discussion.has_human_reply:
            # Gone from a file nobody touched. That is not a fix, it is the model
            # being non-deterministic, so a human argument still deserves a verdict.
            to_adjudicate.append(discussion.id)
            kept_open.append(discussion.id)
        else:
            # The model is not deterministic; a silently vanished finding is not proof of a fix.
            kept_open.append(discussion.id)

    return GateDecision(
        to_create=to_create,
        to_resolve=tuple(to_resolve),
        to_adjudicate=tuple(to_adjudicate),
        kept_open=tuple(kept_open),
        suppressed=suppressed,
        folded=folded,
        not_applicable=not_applicable,
        covered_by_new_tests=covered_by_new_tests,
        outside_diff=outside_diff,
    )
