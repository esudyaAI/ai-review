"""Deterministic decision layer: what to post, what to resolve, whether to block."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import NamedTuple, Sequence

from reviewer.findings import Finding
from reviewer.gitlab_client import Discussion
from reviewer.testlayout import can_hold_a_test, covers, is_test_file


class _Claim(NamedTuple):
    """What an existing thread is arguing about, with whatever coordinates it has."""

    file: str
    category: str | None
    line_bucket: int | None


def _claim_of(discussion: Discussion) -> _Claim | None:
    if discussion.file is None:
        return None
    return _Claim(discussion.file, discussion.category, discussion.line_bucket)


def _argues_about(claim: _Claim, finding: Finding) -> bool:
    """Whether a finding is the claim this thread is already about.

    Both coordinates drift, and each drifts on its own. A push shifts the line
    while the category stays - that is the jobbytime-go !29 case, and matching
    on (category, file) was the answer to it. A relabelling changes the category
    while the line stays: on jobbytime-front !28 an objection answered and closed
    as `rules` came back on the next run as `architecture`, same file, same line,
    same argument, and counted as new. The category is picked by the model, so
    that loop can be restarted at will.

    Requiring both coordinates to match lets either drift restart it. Requiring
    the file alone would swallow every other finding in the file. So: the file,
    plus at least one coordinate that has not moved.

    The cost is named rather than hidden: a genuinely different finding on the
    same ten lines, under a different category, is treated as the same claim.
    Callers report what this suppressed - silence here would read as "nothing
    found".
    """
    if claim.file != finding.file:
        return False
    if claim.category is not None and claim.category == finding.category:
        return True
    return claim.line_bucket is not None and claim.line_bucket == finding.line_bucket


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
    # The findings behind outside_diff, whole. outside_diff says where a
    # dropped finding pointed; this says what it was, which is the only way to
    # tell a true finding lost to an imprecise anchor from noise.
    dropped_outside_diff: tuple[Finding, ...] = ()

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
    # The other half of the rule lives in the prompt (system.md, 1a): the model
    # is told to name a line the merge request changed and to describe a
    # consequence elsewhere in words. Without it the drop costs true findings -
    # it took one of its own on !17, where the code was on 188-189 and the
    # finding was anchored to 185.
    #
    # `added_lines` is None only for callers with no diff at hand - the tests
    # written before this rule. review() always passes the mapping; a filter
    # that dropped everything when it is missing would be worse than no filter.
    outside_diff: tuple[tuple[str, str, int], ...] = ()
    dropped_outside_diff: tuple[Finding, ...] = ()
    dropped_findings: list[Finding] = []
    surviving_findings: list[Finding] = []
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
        dropped_outside_diff = tuple(sorted(
            dropped, key=lambda f: (f.category, f.file, f.line_start, f.title),
        ))
        # A thread already open on a dropped finding is left exactly as it is:
        # resolving it would claim a fix that never happened, and keeping it in
        # kept_open would block the merge on the very finding just dropped.
        #
        # Only when nothing that survived the drop shares its (category, file).
        # A surviving finding whose thread is already open is folded into that
        # thread rather than re-created, so it blocks through kept_open alone -
        # skipping the thread would take the surviving finding out of the gate
        # entirely and turn it green with a valid open finding on the merge
        # request. Caught by the reviewer itself on !17.
        # Kept as findings, not as (category, file) pairs: the thread they have to
        # shield is matched by _argues_about like everywhere else. Remembering a
        # scope here while matching claims elsewhere was a real hole - a dropped
        # finding that came back relabelled failed to shield its own thread, and
        # the thread was resolved as if the finding had been fixed. Found by the
        # reviewer on !22, against the commit that introduced claim matching.
        dropped_findings = dropped
        surviving_findings = kept
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
    accepted_claims = [
        claim for claim in (
            _claim_of(d) for d in discussions if d.resolved and d.was_ever_disputed
        )
        if claim is not None
    ]

    # The same drift also splits a finding away from the thread already arguing
    # about it. Match by scope as well as by exact fingerprint, so a shifted
    # finding lands back in its own thread instead of minting a duplicate next
    # to it, and the argument history stays in one place.
    open_claims = [claim for claim in (_claim_of(d) for d in ours) if claim is not None]

    def argued_anywhere(claims: Sequence[_Claim], f: Finding) -> bool:
        return any(_argues_about(claim, f) for claim in claims)

    not_yet_known = [f for fp, f in found_by_fp.items() if fp not in known_fps]
    to_create_all = [f for f in not_yet_known if not argued_anywhere(open_claims, f)]

    # Scope matching cannot tell "the same finding, shifted" from "a different
    # finding in the same file" - the fingerprint is all we have, and it moved.
    # The second one is folded into the open thread rather than getting its own,
    # so it has to be reported: a finding nobody ever sees is worse than a
    # duplicate thread.
    folded = tuple(sorted(
        {(f.category, f.file) for f in not_yet_known if argued_anywhere(open_claims, f)}
    ))
    to_create = tuple(f for f in to_create_all if not argued_anywhere(accepted_claims, f))
    suppressed = tuple(sorted(
        {(f.category, f.file) for f in to_create_all if argued_anywhere(accepted_claims, f)}
    ))
    to_resolve: list[str] = []
    kept_open: list[str] = []
    to_adjudicate: list[str] = []

    for fp, discussion in open_by_fp.items():
        claim = _claim_of(discussion)
        if claim is not None and any(_argues_about(claim, f) for f in dropped_findings) \
                and not any(_argues_about(claim, f) for f in surviving_findings):
            continue
        # Exact fingerprint, or the same claim under a fingerprint that drifted -
        # both mean the finding still reproduces.
        reproduces = fp in found_by_fp or (
            claim is not None and any(_argues_about(claim, f) for f in findings)
        )

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
        dropped_outside_diff=dropped_outside_diff,
    )
