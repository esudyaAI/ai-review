"""Deterministic decision layer: what to post, what to resolve, whether to block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from reviewer.findings import Finding
from reviewer.gitlab_client import Discussion


@dataclass(frozen=True)
class GateDecision:
    to_create: tuple[Finding, ...]
    to_resolve: tuple[str, ...]
    to_adjudicate: tuple[str, ...]
    kept_open: tuple[str, ...]
    suppressed: tuple[tuple[str, str], ...] = ()
    folded: tuple[tuple[str, str], ...] = ()

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
) -> GateDecision:
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
        # Human reply takes priority: always adjudicate, regardless of whether finding reproduces.
        if discussion.has_human_reply:
            to_adjudicate.append(discussion.id)
            kept_open.append(discussion.id)
            continue

        # Exact fingerprint, or the same (category, file) under a fingerprint the
        # line drift moved - both mean the finding still reproduces.
        if fp in found_by_fp or (discussion.category, discussion.file) in found_scopes:
            kept_open.append(discussion.id)
            continue
        if discussion.file in changed:
            to_resolve.append(discussion.id)
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
    )
