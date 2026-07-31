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

    to_create = tuple(f for fp, f in found_by_fp.items() if fp not in open_by_fp)
    to_resolve: list[str] = []
    kept_open: list[str] = []
    to_adjudicate: list[str] = []

    for fp, discussion in open_by_fp.items():
        if fp in found_by_fp:
            if discussion.has_human_reply:
                to_adjudicate.append(discussion.id)
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
    )
