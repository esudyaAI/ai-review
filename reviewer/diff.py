"""Merge request diff model and size limits."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from reviewer.config import Limits

HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


class DiffTooLarge(Exception):
    """Raised when a merge request is too big to review in one pass."""


@dataclass(frozen=True)
class FileDiff:
    old_path: str
    new_path: str
    diff: str
    new_lines: tuple[int, ...]


@dataclass(frozen=True)
class MergeRequestDiff:
    base_sha: str
    head_sha: str
    start_sha: str
    files: tuple[FileDiff, ...]

    @property
    def changed_line_count(self) -> int:
        return sum(len(f.new_lines) for f in self.files)

    @property
    def changed_files(self) -> tuple[str, ...]:
        return tuple(f.new_path for f in self.files)

    @property
    def added_lines_by_file(self) -> dict[str, frozenset[int]]:
        """Per file, the line numbers this merge request adds on the new side.

        A file that only loses lines stays in the mapping with an empty set: it
        is part of the change, and "no added lines" is an answer, not a gap.
        """
        return {f.new_path: frozenset(f.new_lines) for f in self.files}

    def subset(self, paths: Iterable[str]) -> "MergeRequestDiff":
        """The same merge request narrowed to `paths`, shas untouched.

        The shas have to survive: create_discussion positions a comment against
        them, and a finding from a narrowed pass is posted on the same diff.
        """
        wanted = set(paths)
        return MergeRequestDiff(
            base_sha=self.base_sha,
            head_sha=self.head_sha,
            start_sha=self.start_sha,
            files=tuple(f for f in self.files if f.new_path in wanted),
        )


def parse_new_lines(diff_text: str) -> tuple[int, ...]:
    """Return line numbers on the new side that this diff adds."""
    lines: list[int] = []
    current: int | None = None
    for raw in diff_text.splitlines():
        hunk = HUNK_RE.match(raw)
        if hunk:
            current = int(hunk.group(1))
            continue
        # Ignore lines until first hunk is encountered
        if current is None:
            continue
        # Ignore backslash marker lines (e.g., "\ No newline at end of file")
        if raw.startswith("\\"):
            continue
        if raw.startswith("+"):
            lines.append(current)
            current += 1
        elif raw.startswith("-"):
            continue
        else:
            current += 1
    return tuple(lines)


def check_limits(mr: MergeRequestDiff, limits: Limits) -> None:
    if len(mr.files) > limits.max_files:
        raise DiffTooLarge(
            f"в merge request {len(mr.files)} файлов, лимит {limits.max_files}. "
            "Разбей его на несколько."
        )
    if mr.changed_line_count > limits.max_changed_lines:
        raise DiffTooLarge(
            f"изменено {mr.changed_line_count} строк, лимит {limits.max_changed_lines}. "
            "Разбей merge request на несколько."
        )


def render_diff_block(mr: MergeRequestDiff) -> str:
    parts = [f"### {f.new_path}\n```diff\n{f.diff.rstrip()}\n```" for f in mr.files]
    return "\n\n".join(parts)
