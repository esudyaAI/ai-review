"""Measure whether findings' claims rested on what their pass read.

Second step of three: record (reviewer.reads) -> measure (this) -> gate. It reads
bot threads, takes the hidden read record and the paths the finding talks about,
and crosses "were those paths read" with how the thread ended. It decides nothing.

    python -m reviewer.measure_reads --mr 181 --mr 186 [--json]

Needs CI_API_V4_URL, CI_PROJECT_ID and AI_REVIEW_GITLAB_TOKEN, like the reviewer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field

from reviewer.findings import extract_marker, is_bot_note
from reviewer.reads import decode_marker

READS_RE = re.compile(r"<!--\s*ai-review:reads\s([^>]*?)\s*-->")
FOOTER_START = "<sub>Найдено автоматически"

# Extensions of files that appear in the reviewed repositories. A token is a path
# only if it ends in one of these: "err.Error", "т.д.", "1.2" are not paths.
# Known blind spot: extensionless files (Dockerfile, Makefile) are not recognised.
_EXTENSIONS = (
    "go|py|yml|yaml|json|toml|md|ts|tsx|js|jsx|mjs|cjs|vue|css|scss|html|sql|sh|"
    "mod|sum|txt|cfg|ini|proto|tf|lock"
)
_PATH_RE = re.compile(
    rf"(?<![\w/.-])(\.?[\w-]+(?:[./][\w-]+)*\.(?:{_EXTENSIONS}))"
    r"(?::(\d+)(?:[-–](\d+))?)?(?![\w])"
)

ACCEPTED = "Аргумент автора принят:"
REJECTED = "Замечание остаётся:"
AUTO_FIXED = "Замечание исправлено, тред закрыт автоматически."

# Statuses that mean "the claim was not backed by a read of that place".
UNREAD_STATUSES = frozenset({"other_lines", "diff_only", "unread"})


@dataclass(frozen=True)
class Mention:
    path: str
    start: int | None
    end: int | None

    @property
    def key(self) -> str:
        if self.start is None:
            return self.path
        if self.start == self.end:
            return f"{self.path}:{self.start}"
        return f"{self.path}:{self.start}-{self.end}"


@dataclass
class ThreadRow:
    mr: int | None
    fingerprint: str | None
    file: str | None
    outcome: str
    has_record: bool
    statuses: dict[str, str] = field(default_factory=dict)

    @property
    def unread(self) -> list[str]:
        return [key for key, status in self.statuses.items() if status in UNREAD_STATUSES]


def mentioned_paths(body: str) -> list[Mention]:
    """Paths the finding talks about: title, problem, evidence and suggested fix.

    The fix is included on purpose. front !125 claimed "the CI description does not
    match the jobs" and named `.gitlab-ci.yml` only in the fix - excluding it made
    the very finding this measurement exists for look like it mentioned nothing.
    The footer and markers are excluded: they repeat the file, they claim nothing."""
    claim = body.split(FOOTER_START, 1)[0]
    claim = re.sub(r"<!--.*?-->", "", claim, flags=re.S)
    mentions: list[Mention] = []
    for match in _PATH_RE.finditer(claim):
        start = int(match.group(2)) if match.group(2) else None
        end = int(match.group(3)) if match.group(3) else start
        mention = Mention(match.group(1), start, end)
        if mention not in mentions:
            mentions.append(mention)
    return mentions


def outcome_of(discussion) -> str:
    """How the thread ended. The last bot verdict wins: an argument rejected once
    and accepted after new facts is a refuted finding."""
    outcome = None
    for note in discussion.notes[1:]:
        if not is_bot_note(note.body):
            continue
        if note.body.startswith(ACCEPTED):
            outcome = "refuted"
        elif note.body.startswith(REJECTED):
            outcome = "upheld"
        elif note.body.startswith(AUTO_FIXED):
            # "Stopped reproducing" is not "fixed" when a human argued first:
            # front !125 was refuted with three quotes from .gitlab-ci.yml and then
            # closed by this very message. Counting it as fixed would hide exactly
            # the refuted findings this measurement looks for.
            disputed = any(not is_bot_note(n.body) for n in discussion.notes[1:discussion.notes.index(note)])
            outcome = "vanished_after_dispute" if disputed else "fixed"
    if outcome is not None:
        return outcome
    if discussion.resolved:
        return "resolved_by_human" if discussion.was_ever_disputed else "resolved"
    return "open"


def _resolve(path: str, known: set[str]) -> str | None:
    """The known path a mention refers to: exact, or a unique suffix match
    (the model often writes `sets_test.go` for `ci/internal/sets_test.go`)."""
    if path in known:
        return path
    candidates = [k for k in known if k.endswith("/" + path)]
    return candidates[0] if len(candidates) == 1 else None


def _covers(ranges, start: int, end: int) -> bool:
    return any(lo <= start and end <= hi for lo, hi in ranges)


def _status(mention: Mention, record: dict, changed_files: set[str]) -> str:
    known = set(record["read"]) | set(record["grep"]) | changed_files
    path = _resolve(mention.path, known) or mention.path
    read_ranges = record["read"].get(path)
    grep_ranges = record["grep"].get(path, [])

    if read_ranges is not None and (mention.start is None or _covers(read_ranges, mention.start, mention.end)):
        return "read"
    if grep_ranges and (mention.start is None or _covers(grep_ranges, mention.start, mention.end)):
        return "grep"
    if path in record["grep"] and grep_ranges is None:
        return "grep_no_lines"
    if read_ranges is not None or path in record["grep"]:
        return "other_lines"
    if record["grep_dropped"]:
        # The file may be among the grep hits dropped from the thread copy; the
        # job log has the full record. Not knowing is not "unread".
        return "maybe_dropped"
    if path in changed_files:
        return "diff_only"
    return "unread"


def classify_thread(discussion, changed_files: set[str], mr: int | None = None) -> ThreadRow:
    body = discussion.notes[0].body
    marker = extract_marker(body)
    row = ThreadRow(
        mr=mr,
        fingerprint=marker[0] if marker else None,
        file=marker[1] if marker else None,
        outcome=outcome_of(discussion),
        has_record=False,
    )
    found = READS_RE.search(body)
    if found is None:
        # Threads from before the record existed: no data, never "unread".
        return row
    row.has_record = True
    record = decode_marker(found.group(1))
    for mention in mentioned_paths(body):
        row.statuses[mention.key] = _status(mention, record, changed_files)
    return row


def summarize(rows) -> dict[tuple[str, str], int]:
    counts: Counter = Counter()
    for row in rows:
        if not row.has_record:
            bucket = "no_record"
        elif not row.statuses:
            bucket = "no_mentions"
        elif row.unread:
            bucket = "some_unread"
        else:
            bucket = "all_read"
        counts[(row.outcome, bucket)] += 1
    return dict(counts)


def collect(client, mr_iids) -> list[ThreadRow]:
    rows = []
    for mr_iid in mr_iids:
        changed = {f.new_path for f in client.fetch_diff(mr_iid).files}
        for discussion in client.list_discussions(mr_iid):
            if discussion.fingerprint is None:
                continue
            rows.append(classify_thread(discussion, changed, mr=mr_iid))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m reviewer.measure_reads")
    parser.add_argument("--mr", type=int, action="append", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    import httpx

    from reviewer.gitlab_client import GitLabClient

    try:
        client = GitLabClient(
            api_url=os.environ["CI_API_V4_URL"],
            token=os.environ["AI_REVIEW_GITLAB_TOKEN"],
            project_id=os.environ["CI_PROJECT_ID"],
            client=httpx.Client(timeout=60.0),
        )
    except KeyError as exc:
        print(f"measure_reads: не задана переменная окружения {exc}", file=sys.stderr)
        return 2

    rows = collect(client, args.mr)
    if args.json:
        print(json.dumps([{**asdict(r), "unread": r.unread} for r in rows], ensure_ascii=False, indent=2))
        return 0
    for row in rows:
        detail = ", ".join(f"{k}={v}" for k, v in row.statuses.items()) or "-"
        record = "запись есть" if row.has_record else "записи нет"
        print(f"!{row.mr} fp={row.fingerprint} {row.file} [{row.outcome}] {record}: {detail}")
    print("Сводка (исход, опора на чтение): количество")
    for (outcome, bucket), count in sorted(summarize(rows).items()):
        print(f"  {outcome:18} {bucket:12} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
