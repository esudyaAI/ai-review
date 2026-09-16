"""What a review pass read, recorded from the tool calls actually served.

The model submits all findings at once, so which read served which finding is
unknowable to the runtime. What is knowable - and what this records - is everything
the pass read before its verdict. That is a fact, not the model's account of itself,
and it is what a later gate needs: a finding asserting something about a file the
pass never read."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote, unquote

MARKER_VERSION = "1"

# The published record exactly as encode_marker writes it and nothing else: fixed
# prefix, `v=` first, then fields made only of characters `quote` can emit plus the
# separators. A marker quoted inside finding text ("`<!-- ai-review:reads ... -->`")
# or an unterminated lookalike does not match, so neither stripping nor parsing
# can reach into the text around it. Found by the reviewer's first run on !28.
MARKER_RE = re.compile(
    r"\n?<!-- ai-review:reads (v=\d+(?:;[a-z_]+=[A-Za-z0-9%/._~,|:-]*)*) -->"
)

# `path:line:text` as produced by RepoTools.grep. The path is matched lazily, so a
# colon in the matched text never leaks into it. A path that itself contains
# `:<digits>:` would be split early - such paths do not occur in the reviewed repos.
_GREP_HIT = re.compile(r"^(.+?):(\d+):")


@dataclass(frozen=True)
class Read:
    """One unit of repository content the model saw.

    For read_file and grep, `path` is a file and start/end are the 1-based lines
    seen. For list_files, `path` is the mask and there are no lines: a listing
    proves a file exists, not what it contains."""

    tool: str
    path: str
    start: int | None
    end: int | None


def reads_from_call(name: str, arguments: dict, result: str) -> list[Read]:
    """Reads implied by one successfully served tool call.

    `result` is the tool's raw return value, before any wrapping. Failed calls must
    not be passed here: a refused or missing file was not read."""
    if name == "read_file":
        if not result:
            return []
        start = max(int(arguments.get("offset", 1)), 1)
        end = start + result.count("\n")
        return [Read("read_file", arguments["path"], start, end)]
    if name == "grep":
        reads = []
        for hit in result.splitlines():
            match = _GREP_HIT.match(hit)
            if match:
                line = int(match.group(2))
                reads.append(Read("grep", match.group(1), line, line))
        return reads
    if name == "list_files":
        return [Read("list_files", arguments.get("path_glob", "**/*"), None, None)]
    return []


def _q(text: str) -> str:
    # Encodes every separator used below and `>`, so the record can neither be
    # misparsed nor close the HTML comment it is published in.
    return quote(text, safe="/")


def _ranges(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _grep_paths(reads) -> list[str]:
    return sorted({read.path for read in reads if read.tool == "grep"})


def _lines_section(reads, tool: str) -> str:
    by_path: dict[str, list[tuple[int, int]]] = {}
    for read in reads:
        if read.tool == tool:
            by_path.setdefault(read.path, []).append((read.start, read.end))
    parts = []
    for path in sorted(by_path):
        spans = ",".join(
            str(s) if s == e else f"{s}-{e}" for s, e in _ranges(by_path[path])
        )
        parts.append(f"{_q(path)}:{spans}")
    return "|".join(parts)


# A thread comment has to stay a comment. read_file and list_files are bounded by
# the loop's iteration limit; grep is not (100 hits per call, in as many files).
MAX_THREAD_MARKER_BYTES = 8000


def encode_marker(reads, max_bytes: int | None = None) -> str:
    """Compact, stable, machine-readable record of a pass's reads.

    With `max_bytes`, only the grep section is ever reduced - first to file names
    without line numbers (`grep_lines=0`), then by dropping files from the end
    (`grep_dropped=N`). read= and list= are never cut: they are what the model
    chose to open, and they are bounded anyway. Without `max_bytes` the record is
    complete; the job log keeps that version."""
    reads = list(reads)
    head = f"v={MARKER_VERSION};read={_lines_section(reads, 'read_file')}"
    masks = "|".join(_q(r.path) for r in reads if r.tool == "list_files")

    full = f"{head};grep={_lines_section(reads, 'grep')};list={masks}"
    if max_bytes is None or len(full.encode()) <= max_bytes:
        return full

    paths = [_q(path) for path in _grep_paths(reads)]
    kept = len(paths)
    while True:
        dropped = len(paths) - kept
        marker = (f"{head};grep={'|'.join(paths[:kept])};list={masks}"
                  f";grep_lines=0;grep_dropped={dropped}")
        if len(marker.encode()) <= max_bytes or kept == 0:
            return marker
        # Shrink proportionally rather than one path at a time: thousands of paths
        # would otherwise mean thousands of re-encodings.
        excess = len(marker.encode()) - max_bytes
        average = max(len(marker.encode()) // max(kept, 1), 1)
        kept = max(kept - max(excess // average, 1), 0)


def decode_marker(marker: str) -> dict:
    """Inverse of encode_marker, for the measurement over accumulated threads.

    A grep file without line numbers decodes to None, not to an empty list: the
    lines were seen but not kept."""
    fields = dict(part.split("=", 1) for part in marker.split(";"))
    decoded: dict = {"read": {}, "grep": {}, "list": []}
    for section in ("read", "grep"):
        for entry in filter(None, fields.get(section, "").split("|")):
            if ":" not in entry:
                decoded[section][unquote(entry)] = None
                continue
            path, spans = entry.rsplit(":", 1)
            ranges = []
            for span in spans.split(","):
                start, _, end = span.partition("-")
                ranges.append((int(start), int(end or start)))
            decoded[section][unquote(path)] = ranges
    decoded["list"] = [unquote(m) for m in filter(None, fields.get("list", "").split("|"))]
    decoded["grep_lines"] = fields.get("grep_lines", "1") != "0"
    decoded["grep_dropped"] = int(fields.get("grep_dropped", "0"))
    return decoded
