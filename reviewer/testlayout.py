"""Where a repository keeps its tests, and which files cannot hold one.

Both answers used to be hardcoded to Go. The prompt told the model to grep new
functions against `*_test.go`; in jobbytime-ai-review, a Python repository with
14 test files, that grep found nothing and the model concluded there was no
coverage - three such findings landed on a single .gitlab-ci.yml in !12. "grep
found nothing" has to mean "there are no tests", not "you grepped for the wrong
language".
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable

# Mask -> the way it is recognised. Kept as a table rather than one regex so
# adding an ecosystem is a line, and so the mask handed to the model is exactly
# the one that matched something here.
TEST_GLOBS: tuple[tuple[str, str], ...] = (
    ("*_test.go", "_test.go"),
    ("test_*.py", "test_"),
    ("*_test.py", "_test.py"),
    ("*_spec.rb", "_spec.rb"),
    ("*_test.rb", "_test.rb"),
    ("*.test.js", ".test.js"),
    ("*.test.jsx", ".test.jsx"),
    ("*.test.ts", ".test.ts"),
    ("*.test.tsx", ".test.tsx"),
    ("*.spec.js", ".spec.js"),
    ("*.spec.jsx", ".spec.jsx"),
    ("*.spec.ts", ".spec.ts"),
    ("*.spec.tsx", ".spec.tsx"),
    ("*.bats", ".bats"),
)

# Files with nothing to call. A `tests` finding on one of them asks for a unit
# test of somebody else's configuration format, which cannot be written.
NON_CODE_SUFFIXES = frozenset({
    ".yml", ".yaml", ".json", ".toml", ".ini", ".cfg", ".conf", ".properties",
    ".md", ".rst", ".txt", ".csv", ".lock", ".sum", ".mod", ".sql",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".ico",
})

SKIPPED_DIRS = frozenset({".git", "node_modules", "vendor", "__pycache__", ".venv", "dist", "build"})

# A repository walk is bounded: the answer is "which conventions are present",
# and that is settled long before the file count matters.
MAX_FILES_SCANNED = 20_000


def _matches(name: str, marker: str) -> bool:
    return name.startswith(marker) if marker.endswith("_") else name.endswith(marker)


def detect_test_globs(root: Path) -> tuple[str, ...]:
    """Masks, sorted, that actually match a file in this repository."""
    found: set[str] = set()
    scanned = 0
    for path in root.rglob("*"):
        if scanned >= MAX_FILES_SCANNED:
            break
        if any(part in SKIPPED_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        scanned += 1
        name = path.name
        for glob, marker in TEST_GLOBS:
            if glob not in found and _matches(name, marker):
                found.add(glob)
    return tuple(sorted(found))


def describe_test_layout(root: Path) -> str:
    """One line for the prompt naming the masks to grep, or saying there are none."""
    globs = detect_test_globs(root)
    if not globs:
        return (
            "Тестов в этом репозитории не найдено ни по одной известной маске. "
            "Значит отсутствие теста здесь — не новость и не находка сама по себе."
        )
    return (
        "Тесты в этом репозитории лежат по маскам: "
        + ", ".join(globs)
        + ". Ищи покрытие только по ним — маска другого языка ничего не найдёт "
        "и это не будет означать отсутствия теста."
    )


def can_hold_a_test(path: str) -> bool:
    """False for files with nothing to call - config, docs, data, images."""
    suffix = Path(path).suffix.lower()
    if not suffix:
        # Dockerfile, Makefile, LICENSE and friends: build and packaging inputs,
        # not units under test.
        return False
    return suffix not in NON_CODE_SUFFIXES


def _stem(path: str) -> str:
    name = Path(path).name
    # Two suffixes for app.test.tsx, one for auth_test.go - strip until the name
    # stops looking like a test file.
    for _ in range(2):
        name = name.rsplit(".", 1)[0] if "." in name else name
        if not name.endswith((".test", ".spec")):
            break
    return name


def covers(source_path: str, test_path: str) -> bool:
    """True if `test_path` is, by name, the test for `source_path`.

    By name and not by directory: a directory match excused an unrelated
    metrics_test.go for auth.go, and in a Python layout - tests in tests/unit/,
    code in reviewer/ - it never matched at all. Both were caught on !14.

    Known conventions: auth.go/auth_test.go, gate.py/test_gate.py and
    gate_test.py, app.tsx/app.test.tsx and app.spec.tsx, thing.rb/thing_spec.rb.
    """
    if source_path == test_path:
        return False
    source = _stem(source_path)
    if not source:
        return False
    test = _stem(test_path)
    return test in {source, f"{source}_test", f"test_{source}", f"{source}_spec"}


def is_test_file(path: str, globs: Iterable[str]) -> bool:
    """True if `path` matches one of the repository's own test masks.

    Masks, not a hardcoded suffix: a Go mask in a Python repository is what
    started this whole class of mistakes.
    """
    name = Path(path).name
    return any(fnmatch(name, glob) for glob in globs)
