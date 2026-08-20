"""Read-only tools exposed to the model, locked inside the checkout."""

from __future__ import annotations

import re
from fnmatch import fnmatch
from pathlib import Path

DENIED_GLOBS = (
    ".env", ".env.*", "*.pem", "*.key", "id_rsa*", "id_ed25519*",
    "*.p12", "*.pfx", ".git/**",
)
MAX_MATCHES = 100


class ToolError(Exception):
    """Raised when a tool call is refused or cannot be served."""


class RepoTools:
    def __init__(self, root: Path, denied_globs: tuple[str, ...] = DENIED_GLOBS) -> None:
        self.root = root.resolve()
        self.denied_globs = denied_globs

    def _resolve(self, path: str) -> Path:
        if path.startswith("/"):
            raise ToolError("absolute paths are not allowed")
        candidate = (self.root / path).resolve()
        if not candidate.is_relative_to(self.root):
            raise ToolError("path escapes the repository checkout")
        if self._is_denied(candidate):
            raise ToolError("this file is not readable by the reviewer")
        return candidate

    def _is_denied(self, candidate: Path) -> bool:
        relative = candidate.relative_to(self.root).as_posix()
        return any(
            fnmatch(relative, pattern) or fnmatch(candidate.name, pattern)
            for pattern in self.denied_globs
        )

    def read_file(self, path: str, offset: int = 1, limit: int = 200) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise ToolError(f"no such file: {path}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(offset - 1, 0)
        return "\n".join(lines[start : start + limit])

    @staticmethod
    def _make_recursive(path_glob: str) -> str:
        """Turn a bare file mask into one that reaches the whole tree.

        Path.glob only looks at the repository root unless the pattern says
        otherwise, so `*_test.go` matches nothing in a project whose tests live
        in packages. That is not a hypothetical: the reviewer reported "no tests
        exist" on jobbytime-go merge requests where every test was present, and
        the evidence line it gave was a grep over `*_test.go`. Every test in that
        repository is below the root, so the mask could not reach a single one.

        A pattern that names a directory is left alone - `internal/*.go` is an
        author saying where to look. A bare mask is a question about the whole
        repository, which is how every other search tool reads it.
        """
        if "/" in path_glob:
            return path_glob
        return f"**/{path_glob}"

    def _candidates(self, path_glob: str) -> list[Path]:
        if path_glob.startswith("/"):
            raise ToolError("absolute glob patterns are not allowed")
        path_glob = self._make_recursive(path_glob)
        try:
            matches = sorted(self.root.glob(path_glob))
        except ValueError as exc:
            raise ToolError(f"invalid glob pattern: {exc}") from exc
        candidates = []
        for p in matches:
            try:
                resolved = p.resolve()
            except OSError:
                continue
            if not resolved.is_relative_to(self.root):
                continue
            if not resolved.is_file() or self._is_denied(resolved):
                continue
            candidates.append(p)
        return candidates

    def grep(self, pattern: str, path_glob: str = "**/*") -> list[str]:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from exc
        hits: list[str] = []
        for candidate in self._candidates(path_glob):
            try:
                text = candidate.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    relative = candidate.relative_to(self.root).as_posix()
                    hits.append(f"{relative}:{number}:{line}")
                    if len(hits) >= MAX_MATCHES:
                        return hits
        return hits

    def list_files(self, path_glob: str = "**/*") -> list[str]:
        return [p.relative_to(self.root).as_posix() for p in self._candidates(path_glob)]

    def dispatch(self, name: str, arguments: dict) -> str:
        if name == "read_file":
            return self.read_file(
                arguments["path"],
                int(arguments.get("offset", 1)),
                int(arguments.get("limit", 200)),
            )
        if name == "grep":
            return "\n".join(self.grep(arguments["pattern"], arguments.get("path_glob", "**/*")))
        if name == "list_files":
            return "\n".join(self.list_files(arguments.get("path_glob", "**/*")))
        raise ToolError(f"unknown tool: {name}")


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the repository checkout.",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search the repository with a regular expression. "
                "path_glob matches the whole tree: a bare mask such as "
                "'*_test.go' searches every directory, not only the root. "
                "Name a directory - 'internal/*.go' - to narrow it."
            ),
            "parameters": {
                "type": "object",
                "required": ["pattern"],
                "additionalProperties": False,
                "properties": {
                    "pattern": {"type": "string"},
                    "path_glob": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "List repository files matching a glob. path_glob matches the "
                "whole tree: a bare mask such as '*_test.go' lists every "
                "directory, not only the root."
            ),
            "parameters": {
                "type": "object",
                "required": [],
                "additionalProperties": False,
                "properties": {"path_glob": {"type": "string"}},
            },
        },
    },
]
