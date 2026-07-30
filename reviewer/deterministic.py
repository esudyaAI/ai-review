"""Deterministic checks that do not go through the model."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

from reviewer.findings import Finding

LINTER_COMMANDS = {
    "golangci-lint": ["golangci-lint", "run", "--out-format", "json"],
    "eslint": ["npx", "--no-install", "eslint", "--format", "json", "."],
    "ruff": ["ruff", "check", "--output-format", "json", "."],
    "shellcheck": ["shellcheck", "--format", "json", "-S", "warning", "-x", "."],
}
PROCESS_TIMEOUT_SECONDS = 300


def parse_gitleaks(report: str, changed_files: Sequence[str]) -> list[Finding]:
    changed = set(changed_files)
    findings: list[Finding] = []
    try:
        items = json.loads(report or "[]") or []
    except json.JSONDecodeError:
        return []
    for item in items:
        path = item.get("File", "")
        if path not in changed:
            continue
        line = int(item.get("StartLine", 1))
        rule = item.get("RuleID", "secret")
        # The secret value itself is never copied into the finding.
        findings.append(
            Finding(
                category="security",
                file=path,
                line_start=line,
                line_end=int(item.get("EndLine", line)),
                title=f"Секрет в коде: {item.get('Description', rule)}",
                problem=(
                    "gitleaks нашёл в коде значение, похожее на секрет. "
                    "Секреты передаются только через переменные окружения."
                ),
                evidence=f"правило gitleaks {rule}, строка {line}",
                suggested_fix="Убрать значение из кода, перенести в переменные окружения и отозвать скомпрометированный ключ",
                confidence="high",
                source="gitleaks",
            )
        )
    return findings


def parse_golangci(report: str, changed_files: Sequence[str]) -> list[Finding]:
    changed = set(changed_files)
    try:
        payload = json.loads(report or "{}") or {}
    except json.JSONDecodeError:
        return []
    findings: list[Finding] = []
    for issue in payload.get("Issues") or []:
        position = issue.get("Pos") or {}
        path = position.get("Filename", "")
        if path not in changed:
            continue
        line = int(position.get("Line", 1))
        linter = issue.get("FromLinter", "linter")
        text = issue.get("Text", "")
        findings.append(
            Finding(
                category="bug",
                file=path,
                line_start=line,
                line_end=line,
                title=f"Линтер: {text}",
                problem=text,
                evidence=f"{linter}, строка {line}",
                suggested_fix="Исправить замечание линтера",
                confidence="high",
                source="linter",
            )
        )
    return findings


def run_deterministic(repo: Path, linter: str, changed_files: Sequence[str]) -> list[Finding]:
    findings = parse_gitleaks(_run(["gitleaks", "detect", "--no-git", "--report-format", "json",
                                    "--report-path", "/dev/stdout", "--exit-code", "0"], repo),
                              changed_files)
    if linter == "golangci-lint":
        findings.extend(parse_golangci(_run(LINTER_COMMANDS[linter], repo), changed_files))
    return findings


def _run(command: list[str], repo: Path) -> str:
    try:
        completed = subprocess.run(
            command, cwd=repo, capture_output=True, text=True,
            timeout=PROCESS_TIMEOUT_SECONDS, check=False,
        )
    except FileNotFoundError:
        return ""
    return completed.stdout
