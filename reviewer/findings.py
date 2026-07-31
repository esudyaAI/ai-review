"""Finding model, comment rendering and the submit_findings contract."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

CATEGORIES = frozenset({"bug", "security", "tests", "rules", "architecture"})
CONFIDENCES = frozenset({"high", "medium", "low"})
REQUIRED_FIELDS = (
    "category", "file", "line_start", "line_end",
    "title", "problem", "evidence", "suggested_fix", "confidence",
)

MARKER_RE = re.compile(r"<!--\s*ai-review:fp=([0-9a-f]{16});file=(.+?)\s*-->")
BOT_NOTE_MARKER = "<!-- ai-review:bot -->"


class FindingsSchemaError(Exception):
    """Raised when the model returns a payload we cannot trust."""


@dataclass(frozen=True)
class Finding:
    category: str
    file: str
    line_start: int
    line_end: int
    title: str
    problem: str
    evidence: str
    suggested_fix: str
    confidence: str
    source: str = "llm"

    @property
    def fingerprint(self) -> str:
        key = f"{self.category}|{self.file}|{_normalize_title(self.title)}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().lower()


def render_comment(finding: Finding) -> str:
    return (
        f"**[{finding.category}] {finding.title}**\n\n"
        f"{finding.problem}\n\n"
        f"_Основание:_ {finding.evidence}\n\n"
        f"_Предлагаемая правка:_ {finding.suggested_fix}\n\n"
        f"<sub>Найдено автоматически ({finding.source}). "
        f"Если это ложное срабатывание — ответь в этом треде с обоснованием, "
        f"следующий прогон его учтёт.</sub>\n"
        f"<!-- ai-review:fp={finding.fingerprint};file={finding.file} -->"
    )


def extract_marker(body: str) -> tuple[str, str] | None:
    match = MARKER_RE.search(body)
    if match is None:
        return None
    return match.group(1), match.group(2)


def is_bot_note(body: str) -> bool:
    """True if this note was authored by the reviewer itself (original finding or a follow-up)."""
    return extract_marker(body) is not None or BOT_NOTE_MARKER in body


def parse_findings(payload: dict) -> list[Finding]:
    if not isinstance(payload, dict) or "findings" not in payload:
        raise FindingsSchemaError("payload must be an object with a 'findings' key")
    raw_findings = payload["findings"]
    if not isinstance(raw_findings, list):
        raise FindingsSchemaError("'findings' must be a list")

    string_fields = {"title", "problem", "evidence", "suggested_fix", "file", "category", "confidence"}

    findings: list[Finding] = []
    for index, raw in enumerate(raw_findings):
        if not isinstance(raw, dict):
            raise FindingsSchemaError(f"finding #{index} is not an object")
        missing = [field for field in REQUIRED_FIELDS if field not in raw]
        if missing:
            raise FindingsSchemaError(f"finding #{index} misses fields: {missing}")

        # Validate string fields
        for field in string_fields:
            if field in raw and not isinstance(raw[field], str):
                raise FindingsSchemaError(f"finding #{index} field '{field}' must be a string, got {type(raw[field]).__name__}")

        if raw["category"] not in CATEGORIES:
            raise FindingsSchemaError(f"finding #{index} has unknown category {raw['category']}")
        if raw["confidence"] not in CONFIDENCES:
            raise FindingsSchemaError(f"finding #{index} has unknown confidence {raw['confidence']}")
        try:
            line_start = int(raw["line_start"])
            line_end = int(raw["line_end"])
        except (TypeError, ValueError) as exc:
            raise FindingsSchemaError(f"finding #{index} has non-numeric lines") from exc
        findings.append(
            Finding(
                category=raw["category"],
                file=raw["file"],
                line_start=line_start,
                line_end=line_end,
                title=raw["title"],
                problem=raw["problem"],
                evidence=raw["evidence"],
                suggested_fix=raw["suggested_fix"],
                confidence=raw["confidence"],
            )
        )
    return findings


def keep_blocking(findings: Iterable[Finding]) -> list[Finding]:
    # Reporting discipline: only findings the model is sure about reach the merge request.
    return [f for f in findings if f.confidence == "high"]


SUBMIT_FINDINGS_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "submit_findings",
        "description": "Submit the final review verdict. Must be called exactly once.",
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["summary", "findings"],
            "properties": {
                "summary": {"type": "string"},
                "findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(REQUIRED_FIELDS),
                        "properties": {
                            "category": {"type": "string", "enum": sorted(CATEGORIES)},
                            "file": {"type": "string"},
                            "line_start": {"type": "integer"},
                            "line_end": {"type": "integer"},
                            "title": {"type": "string"},
                            "problem": {"type": "string"},
                            "evidence": {"type": "string"},
                            "suggested_fix": {"type": "string"},
                            "confidence": {"type": "string", "enum": sorted(CONFIDENCES)},
                        },
                    },
                },
            },
        },
    },
}
