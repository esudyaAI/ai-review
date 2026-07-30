"""Thin GitLab REST client: diffs and merge request discussions."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from reviewer.diff import FileDiff, MergeRequestDiff, parse_new_lines
from reviewer.findings import Finding, extract_marker, render_comment

PAGE_SIZE = 100


@dataclass(frozen=True)
class Note:
    id: int
    body: str


@dataclass(frozen=True)
class Discussion:
    id: str
    notes: tuple[Note, ...]
    resolved: bool

    @property
    def _marker(self) -> tuple[str, str] | None:
        if not self.notes:
            return None
        return extract_marker(self.notes[0].body)

    @property
    def fingerprint(self) -> str | None:
        marker = self._marker
        return marker[0] if marker else None

    @property
    def file(self) -> str | None:
        marker = self._marker
        return marker[1] if marker else None

    @property
    def has_human_reply(self) -> bool:
        # Our own notes always carry the marker; anything else came from a person.
        return any(extract_marker(note.body) is None for note in self.notes[1:])


class GitLabClient:
    def __init__(self, api_url: str, token: str, project_id: str, client: httpx.Client) -> None:
        self._api = api_url.rstrip("/")
        self._project = project_id
        self._http = client
        self._headers = {"PRIVATE-TOKEN": token}

    def _url(self, suffix: str) -> str:
        return f"{self._api}/projects/{self._project}/{suffix}"

    def _get(self, suffix: str, params: dict | None = None):
        response = self._http.get(self._url(suffix), headers=self._headers, params=params)
        response.raise_for_status()
        return response.json()

    def fetch_diff(self, mr_iid: int) -> MergeRequestDiff:
        raw_diffs = self._get(f"merge_requests/{mr_iid}/diffs", {"per_page": PAGE_SIZE})
        refs = self._get(f"merge_requests/{mr_iid}")["diff_refs"]
        files = tuple(
            FileDiff(
                old_path=item["old_path"],
                new_path=item["new_path"],
                diff=item["diff"],
                new_lines=parse_new_lines(item["diff"]),
            )
            for item in raw_diffs
        )
        return MergeRequestDiff(
            base_sha=refs["base_sha"],
            head_sha=refs["head_sha"],
            start_sha=refs["start_sha"],
            files=files,
        )

    def list_discussions(self, mr_iid: int) -> list[Discussion]:
        raw = self._get(f"merge_requests/{mr_iid}/discussions", {"per_page": PAGE_SIZE})
        discussions = []
        for item in raw:
            notes = tuple(Note(id=n["id"], body=n.get("body", "")) for n in item.get("notes", []))
            resolved = all(n.get("resolved", False) for n in item.get("notes", [])) if notes else False
            discussions.append(Discussion(id=item["id"], notes=notes, resolved=resolved))
        return discussions

    def create_discussion(self, mr_iid: int, finding: Finding, mr: MergeRequestDiff) -> str:
        payload = {
            "body": render_comment(finding),
            "position": {
                "position_type": "text",
                "base_sha": mr.base_sha,
                "head_sha": mr.head_sha,
                "start_sha": mr.start_sha,
                "new_path": finding.file,
                "old_path": finding.file,
                "new_line": finding.line_start,
            },
        }
        response = self._http.post(
            self._url(f"merge_requests/{mr_iid}/discussions"),
            headers=self._headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()["id"]

    def reply(self, mr_iid: int, discussion_id: str, body: str) -> None:
        response = self._http.post(
            self._url(f"merge_requests/{mr_iid}/discussions/{discussion_id}/notes"),
            headers=self._headers,
            json={"body": body},
        )
        response.raise_for_status()

    def resolve(self, mr_iid: int, discussion_id: str, body: str) -> None:
        self.reply(mr_iid, discussion_id, body)
        response = self._http.put(
            self._url(f"merge_requests/{mr_iid}/discussions/{discussion_id}"),
            headers=self._headers,
            json={"resolved": True},
        )
        response.raise_for_status()
