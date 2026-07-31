import json
from pathlib import Path

import httpx
import pytest
import respx

from reviewer.diff import FileDiff, MergeRequestDiff
from reviewer.findings import Finding
from reviewer.gitlab_client import Discussion, GitLabClient, Note

API = "https://gitlab.example.com/api/v4"
PROJECT = "42"
FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture()
def client():
    return GitLabClient(api_url=API, token="t0ken", project_id=PROJECT, client=httpx.Client())


def make_finding() -> Finding:
    return Finding(
        category="bug",
        file="internal/billing/config.go",
        line_start=11,
        line_end=11,
        title="Ошибка не проверяется",
        problem="err игнорируется",
        evidence="строка 11",
        suggested_fix="проверить err",
        confidence="high",
    )


@respx.mock
def test_fetch_diff_builds_model(client):
    payload = json.loads((FIXTURES / "mr_diffs.json").read_text(encoding="utf-8"))
    respx.get(f"{API}/projects/{PROJECT}/merge_requests/7/diffs?per_page=100").mock(
        return_value=httpx.Response(200, json=payload["diffs"])
    )
    respx.get(f"{API}/projects/{PROJECT}/merge_requests/7").mock(
        return_value=httpx.Response(200, json={"diff_refs": {
            "base_sha": payload["base_commit_sha"],
            "head_sha": payload["head_commit_sha"],
            "start_sha": payload["start_commit_sha"],
        }})
    )
    mr = client.fetch_diff(7)
    assert mr.head_sha == payload["head_commit_sha"]
    assert mr.changed_files == ("internal/billing/config.go",)
    assert mr.files[0].new_lines == (11, 12)


@respx.mock
def test_token_is_sent_in_header(client):
    route = respx.get(f"{API}/projects/{PROJECT}/merge_requests/7/discussions?per_page=100").mock(
        return_value=httpx.Response(200, json=[])
    )
    client.list_discussions(7)
    assert route.calls.last.request.headers["PRIVATE-TOKEN"] == "t0ken"


@respx.mock
def test_list_discussions_maps_notes(client):
    respx.get(f"{API}/projects/{PROJECT}/merge_requests/7/discussions?per_page=100").mock(
        return_value=httpx.Response(200, json=[
            {
                "id": "abc",
                "notes": [
                    {"id": 1, "body": "тело\n<!-- ai-review:fp=0123456789abcdef;file=a.go -->",
                     "resolved": False},
                    {"id": 2, "body": "это не баг", "resolved": False},
                ],
            }
        ])
    )
    discussions = client.list_discussions(7)
    assert discussions[0].fingerprint == "0123456789abcdef"
    assert discussions[0].file == "a.go"
    assert discussions[0].has_human_reply is True
    assert discussions[0].resolved is False


@respx.mock
def test_list_discussions_follows_link_header_pagination(client):
    page1_url = f"{API}/projects/{PROJECT}/merge_requests/7/discussions?per_page=100"
    page2_url = f"{API}/projects/{PROJECT}/merge_requests/7/discussions?per_page=100&page=2"
    respx.get(page1_url).mock(
        return_value=httpx.Response(
            200,
            json=[{"id": "d1", "notes": []}],
            headers={"Link": f'<{page2_url}>; rel="next"'},
        )
    )
    respx.get(page2_url).mock(
        return_value=httpx.Response(200, json=[{"id": "d2", "notes": []}])
    )
    discussions = client.list_discussions(7)
    assert [d.id for d in discussions] == ["d1", "d2"]


@respx.mock
def test_list_discussions_stops_when_no_link_header(client):
    route = respx.get(f"{API}/projects/{PROJECT}/merge_requests/7/discussions?per_page=100").mock(
        return_value=httpx.Response(200, json=[{"id": "d1", "notes": []}])
    )
    discussions = client.list_discussions(7)
    assert [d.id for d in discussions] == ["d1"]
    assert route.call_count == 1


@respx.mock
def test_create_discussion_positions_the_comment(client):
    payload = json.loads((FIXTURES / "mr_diffs.json").read_text(encoding="utf-8"))
    respx.get(f"{API}/projects/{PROJECT}/merge_requests/7/diffs?per_page=100").mock(
        return_value=httpx.Response(200, json=payload["diffs"])
    )
    respx.get(f"{API}/projects/{PROJECT}/merge_requests/7").mock(
        return_value=httpx.Response(200, json={"diff_refs": {
            "base_sha": payload["base_commit_sha"],
            "head_sha": payload["head_commit_sha"],
            "start_sha": payload["start_commit_sha"],
        }})
    )
    mr = client.fetch_diff(7)
    route = respx.post(f"{API}/projects/{PROJECT}/merge_requests/7/discussions").mock(
        return_value=httpx.Response(201, json={"id": "new-discussion"})
    )
    discussion_id = client.create_discussion(7, make_finding(), mr)
    assert discussion_id == "new-discussion"
    sent = json.loads(route.calls.last.request.content)
    assert sent["position"]["new_line"] == 11
    assert sent["position"]["new_path"] == "internal/billing/config.go"
    assert sent["position"]["head_sha"] == payload["head_commit_sha"]
    assert "ai-review:fp=" in sent["body"]


@respx.mock
def test_create_discussion_falls_back_when_line_not_in_diff(client):
    """A cited line that isn't part of the diff must not crash create_discussion (GitLab
    would reject the position with a 400). We fall back to an unpositioned discussion."""
    payload = json.loads((FIXTURES / "mr_diffs.json").read_text(encoding="utf-8"))
    respx.get(f"{API}/projects/{PROJECT}/merge_requests/7/diffs?per_page=100").mock(
        return_value=httpx.Response(200, json=payload["diffs"])
    )
    respx.get(f"{API}/projects/{PROJECT}/merge_requests/7").mock(
        return_value=httpx.Response(200, json={"diff_refs": {
            "base_sha": payload["base_commit_sha"],
            "head_sha": payload["head_commit_sha"],
            "start_sha": payload["start_commit_sha"],
        }})
    )
    mr = client.fetch_diff(7)
    route = respx.post(f"{API}/projects/{PROJECT}/merge_requests/7/discussions").mock(
        return_value=httpx.Response(201, json={"id": "new-discussion"})
    )
    finding = Finding(
        category="bug",
        file="internal/billing/config.go",
        line_start=999,  # not in mr.files[0].new_lines == (11, 12)
        line_end=999,
        title="Ошибка не проверяется",
        problem="err игнорируется",
        evidence="строка 999",
        suggested_fix="проверить err",
        confidence="high",
    )
    discussion_id = client.create_discussion(7, finding, mr)
    assert discussion_id == "new-discussion"
    sent = json.loads(route.calls.last.request.content)
    assert "position" not in sent
    assert "вне изменённого диффа" in sent["body"]
    assert "ai-review:fp=" in sent["body"]


@respx.mock
def test_create_discussion_uses_diff_old_path_for_renamed_files(client):
    """old_path must come from the file's diff entry, not from finding.file (which is
    always the new path) - otherwise a rename would send a bogus old_path to GitLab."""
    mr = MergeRequestDiff(
        base_sha="b" * 40, head_sha="h" * 40, start_sha="s" * 40,
        files=(
            FileDiff(
                old_path="old/name.go",
                new_path="new/name.go",
                diff="@@ -1,2 +1,3 @@\n x\n+y\n",
                new_lines=(2,),
            ),
        ),
    )
    route = respx.post(f"{API}/projects/{PROJECT}/merge_requests/7/discussions").mock(
        return_value=httpx.Response(201, json={"id": "new-discussion"})
    )
    finding = Finding(
        category="bug", file="new/name.go", line_start=2, line_end=2,
        title="t", problem="p", evidence="e", suggested_fix="s", confidence="high",
    )
    client.create_discussion(7, finding, mr)
    sent = json.loads(route.calls.last.request.content)
    assert sent["position"]["old_path"] == "old/name.go"
    assert sent["position"]["new_path"] == "new/name.go"


@respx.mock
def test_resolve_posts_reply_then_marks_resolved(client):
    reply = respx.post(f"{API}/projects/{PROJECT}/merge_requests/7/discussions/abc/notes").mock(
        return_value=httpx.Response(201, json={"id": 9})
    )
    resolve = respx.put(f"{API}/projects/{PROJECT}/merge_requests/7/discussions/abc").mock(
        return_value=httpx.Response(200, json={"id": "abc"})
    )
    client.resolve(7, "abc", "исправлено в abc1234")
    assert reply.called
    assert json.loads(resolve.calls.last.request.content) == {"resolved": True}


@respx.mock
def test_reply_adds_bot_marker_to_body(client):
    from reviewer.findings import BOT_NOTE_MARKER
    route = respx.post(f"{API}/projects/{PROJECT}/merge_requests/7/discussions/abc/notes").mock(
        return_value=httpx.Response(201, json={"id": 9})
    )
    client.reply(7, "abc", "исправлено в abc1234")
    sent = json.loads(route.calls.last.request.content)
    assert BOT_NOTE_MARKER in sent["body"]
    assert "исправлено в abc1234" in sent["body"]


@respx.mock
def test_has_human_reply_excludes_bot_followup_with_marker(client):
    respx.get(f"{API}/projects/{PROJECT}/merge_requests/7/discussions?per_page=100").mock(
        return_value=httpx.Response(200, json=[
            {
                "id": "d1",
                "notes": [
                    {"id": 1, "body": "**[bug] Ошибка**\n\n<!-- ai-review:fp=0123456789abcdef;file=a.go -->",
                     "resolved": False},
                    {"id": 2, "body": "Bot follow-up\n\n<!-- ai-review:bot -->", "resolved": False},
                    {"id": 3, "body": "Human reply: это баг", "resolved": False},
                ],
            }
        ])
    )
    discussions = client.list_discussions(7)
    assert discussions[0].has_human_reply is True


@respx.mock
def test_api_error_raises(client):
    respx.get(f"{API}/projects/{PROJECT}/merge_requests/7/discussions?per_page=100").mock(
        return_value=httpx.Response(500, json={"message": "boom"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.list_discussions(7)
