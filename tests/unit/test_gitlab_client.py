import json
from pathlib import Path

import httpx
import pytest
import respx

from reviewer.diff import FileDiff, MergeRequestDiff
from reviewer.findings import Finding
from reviewer.gate import decide
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
def test_create_discussion_fallback_round_trips_through_gate(client):
    """The fallback (unpositioned) body must stay dedupable: once GitLab hands the note
    back to us on the next run, Discussion.fingerprint/.file must parse correctly from
    it, and gate.decide() must not create a duplicate thread for the same finding."""
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
    client.create_discussion(7, finding, mr)
    sent_body = json.loads(route.calls.last.request.content)["body"]

    # Simulate GitLab handing this exact note back to us on the next run's
    # list_discussions() call.
    discussion = Discussion(id="new-discussion", notes=(Note(id=1, body=sent_body),), resolved=False)
    assert discussion.fingerprint == finding.fingerprint
    assert discussion.file == finding.file

    decision = decide([finding], [discussion], mr.changed_files)
    assert decision.to_create == ()
    assert decision.blocking_count == 1


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
def test_has_human_reply_is_false_once_we_replied_last(client):
    """Once we reply() to a rejected counter-argument, our bot-marked note becomes the
    last one in the thread. The dispute must not be re-adjudicated (and the same
    rejection re-posted) on every subsequent run until the human writes again."""
    from reviewer.findings import BOT_NOTE_MARKER
    respx.get(f"{API}/projects/{PROJECT}/merge_requests/7/discussions?per_page=100").mock(
        return_value=httpx.Response(200, json=[
            {
                "id": "d1",
                "notes": [
                    {"id": 1, "body": "**[bug] Ошибка**\n\n<!-- ai-review:fp=0123456789abcdef;file=a.go -->",
                     "resolved": False},
                    {"id": 2, "body": "Это не баг: ошибка проверяется выше по стеку", "resolved": False},
                    {"id": 3, "body": f"Замечание остаётся: ...\n\n{BOT_NOTE_MARKER}", "resolved": False},
                ],
            }
        ])
    )
    discussions = client.list_discussions(7)
    assert discussions[0].has_human_reply is False


def test_was_ever_disputed_is_false_with_no_human_reply():
    only_finding = Discussion(
        id="d1",
        notes=(Note(id=1, body="**[bug] x**\n\n<!-- ai-review:fp=0123456789abcdef;file=a.go -->"),),
        resolved=True,
    )
    assert only_finding.was_ever_disputed is False


def test_was_ever_disputed_is_true_with_a_human_reply():
    with_human = Discussion(
        id="d1",
        notes=(
            Note(id=1, body="**[bug] x**\n\n<!-- ai-review:fp=0123456789abcdef;file=a.go -->"),
            Note(id=2, body="Не баг: обрабатывается выше"),
        ),
        resolved=True,
    )
    assert with_human.was_ever_disputed is True


def test_was_ever_disputed_stays_true_after_our_own_reply_unlike_has_human_reply():
    from reviewer.findings import BOT_NOTE_MARKER
    with_bot_followup = Discussion(
        id="d1",
        notes=(
            Note(id=1, body="**[bug] x**\n\n<!-- ai-review:fp=0123456789abcdef;file=a.go -->"),
            Note(id=2, body="Не баг: обрабатывается выше"),
            Note(id=3, body=f"Аргумент автора принят: ...\n\n{BOT_NOTE_MARKER}"),
        ),
        resolved=True,
    )
    # was_ever_disputed looks at the whole thread's history, unlike has_human_reply
    # (which is now False here since our reply is the last note).
    assert with_bot_followup.was_ever_disputed is True
    assert with_bot_followup.has_human_reply is False


@respx.mock
def test_api_error_raises(client):
    respx.get(f"{API}/projects/{PROJECT}/merge_requests/7/discussions?per_page=100").mock(
        return_value=httpx.Response(500, json={"message": "boom"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.list_discussions(7)


# Seen on jobbytime-go MR !29. When the author closes a thread and then edits the
# same file, GitLab appends a system note ("changed this line in version N of the
# diff"). It is not resolvable and its `resolved` is null, so the old
# all(n.get("resolved", False)) check declared a closed thread open - forever, as
# the note never goes away. Every later run re-adjudicated it and kept blocking
# the merge, which made the two-rounds rule impossible to apply.
@respx.mock
def test_resolved_survives_unresolvable_system_note(client):
    payload = [{
        "id": "d1",
        "notes": [
            {"id": 1, "body": "**[tests] что-то**", "resolvable": True, "resolved": True},
            {"id": 2, "body": "ответ автора", "resolvable": True, "resolved": True},
            {"id": 3, "body": "changed this line in version 4 of the diff",
             "resolvable": False, "resolved": None},
        ],
    }]
    respx.get(f"{API}/projects/{PROJECT}/merge_requests/29/discussions?per_page=100").mock(
        return_value=httpx.Response(200, json=payload))
    assert client.list_discussions(29)[0].resolved is True


@respx.mock
def test_open_thread_with_system_note_stays_open(client):
    payload = [{
        "id": "d2",
        "notes": [
            {"id": 1, "body": "**[tests] что-то**", "resolvable": True, "resolved": False},
            {"id": 2, "body": "changed this line in version 4 of the diff",
             "resolvable": False, "resolved": None},
        ],
    }]
    respx.get(f"{API}/projects/{PROJECT}/merge_requests/29/discussions?per_page=100").mock(
        return_value=httpx.Response(200, json=payload))
    assert client.list_discussions(29)[0].resolved is False


# A thread with no resolvable note at all is not closed: there is nothing to close.
@respx.mock
def test_thread_of_only_system_notes_is_not_resolved(client):
    payload = [{
        "id": "d3",
        "notes": [
            {"id": 1, "body": "changed this line in version 4 of the diff",
             "resolvable": False, "resolved": None},
        ],
    }]
    respx.get(f"{API}/projects/{PROJECT}/merge_requests/29/discussions?per_page=100").mock(
        return_value=httpx.Response(200, json=payload))
    assert client.list_discussions(29)[0].resolved is False
