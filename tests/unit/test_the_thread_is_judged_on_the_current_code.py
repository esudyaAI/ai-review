"""A thread is judged on what the reviewer says about the current commit.

jobbytime-front !291, 2026-10-12: the first note said "console is replaced and
never restored". The author restored it. The next run reproduced a DIFFERENT
finding on the new line, matched it to the open thread by scope, kept the
thread open - and the new finding's text went nowhere: not into the thread,
not into the log. The adjudicator, which saw only the first note and the
replies and no code at all, then answered three times "the original finding
was valid" about code that no longer existed.

Three things are held here:
- the gate names the finding that keeps each thread open;
- a finding other than the thread's own is written into the thread, once;
- the adjudicator receives that finding and the code it points at.
"""
from reviewer.__main__ import EXIT_FINDINGS, review
from reviewer.adjudication import adjudicate
from reviewer.findings import Finding, render_comment
from reviewer.gate import decide
from reviewer.gitlab_client import Discussion, Note
from reviewer.llm.base import ProviderResponse, ToolCall

from reviewer.diff import FileDiff, MergeRequestDiff

from test_cli import FakeClient, FakeReviewer, make_config


def make_mr():
    # Both lines are added by this merge request: a finding outside the diff is
    # dropped before the gate (see decide), which is a different story.
    return MergeRequestDiff(
        base_sha="b" * 40, head_sha="h" * 40, start_sha="s" * 40,
        files=(FileDiff("a.go", "a.go", "@@ -1,2 +1,3 @@\n x\n+y\n", (2, 12)),),
    )


def _finding(line, title, problem="p"):
    return Finding(
        category="bug", file="a.go", line_start=line, line_end=line, title=title,
        problem=problem, evidence="e", suggested_fix="s", confidence="high",
    )


ORIGINAL = _finding(2, "Консоль не восстанавливается")
# Same file, same category, another ten-line window: a different fingerprint
# that _argues_about still matches to the thread.
NOW = _finding(12, "afterAll срабатывает слишком поздно", problem="вывод хуков уходит в лог")


def _thread(*replies):
    notes = [Note(id=1, body=render_comment(ORIGINAL))]
    notes += [Note(id=i + 2, body=b) for i, b in enumerate(replies)]
    return Discussion(id="d-orig", notes=tuple(notes), resolved=False)


def test_the_precondition_holds():
    assert NOW.fingerprint != ORIGINAL.fingerprint, "otherwise nothing here is tested"


# --- gate -------------------------------------------------------------------

def test_the_gate_names_the_finding_that_keeps_a_thread_open():
    result = decide([NOW], [_thread()], ["a.go"])
    assert result.kept_open == ("d-orig",)
    assert dict(result.reproduced_by) == {"d-orig": NOW}


def test_a_thread_reproduced_by_its_own_finding_is_named_with_it():
    result = decide([ORIGINAL], [_thread()], ["a.go"])
    assert dict(result.reproduced_by) == {"d-orig": ORIGINAL}


# --- the thread -------------------------------------------------------------

def test_a_different_finding_is_written_into_the_thread_once(tmp_path):
    client = FakeClient(make_mr(), [_thread()])
    code = review(make_config(), tmp_path, client, lambda model: FakeReviewer([NOW]),
                  lambda d, *_: None, mr_iid=7)
    assert code == EXIT_FINDINGS
    assert len(client.replies) == 1
    thread_id, body = client.replies[0]
    assert thread_id == "d-orig"
    assert "afterAll срабатывает слишком поздно" in body
    assert "вывод хуков уходит в лог" in body
    assert f"ai-review:restated fp={NOW.fingerprint}" in body
    assert "<!-- ai-review:bot -->" in body, "a bot note, or it would be adjudicated as the author's"

    # The next run sees its own note and does not repeat it.
    again = FakeClient(make_mr(), [_thread(body)])
    review(make_config(), tmp_path, again, lambda model: FakeReviewer([NOW]),
           lambda d, *_: None, mr_iid=7)
    assert again.replies == []


def test_the_thread_s_own_finding_is_not_restated(tmp_path):
    client = FakeClient(make_mr(), [_thread()])
    review(make_config(), tmp_path, client, lambda model: FakeReviewer([ORIGINAL]),
           lambda d, *_: None, mr_iid=7)
    assert client.replies == []


def test_the_log_says_what_the_other_finding_is(tmp_path, capsys):
    client = FakeClient(make_mr(), [_thread()])
    review(make_config(), tmp_path, client, lambda model: FakeReviewer([NOW]),
           lambda d, *_: None, mr_iid=7)
    out = capsys.readouterr().out
    assert "тред d-orig держит другая находка" in out
    assert "afterAll срабатывает слишком поздно" in out


# --- adjudication -----------------------------------------------------------

def test_the_adjudicator_gets_the_current_finding_and_its_code(tmp_path):
    (tmp_path / "a.go").write_text("".join(f"line {n}\n" for n in range(1, 31)))
    seen = {}

    class Rejecting:
        accepted = False
        reason = "afterAll и правда поздно"

    def adjudicator(discussion, current=None, code=None):
        seen.update(discussion=discussion.id, current=current, code=code)
        return Rejecting()

    client = FakeClient(make_mr(), [_thread("исправлено: восстановление добавлено")])
    review(make_config(), tmp_path, client, lambda model: FakeReviewer([NOW]), adjudicator, mr_iid=7)

    assert seen["current"] == NOW
    assert seen["code"].startswith("a.go:6-18\n")
    assert "   12  line 12" in seen["code"]
    # The rejection carries the finding it is about, since the thread did not have it yet.
    _, body = client.replies[0]
    assert body.startswith("Замечание остаётся: afterAll и правда поздно")
    assert "afterAll срабатывает слишком поздно" in body


class OneShot:
    def __init__(self):
        self.messages = None

    def complete(self, messages, tools):
        self.messages = messages
        return ProviderResponse(
            tool_calls=(ToolCall(id="a", name="adjudicate", arguments={"accepted": False, "reason": "r"}),),
            text=None,
        )


def test_the_model_is_shown_the_current_finding_and_code():
    provider = OneShot()
    adjudicate(provider, _thread("исправлено"), NOW, "a.go:10-14\n   12  afterAll(restore)")
    system, user = provider.messages[0]["content"], provider.messages[1]["content"]
    assert "суди по ним" in system
    assert "<замечание_на_текущем_коммите>" in user and "afterAll срабатывает слишком поздно" in user
    assert "<код_на_текущем_коммите>" in user and "afterAll(restore)" in user


def test_without_a_current_finding_the_conversation_is_as_before():
    provider = OneShot()
    adjudicate(provider, _thread("не баг"))
    user = provider.messages[1]["content"]
    assert "<замечание_на_текущем_коммите>" not in user
    assert "<код_на_текущем_коммите>" not in user


def test_code_outside_the_checkout_is_never_read(tmp_path):
    from reviewer.__main__ import _excerpt

    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "secret.txt").write_text("token\n")
    escaping = _finding(1, "t")
    escaping = Finding(**{**escaping.__dict__, "file": "../secret.txt"})
    assert _excerpt(repo, escaping) is None
