from reviewer.findings import Finding, render_comment
from reviewer.gate import decide
from reviewer.gitlab_client import Discussion, Note


def finding(file="a.go", title="Ошибка не проверяется", category="bug") -> Finding:
    return Finding(
        category=category, file=file, line_start=1, line_end=1, title=title,
        problem="p", evidence="e", suggested_fix="s", confidence="high",
    )


def discussion_for(f: Finding, resolved=False, human_reply=False) -> Discussion:
    notes = [Note(id=1, body=render_comment(f))]
    if human_reply:
        notes.append(Note(id=2, body="Это не баг: ошибка проверяется выше по стеку"))
    return Discussion(id=f"d-{f.fingerprint}", notes=tuple(notes), resolved=resolved)


def test_new_finding_is_created_and_blocks():
    result = decide([finding()], [], ["a.go"])
    assert [f.title for f in result.to_create] == ["Ошибка не проверяется"]
    assert result.blocking_count == 1
    assert result.exit_code == 1


def test_known_finding_is_not_duplicated():
    f = finding()
    result = decide([f], [discussion_for(f)], ["a.go"])
    assert result.to_create == ()
    assert result.blocking_count == 1


def test_disappeared_finding_in_changed_file_is_resolved():
    f = finding()
    result = decide([], [discussion_for(f)], ["a.go"])
    assert result.to_resolve == (f"d-{f.fingerprint}",)
    assert result.blocking_count == 0
    assert result.exit_code == 0


def test_disappeared_finding_in_untouched_file_stays_open():
    f = finding(file="b.go")
    result = decide([], [discussion_for(f)], ["a.go"])
    assert result.to_resolve == ()
    assert result.kept_open == (f"d-{f.fingerprint}",)
    assert result.exit_code == 1


def test_human_reply_goes_to_adjudication():
    f = finding()
    result = decide([f], [discussion_for(f, human_reply=True)], ["a.go"])
    assert result.to_adjudicate == (f"d-{f.fingerprint}",)


def test_already_resolved_thread_does_not_block():
    f = finding()
    result = decide([], [discussion_for(f, resolved=True)], ["a.go"])
    assert result.to_resolve == ()
    assert result.blocking_count == 0


def test_foreign_discussions_are_ignored():
    human_thread = Discussion(id="human", notes=(Note(id=1, body="а давай ещё поле добавим"),), resolved=False)
    result = decide([], [human_thread], ["a.go"])
    assert result.blocking_count == 0
    assert result.exit_code == 0


def test_clean_review_returns_zero():
    assert decide([], [], ["a.go"]).exit_code == 0


def test_human_reply_with_disappeared_finding_in_changed_file_goes_to_adjudication_not_resolve():
    f = finding()
    result = decide([], [discussion_for(f, human_reply=True)], ["a.go"])
    assert result.to_adjudicate == (f"d-{f.fingerprint}",)
    assert result.to_resolve == ()
    assert result.kept_open == (f"d-{f.fingerprint}",)


def test_human_reply_with_disappeared_finding_in_untouched_file_goes_to_adjudication():
    f = finding(file="b.go")
    result = decide([], [discussion_for(f, human_reply=True)], ["a.go"])
    assert result.to_adjudicate == (f"d-{f.fingerprint}",)
    assert result.to_resolve == ()
    assert result.kept_open == (f"d-{f.fingerprint}",)
