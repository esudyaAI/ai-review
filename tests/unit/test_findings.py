import pytest

from reviewer.findings import (
    Finding,
    FindingsSchemaError,
    extract_marker,
    keep_blocking,
    parse_findings,
    render_comment,
)


def make_finding(**overrides) -> Finding:
    data = dict(
        category="bug",
        file="internal/billing/subscription.go",
        line_start=142,
        line_end=147,
        title="Ошибка Stripe API проглатывается",
        problem="Ответ с ошибкой игнорируется, подписка остаётся активной",
        evidence="err присваивается и не проверяется на строке 145",
        suggested_fix="Вернуть ошибку вызывающему коду",
        confidence="high",
    )
    data.update(overrides)
    return Finding(**data)


def test_fingerprint_ignores_title_wording_entirely():
    # The model rephrases the same finding differently across runs - fingerprint
    # must not depend on title at all, or the same defect mints a new fingerprint
    # every run (old thread looks "gone" and gets auto-resolved, a duplicate
    # appears next to it).
    a = make_finding(title="Ошибка Stripe API проглатывается")
    b = make_finding(title="В обработчике платежа Stripe теряется ошибка ответа")
    assert a.fingerprint == b.fingerprint
    assert len(a.fingerprint) == 16


def test_fingerprint_changes_with_file_and_category():
    base = make_finding()
    assert base.fingerprint != make_finding(file="internal/user/user.go").fingerprint
    assert base.fingerprint != make_finding(category="security").fingerprint


def test_fingerprint_is_stable_within_a_ten_line_window():
    base = make_finding(line_start=142)
    assert base.fingerprint == make_finding(line_start=140).fingerprint
    assert base.fingerprint == make_finding(line_start=149).fingerprint


def test_fingerprint_changes_across_a_ten_line_window_boundary():
    base = make_finding(line_start=142)
    assert base.fingerprint != make_finding(line_start=150).fingerprint
    assert base.fingerprint != make_finding(line_start=139).fingerprint


def test_comment_roundtrips_through_marker():
    finding = make_finding()
    body = render_comment(finding)
    assert finding.title in body
    assert finding.suggested_fix in body
    assert extract_marker(body) == (finding.fingerprint, finding.file, finding.category, finding.line_start)


def test_extract_marker_returns_none_for_human_comment():
    assert extract_marker("Это не баг, потому что мы проверяем ошибку выше") is None


def test_parse_findings_accepts_valid_payload():
    payload = {
        "summary": "Одна проблема в биллинге",
        "findings": [
            {
                "category": "bug",
                "file": "internal/billing/subscription.go",
                "line_start": 142,
                "line_end": 147,
                "title": "Ошибка проглатывается",
                "problem": "Подписка остаётся активной",
                "evidence": "строка 145",
                "suggested_fix": "вернуть ошибку",
                "confidence": "high",
            }
        ],
    }
    findings = parse_findings(payload)
    assert len(findings) == 1
    assert findings[0].category == "bug"


@pytest.mark.parametrize(
    "broken",
    [
        {"findings": [{"category": "bug"}]},
        {"findings": [dict(category="wat", file="a.go", line_start=1, line_end=1,
                           title="t", problem="p", evidence="e", suggested_fix="s",
                           confidence="high")]},
        {"findings": "not a list"},
        {},
    ],
)
def test_parse_findings_rejects_broken_payload(broken):
    with pytest.raises(FindingsSchemaError):
        parse_findings(broken)


def test_keep_blocking_drops_low_confidence():
    findings = [
        make_finding(confidence="high"),
        make_finding(confidence="medium", title="Может быть красивее"),
        make_finding(confidence="low", title="Стиль"),
    ]
    kept = keep_blocking(findings)
    assert [f.confidence for f in kept] == ["high"]


@pytest.mark.parametrize(
    "field,invalid_value",
    [
        ("title", None),
        ("title", 123),
        ("problem", None),
        ("problem", 456),
        ("evidence", None),
        ("evidence", []),
        ("suggested_fix", None),
        ("suggested_fix", {"key": "value"}),
        ("file", None),
        ("file", 789),
        ("category", None),
        ("category", 999),
        ("confidence", None),
        ("confidence", 111),
    ],
)
def test_parse_findings_rejects_non_string_fields(field, invalid_value):
    payload = {
        "summary": "Test",
        "findings": [
            {
                "category": "bug",
                "file": "test.go",
                "line_start": 1,
                "line_end": 1,
                "title": "Test",
                "problem": "Test",
                "evidence": "Test",
                "suggested_fix": "Test",
                "confidence": "high",
            }
        ],
    }
    payload["findings"][0][field] = invalid_value
    with pytest.raises(FindingsSchemaError):
        parse_findings(payload)


# Threads created before cat= appeared in the marker must still parse: otherwise
# the whole fingerprint history goes foreign the moment a new version ships.
def test_marker_without_category_is_still_parsed():
    legacy = "<!-- ai-review:fp=0123456789abcdef;file=internal/service/a.go -->"
    assert extract_marker(legacy) == ("0123456789abcdef", "internal/service/a.go", None, None)


def test_marker_carries_the_line_the_finding_was_made_on():
    # The line is what tells a relabelled finding from a genuinely different
    # one: the model can change the category at will, but it cannot move the
    # code it is objecting to. Without the line in the marker, a thread read
    # back from GitLab has no coordinate except the category.
    marker = extract_marker(render_comment(make_finding(line_start=132)))
    assert marker == (make_finding(line_start=132).fingerprint, "internal/billing/subscription.go", "bug", 132)


def test_marker_written_before_the_line_existed_is_still_readable():
    # Threads posted by earlier versions carry no line. They must keep parsing,
    # and say so with None rather than a guess - a guessed line would match
    # something.
    old = "<!-- ai-review:fp=0123456789abcdef;file=a.go;cat=tests -->"
    assert extract_marker(old) == ("0123456789abcdef", "a.go", "tests", None)



# --- The read record in the published comment ------------------------------------
from reviewer.reads import Read  # noqa: E402


def test_llm_finding_comment_carries_a_hidden_read_record():
    finding = make_finding(reads=(Read("read_file", "ci/.gitlab-ci.yml", 1, 200),))
    body = render_comment(finding)
    assert "<!-- ai-review:reads v=1;read=ci/.gitlab-ci.yml:1-200;grep=;list= -->" in body


def test_empty_read_record_is_published_not_omitted():
    body = render_comment(make_finding(reads=()))
    assert "<!-- ai-review:reads v=1;read=;grep=;list= -->" in body


def test_finding_without_a_record_publishes_no_read_marker():
    # Deterministic findings are not formed by reading; "no record" must not be
    # confused with "read nothing".
    assert "ai-review:reads" not in render_comment(make_finding())


def test_read_record_does_not_disturb_the_thread_marker():
    plain = extract_marker(render_comment(make_finding()))
    with_reads = extract_marker(render_comment(make_finding(reads=(Read("grep", "a;b=c.go", 1, 1),))))
    assert plain == with_reads


def test_read_record_does_not_change_identity_or_equality():
    base = make_finding()
    recorded = make_finding(reads=(Read("read_file", "a.go", 1, 2),))
    assert base.fingerprint == recorded.fingerprint
    assert base == recorded
