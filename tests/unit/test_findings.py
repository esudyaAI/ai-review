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
    assert extract_marker(body) == (finding.fingerprint, finding.file)


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
