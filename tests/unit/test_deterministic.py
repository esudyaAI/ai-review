from pathlib import Path

from reviewer.deterministic import parse_gitleaks, parse_golangci

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_gitleaks_findings_are_security_and_high_confidence():
    report = (FIXTURES / "gitleaks.json").read_text(encoding="utf-8")
    findings = parse_gitleaks(report, ["internal/billing/config.go"])
    assert len(findings) == 1
    finding = findings[0]
    assert finding.category == "security"
    assert finding.file == "internal/billing/config.go"
    assert finding.line_start == 12
    assert finding.confidence == "high"
    assert finding.source == "gitleaks"


def test_gitleaks_never_leaks_the_secret_value():
    report = (FIXTURES / "gitleaks.json").read_text(encoding="utf-8")
    finding = parse_gitleaks(report, ["internal/billing/config.go"])[0]
    rendered = " ".join([finding.title, finding.problem, finding.evidence, finding.suggested_fix])
    assert "REDACTED_BY_FIXTURE" not in rendered
    assert "sk_live" not in rendered


def test_gitleaks_ignores_files_outside_the_diff():
    report = (FIXTURES / "gitleaks.json").read_text(encoding="utf-8")
    assert parse_gitleaks(report, ["internal/user/user.go"]) == []


def test_golangci_findings_are_mapped_to_changed_files():
    report = (FIXTURES / "golangci.json").read_text(encoding="utf-8")
    findings = parse_golangci(report, ["internal/billing/config.go"])
    assert len(findings) == 1
    finding = findings[0]
    assert finding.category == "bug"
    assert finding.line_start == 42
    assert "errcheck" in finding.evidence
    assert finding.source == "linter"


def test_empty_reports_produce_no_findings():
    assert parse_gitleaks("[]", ["a.go"]) == []
    assert parse_golangci('{"Issues": null}', ["a.go"]) == []
