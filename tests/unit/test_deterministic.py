from pathlib import Path
from unittest.mock import patch

from reviewer.deterministic import _run, parse_gitleaks, run_deterministic

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


def test_empty_reports_produce_no_findings():
    assert parse_gitleaks("[]", ["a.go"]) == []


def test_gitleaks_handles_invalid_json():
    """Parse gitleaks gracefully degrades on malformed JSON."""
    assert parse_gitleaks("{invalid json", ["a.go"]) == []
    assert parse_gitleaks("not json at all", ["a.go"]) == []
    assert parse_gitleaks("", ["a.go"]) == []


def test_run_deterministic_with_missing_tools():
    """Run deterministic gracefully handles a missing gitleaks binary."""
    repo_path = Path("/tmp/fake-repo")
    # _run catches FileNotFoundError and returns "", which parse_gitleaks handles.
    findings = run_deterministic(repo_path, ["a.go"])
    assert findings == []


def test_run_deterministic_calls_gitleaks():
    """Run deterministic runs gitleaks and maps its findings."""
    repo_path = Path("/tmp/fake-repo")
    gitleaks_report = (FIXTURES / "gitleaks.json").read_text(encoding="utf-8")

    with patch("reviewer.deterministic._run", return_value=gitleaks_report) as mock_run:
        findings = run_deterministic(repo_path, ["internal/billing/config.go"])

        assert len(findings) == 1
        assert findings[0].source == "gitleaks"
        assert mock_run.call_count == 1


def test_run_warns_to_stderr_when_the_binary_is_missing(capsys):
    """A configured binary that isn't in PATH must not fail silently."""
    result = _run(["definitely-not-a-real-binary-xyz"], Path("/tmp/fake-repo"))
    assert result == ""
    captured = capsys.readouterr()
    assert "definitely-not-a-real-binary-xyz" in captured.err
    assert "не найден в PATH" in captured.err
