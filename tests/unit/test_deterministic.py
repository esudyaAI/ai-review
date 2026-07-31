from pathlib import Path
from unittest.mock import patch

from reviewer.deterministic import _run, parse_gitleaks, parse_golangci, run_deterministic

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


def test_gitleaks_handles_invalid_json():
    """Parse gitleaks gracefully degrades on malformed JSON."""
    assert parse_gitleaks("{invalid json", ["a.go"]) == []
    assert parse_gitleaks("not json at all", ["a.go"]) == []
    assert parse_gitleaks("", ["a.go"]) == []


def test_golangci_handles_invalid_json():
    """Parse golangci gracefully degrades on malformed JSON."""
    assert parse_golangci("{invalid json", ["a.go"]) == []
    assert parse_golangci("not json at all", ["a.go"]) == []
    assert parse_golangci("", ["a.go"]) == []


def test_run_deterministic_with_missing_tools():
    """Run deterministic gracefully handles missing gitleaks/golangci binaries."""
    repo_path = Path("/tmp/fake-repo")
    # _run catches FileNotFoundError and returns "", which both parsers handle
    findings = run_deterministic(repo_path, "golangci-lint", ["a.go"])
    assert findings == []


def test_run_deterministic_calls_both_parsers():
    """Run deterministic combines gitleaks and linter findings."""
    repo_path = Path("/tmp/fake-repo")

    gitleaks_report = (FIXTURES / "gitleaks.json").read_text(encoding="utf-8")
    golangci_report = (FIXTURES / "golangci.json").read_text(encoding="utf-8")

    with patch("reviewer.deterministic._run") as mock_run:
        # First call: gitleaks, second call: golangci-lint
        mock_run.side_effect = [gitleaks_report, golangci_report]

        findings = run_deterministic(
            repo_path,
            "golangci-lint",
            ["internal/billing/config.go"]
        )

        # Should have 1 gitleaks finding + 1 golangci finding
        assert len(findings) == 2
        assert findings[0].source == "gitleaks"
        assert findings[1].source == "linter"
        assert mock_run.call_count == 2


def test_run_deterministic_warns_about_unsupported_linter(capsys):
    """A configured linter run_deterministic doesn't know how to parse (eslint/ruff/
    shellcheck) must not silently drop findings without any warning."""
    repo_path = Path("/tmp/fake-repo")
    with patch("reviewer.deterministic._run", return_value=""):
        findings = run_deterministic(repo_path, "eslint", ["a.go"])
    assert findings == []
    captured = capsys.readouterr()
    assert "eslint" in captured.err
    assert "не поддержан" in captured.err


def test_run_deterministic_none_linter_does_not_warn(capsys):
    repo_path = Path("/tmp/fake-repo")
    with patch("reviewer.deterministic._run", return_value=""):
        run_deterministic(repo_path, "none", ["a.go"])
    assert "не поддержан" not in capsys.readouterr().err


def test_run_warns_to_stderr_when_the_binary_is_missing(capsys):
    """A configured linter/gitleaks binary that isn't in PATH must not fail silently."""
    result = _run(["definitely-not-a-real-binary-xyz"], Path("/tmp/fake-repo"))
    assert result == ""
    captured = capsys.readouterr()
    assert "definitely-not-a-real-binary-xyz" in captured.err
    assert "не найден в PATH" in captured.err
