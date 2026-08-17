from pathlib import Path

from reviewer.testlayout import (
    NON_CODE_SUFFIXES,
    can_hold_a_test,
    covers,
    describe_test_layout,
    detect_test_globs,
)


def _touch(root: Path, *relative_paths: str) -> None:
    for rel in relative_paths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")


# The prompt used to hardcode `*_test.go`. In a Python repository the model
# grepped for a Go mask, found nothing and concluded there was no coverage -
# three such findings landed on one .gitlab-ci.yml in jobbytime-ai-review !12,
# a repository with 14 test files and 176 passing tests.
def test_python_layout_is_detected(tmp_path):
    _touch(tmp_path, "reviewer/gate.py", "tests/unit/test_gate.py", "tests/unit/test_cli.py")
    assert detect_test_globs(tmp_path) == ("test_*.py",)


def test_go_layout_is_detected(tmp_path):
    _touch(tmp_path, "internal/service/auth.go", "internal/service/auth_test.go")
    assert detect_test_globs(tmp_path) == ("*_test.go",)


def test_javascript_layout_is_detected(tmp_path):
    _touch(tmp_path, "src/app.tsx", "src/app.test.tsx", "src/util.spec.ts")
    assert detect_test_globs(tmp_path) == ("*.spec.ts", "*.test.tsx")


def test_mixed_repository_reports_every_convention_present(tmp_path):
    _touch(tmp_path, "main.go", "main_test.go", "tests/test_x.py", "web/a.test.js")
    assert detect_test_globs(tmp_path) == ("*.test.js", "*_test.go", "test_*.py")


# A repository with no tests at all must say so plainly rather than name a mask
# that matches nothing: "grep found nothing" then means "there are no tests",
# not "you grepped for the wrong thing".
def test_repository_without_tests_reports_nothing(tmp_path):
    _touch(tmp_path, "main.go", "README.md")
    assert detect_test_globs(tmp_path) == ()
    assert "не найдено" in describe_test_layout(tmp_path)


def test_description_names_the_masks(tmp_path):
    _touch(tmp_path, "tests/unit/test_gate.py")
    assert "test_*.py" in describe_test_layout(tmp_path)


def test_non_code_suffixes_cover_the_files_that_cannot_hold_a_test():
    for suffix in (".yml", ".yaml", ".json", ".md", ".toml", ".ini", ".txt", ".lock"):
        assert suffix in NON_CODE_SUFFIXES
    for suffix in (".go", ".py", ".ts", ".js"):
        assert suffix not in NON_CODE_SUFFIXES


# The gate drops `tests` findings on files this function rejects, so a mistake
# here silently swallows real findings - the one thing the drop must never do.
def test_can_hold_a_test_accepts_source_files():
    for path in (
        "internal/service/auth.go", "reviewer/gate.py", "src/app.tsx",
        "src/util.ts", "web/main.js", "lib/thing.rb", "scripts/run.sh",
    ):
        assert can_hold_a_test(path) is True, path


def test_can_hold_a_test_rejects_files_with_nothing_to_call():
    for path in (
        ".gitlab-ci.yml", "config.yaml", "package.json", "pyproject.toml",
        "README.md", "docs/guide.rst", "data/rows.csv", "go.sum", "go.mod",
        "poetry.lock", "migrations/001.sql", "assets/logo.svg", "assets/photo.png",
    ):
        assert can_hold_a_test(path) is False, path


# Dockerfile, Makefile, LICENSE: build and packaging inputs, not units under test.
def test_can_hold_a_test_rejects_files_without_a_suffix():
    for path in ("Dockerfile", "Makefile", "LICENSE", "deploy/Dockerfile"):
        assert can_hold_a_test(path) is False, path


def test_can_hold_a_test_ignores_suffix_case():
    assert can_hold_a_test("CONFIG.YML") is False
    assert can_hold_a_test("Main.GO") is True


# Matching a test to its source by directory was wrong twice over, and the
# reviewer caught both on !14: an unrelated metrics_test.go excused auth.go, and
# in this repository - tests in tests/unit/, code in reviewer/ - it never matched
# at all. Names carry the relation; directories do not.
def test_go_test_matches_its_source_by_name():
    assert covers("internal/service/auth.go", "internal/service/auth_test.go") is True
    assert covers("internal/service/auth.go", "internal/service/metrics_test.go") is False


def test_python_test_matches_across_directories():
    assert covers("reviewer/gate.py", "tests/unit/test_gate.py") is True
    assert covers("reviewer/gate.py", "tests/unit/test_findings.py") is False


def test_javascript_test_and_spec_both_match():
    assert covers("src/app.tsx", "src/app.test.tsx") is True
    assert covers("src/app.tsx", "src/app.spec.tsx") is True
    assert covers("src/app.tsx", "src/other.test.tsx") is False


def test_a_source_file_does_not_cover_itself():
    assert covers("reviewer/gate.py", "reviewer/gate.py") is False
