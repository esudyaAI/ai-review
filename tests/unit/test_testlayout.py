from pathlib import Path

from reviewer.testlayout import NON_CODE_SUFFIXES, describe_test_layout, detect_test_globs


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
