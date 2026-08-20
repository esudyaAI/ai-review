"""Checks on the script that checks the image.

The script runs only in build_image on main - that is, where a mistake in it has
nobody left to notice. That is exactly what was wrong with its predecessor:
`docker run ... || true` could never fail, and nobody noticed the check was
checking nothing. So what is tested here is not what the script prints but the one
property that broke last time: a failure of any single check must fail the script.

The container runtime is replaced by a stub through DOCKER_CMD, the same way
jobbytime-go's scripts replace the scanner through SECRET_SCAN_CMD.

The Russian strings in the stub are data, not prose: they imitate the reviewer's
real output, which the script greps for.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "image-selfcheck.sh"

STUB = '''#!/usr/bin/env python3
import os
import sys

args = sys.argv[1:]
mode = os.environ.get("STUB_MODE", "ok")

if "gitleaks" in args:
    if mode == "no-gitleaks":
        sys.exit(127)
    print("8.18.4")
    sys.exit(0)

if "--help" in args:
    if mode == "broken-import":
        print("ModuleNotFoundError", file=sys.stderr)
        sys.exit(1)
    print("usage: reviewer")
    sys.exit(0)

if "review" in args:
    if mode == "startup-returns-findings-code":
        print("AI review: итог — есть блокирующие замечания, код выхода 1")
        sys.exit(1)
    if mode == "no-outcome-line":
        print("AI review: техническая ошибка при старте: ConfigError: нет файла")
        sys.exit(2)
    print("AI review: техническая ошибка при старте: ConfigError: нет файла")
    print("AI review: итог — ревью не состоялось из-за технической ошибки, код выхода 2")
    sys.exit(2)

if "python" in args:
    sys.stdin.read()
    if mode == "bad-package":
        print("prompts/system.md is empty", file=sys.stderr)
        sys.exit(1)
    print("installed at /usr/local/lib/python3.12/site-packages/reviewer")
    sys.exit(0)

print("unexpected stub call: " + " ".join(args), file=sys.stderr)
sys.exit(3)
'''


@pytest.fixture
def stub(tmp_path) -> Path:
    path = tmp_path / "docker-stub"
    path.write_text(STUB, encoding="utf-8")
    path.chmod(0o755)
    return path


# The stub has a `#!/usr/bin/env python3` shebang, so the interpreter's own directory
# has to be on PATH. In the CI image (python:3.12-slim) python lives in /usr/local/bin
# and there is no /usr/bin/python3 at all - a hardcoded "/usr/bin:/bin" would pass here
# and fail in the pipeline.
STUB_PATH = f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin"


def run(stub: Path, mode: str, image: str = "registry.example/img:sha") -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sh", str(SCRIPT), image],
        env={"PATH": STUB_PATH, "DOCKER_CMD": str(stub), "STUB_MODE": mode},
        capture_output=True,
        text=True,
    )


def test_a_healthy_image_passes(stub):
    result = run(stub, "ok")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fit to push" in result.stdout


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("no-gitleaks", "gitleaks"),
        ("broken-import", "--help"),
        ("startup-returns-findings-code", "exit code 2"),
        ("no-outcome-line", "outcome line"),
        ("bad-package", ""),
    ],
)
def test_every_broken_image_fails_the_script(stub, mode, expected):
    """One broken property per row. If some check stops failing the script, exactly
    its own parameter fails, and it is visible which one."""
    result = run(stub, mode)
    assert result.returncode != 0, f"{mode}: the script did not fail\n{result.stdout}{result.stderr}"
    if expected:
        assert expected in result.stderr, f"{mode}: {result.stderr}"


def test_the_exit_code_check_does_not_accept_the_findings_code(stub):
    """Spelled out separately because it is the quietest failure available: an image
    whose failed start returns 1 looks to the consumer like "the reviewer found
    something". The pipeline goes red for the right reason with the wrong
    explanation, and a retry keyed on exit code 2 never fires."""
    result = run(stub, "startup-returns-findings-code")
    assert result.returncode == 1
    assert "got 1" in result.stderr, result.stderr


def test_a_missing_argument_is_a_usage_error_not_a_pass(stub):
    result = subprocess.run(
        ["sh", str(SCRIPT)],
        env={"PATH": STUB_PATH, "DOCKER_CMD": str(stub)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "usage" in result.stderr
