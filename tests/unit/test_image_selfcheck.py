"""Проверки самого скрипта, проверяющего образ.

Скрипт запускается только в джобе build_image на main — то есть там, где ошибку в нём
уже некому заметить. Ровно этим и был плох его предшественник: `docker run ... || true`
никогда не падал, и никто год не замечал, что проверка ничего не проверяет. Поэтому
здесь проверяется не «что печатает скрипт», а единственное свойство, которое у него
сломалось в прошлый раз: провал любой из проверок обязан завалить скрипт.

Контейнерный рантайм подменяется заглушкой через DOCKER_CMD — так же, как в
jobbytime-go скрипты подменяют сканер через SECRET_SCAN_CMD.
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
    """По одному сломанному свойству на строку. Если какая-то проверка перестанет
    заваливать скрипт — падёт ровно её параметр, и видно, какая именно."""
    result = run(stub, mode)
    assert result.returncode != 0, f"{mode}: скрипт не упал\n{result.stdout}{result.stderr}"
    if expected:
        assert expected in result.stderr, f"{mode}: {result.stderr}"


def test_the_exit_code_check_does_not_accept_the_findings_code(stub):
    """Отдельно и подробно, потому что это самая тихая из возможных поломок: образ,
    у которого упавший старт возвращает 1, выглядит для потребителя как «ревьюер нашёл
    замечания». Пайплайн краснеет по правильной причине с неправильным объяснением, а
    ретрай, настроенный на код 2, не срабатывает."""
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
