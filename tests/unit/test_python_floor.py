"""Объявленная нижняя версия Python проверяется прогоном, а не словом.

requires-python в pyproject.toml — утверждение. До 2026-09-15 его не проверяло
ничто, и оно разошлось с фактом: стояло ">=3.12", при том что весь пакет
компилируется и все тесты проходят на 3.10. Разошлось не в безопасную сторону:
матчер и infra живут на 3.11, и пол выше их означал бы, что общий код нельзя
переиспользовать, хотя технически можно.

Проверяется здесь связь двух мест, а не сама совместимость: совместимость
доказывает джоба test_floor, которая гоняет сюиту на образе этой версии. Этот
файл держит то, что образ джобы и пол в pyproject — одно и то же число. Два
числа, названных руками в разных файлах, расходятся молча; так и вышло.
"""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
CI = ROOT / ".gitlab-ci.yml"


def declared_floor() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^requires-python\s*=\s*"[><=]*\s*([0-9]+\.[0-9]+)"', text, re.M)
    assert match, "в pyproject.toml нет requires-python — проверять нечего"
    return match.group(1)


def ci_jobs() -> dict:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))


def test_a_job_runs_the_suite_on_the_declared_floor():
    floor = declared_floor()
    want = f"python:{floor}-slim"

    jobs = ci_jobs()
    on_floor = [
        name
        for name, body in jobs.items()
        if isinstance(body, dict) and body.get("image") == want and "script" in body
    ]
    assert on_floor, (
        f"ни одна джоба не гоняется на объявленном полу {want}; "
        f"requires-python = {floor} остаётся утверждением, которое ничто не проверяет. "
        f"Образы джоб сейчас: "
        + ", ".join(
            sorted({b.get("image", "") for b in jobs.values() if isinstance(b, dict) and b.get("image")})
        )
    )

    for name in on_floor:
        script = " ".join(jobs[name]["script"])
        assert "pytest" in script, f"джоба {name} стоит на полу, но сюиту не гоняет: {script}"


def test_the_floor_is_not_above_the_runtime_image():
    """Пол не может быть выше образа, в котором сервис работает.

    Обратное — пол ниже рантайма — нормально и сегодня так и есть: Dockerfile на
    3.12, пол 3.11. Это значит «работает на 3.11 и выше», а катаем мы 3.12.
    """
    floor = tuple(int(p) for p in declared_floor().split("."))

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    match = re.search(r"^FROM\s+python:([0-9]+\.[0-9]+)", dockerfile, re.M)
    assert match, "в Dockerfile нет строки FROM python:X.Y"
    runtime = tuple(int(p) for p in match.group(1).split("."))

    assert floor <= runtime, (
        f"объявленный пол {floor} выше образа рантайма {runtime}: "
        "пакет объявлен несовместимым с тем, на чём он работает"
    )
