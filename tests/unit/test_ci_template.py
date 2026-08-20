"""Структурные проверки шаблона, который подключают потребители.

Шаблон - такой же интерфейс, как код: `templates/ai-review.yml` подключается в
jobbytime-go по `ref: v1`, и опечатка в нём ломает чужой пайплайн, а не наш.
Здесь проверяется ровно то, что должно быть верно, чтобы коды выхода ревьюера
что-то значили в пайплайне потребителя.
"""

from pathlib import Path

import pytest
import yaml

from reviewer.__main__ import EXIT_FINDINGS, EXIT_TECHNICAL, EXIT_TOO_LARGE

TEMPLATE = Path(__file__).resolve().parents[2] / "templates" / "ai-review.yml"


@pytest.fixture(scope="module")
def job() -> dict:
    # У шаблона два YAML-документа: spec с inputs и сами джобы. Нужен второй.
    documents = [d for d in yaml.safe_load_all(TEMPLATE.read_text(encoding="utf-8")) if d]
    body = documents[-1]
    assert "ai-review" in body, f"джоб ai-review исчез из шаблона: {sorted(body)}"
    return body["ai-review"]


def test_only_a_technical_failure_is_retried(job):
    """Ретрай нужен ровно для одного класса сбоя: ревью не состоялось по технической
    причине (сеть, потолок итераций, упавший старт). Один повтор - потому что второй
    уже не «блип», а поломка, которую надо чинить руками."""
    retry = job.get("retry")
    assert retry, "у ai-review нет retry: технический сбой требует ручного перезапуска"
    assert retry["max"] == 1, retry
    assert retry["exit_codes"] == [EXIT_TECHNICAL], retry


def test_findings_are_never_retried(job):
    """Повтор ревью, которое нашло замечания, стоит вторых денег и ничего не меняет:
    находки детерминированы относительно диффа, а не относительно попытки."""
    assert EXIT_FINDINGS not in job["retry"]["exit_codes"]


def test_an_oversized_diff_is_never_retried(job):
    """Дифф больше лимита при повторе останется больше лимита. Это чинится разбиением
    merge request'а, а не перезапуском джоба."""
    assert EXIT_TOO_LARGE not in job["retry"]["exit_codes"]


def test_the_job_still_blocks_by_default_choice_of_the_consumer(job):
    """allow_failure остаётся входом шаблона, а не зашитым значением: решение
    «блокирует или нет» принимает подключающий репозиторий, и оно видно в его
    .gitlab-ci.yml, а не спрятано здесь."""
    assert job["allow_failure"] == "$[[ inputs.allow_failure ]]"
