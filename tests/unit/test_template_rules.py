"""Правила запуска джобы ревью.

Тестов на сам шаблон не было, хотя он и решает, когда ревью стоит денег.
Замер за 2026-08-30/31: 23 прогона в двух репозиториях, от $0.03 до $0.74 за
штуку. Самый дорогой — на merge request в статусе Draft: 20 итераций,
165 КБ прочитано, $0.74. Черновик по определению не готов к ревью, и платить
за него незачем.

Шаблон читается как текст, а не как YAML-документ: GitLab-специфичный
синтаксис `$[[ inputs.… ]]` и разделитель `---` между spec и телом делают
разбор через yaml.safe_load неоднозначным (два документа, второй — сам
шаблон). Проверять нужно именно то, что попадёт в конфигурацию, поэтому
сравнение идёт по строкам файла.
"""

import pathlib
import re

TEMPLATE = pathlib.Path(__file__).resolve().parents[2] / "templates" / "ai-review.yml"


def _rules(only_rules: bool = True) -> list:
    """Возвращает строки блока rules: джобы ai-review.

    only_rules отбрасывает комментарии. Это не косметика: первая версия теста
    искала строку со словом Draft и находила КОММЕНТАРИЙ, объясняющий правило,
    а не само правило. Проверка тогда смотрела на текст рядом с кодом вместо
    кода — та же ошибка, что искать подстроку по всему скрипту вместо команды.
    """
    text = TEMPLATE.read_text(encoding="utf-8")
    match = re.search(r"\n  rules:\n((?:    .*\n)+)", text)
    assert match, "в шаблоне нет блока rules — джоба запускается всегда"

    lines = match.group(1).rstrip().splitlines()
    if only_rules:
        lines = [line for line in lines if not line.strip().startswith("#")]
    return lines


def test_draft_merge_requests_are_skipped():
    """Draft не ревьюируется.

    Правило стоит ПЕРВЫМ: GitLab берёт первое подошедшее, и правило про
    merge_request_event поймало бы черновик тоже.
    """
    rules = _rules()

    draft_at = next(
        (i for i, line in enumerate(rules) if "CI_MERGE_REQUEST_TITLE" in line), None)
    assert draft_at is not None, (
        "нет правила про Draft: ревью запустится на черновике и потратит "
        "деньги на код, который ещё пишут"
    )
    assert draft_at == 0, (
        "правило про Draft не первое: GitLab берёт первое подошедшее, и "
        "черновик поймает правило про merge_request_event раньше"
    )
    assert "Draft" in rules[0], "правило не про Draft: %s" % rules[0].strip()
    assert "when: never" in rules[draft_at + 1], (
        "за правилом про Draft не следует when: never — черновик всё равно "
        "запустит ревью:\n  %s" % rules[draft_at + 1]
    )


def test_ordinary_merge_requests_still_run():
    """Обратная сторона: обычный merge request по-прежнему ревьюируется.

    Без этой проверки правило про Draft можно было бы «починить», отключив
    ревью совсем.
    """
    rules = "\n".join(_rules())
    assert 'CI_PIPELINE_SOURCE == "merge_request_event"' in rules, (
        "исчезло правило, по которому ревью вообще запускается"
    )


def test_nothing_else_triggers_a_review():
    """Ревью не запускается ни по расписанию, ни по пушу в ветку.

    Последним правилом стоит `when: never`: всё, что не подошло выше, не
    запускается вовсе.
    """
    rules = _rules()
    assert rules[-1].strip() == "- when: never", (
        "последнее правило не `when: never` — ревью может запуститься там, "
        "где его не ждут: по расписанию, по пушу в ветку, вручную"
    )
