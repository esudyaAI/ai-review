from pathlib import Path

import pytest

from reviewer.config import Config, ConfigError, Limits, load_config

FIXTURE = Path(__file__).parent.parent / "fixtures" / "ai-review.yml"


def test_loads_values_from_file():
    config = load_config(FIXTURE)
    assert config.test_command == "make test-unit"
    assert config.model == "review-model"
    assert config.model_high_risk == "review-model-strong"
    assert config.limits == Limits(max_files=40, max_changed_lines=2000)


def test_missing_file_without_model_is_rejected():
    # A missing .ai-review.yml (or one that omits `model`) previously produced a
    # Config with model == "" that passed validate() silently, only to fail later
    # with a confusing error from the OpenAI call itself. It must be rejected here.
    with pytest.raises(ConfigError):
        load_config(Path("/nonexistent/ai-review.yml"))


def test_missing_model_is_rejected_even_with_other_defaults_valid():
    with pytest.raises(ConfigError):
        Config(
            test_command="",
            provider="openai",
            base_url="https://api.openai.com/v1",
            model="",
            model_high_risk="",
            high_risk_paths=(),
            limits=Limits(),
        ).validate()


def test_high_risk_path_selects_stronger_model():
    config = load_config(FIXTURE)
    assert config.model_for(["internal/user/user.go"]) == "review-model"
    assert config.model_for(["internal/billing/subscription.go"]) == "review-model-strong"
    assert config.model_for(["migrations/new/001_init.sql"]) == "review-model-strong"


def test_unsupported_provider_is_rejected():
    # "anthropic" used to be the example here; it is supported since the Claude
    # provider landed, so the example is a provider the reviewer does not know.
    with pytest.raises(ConfigError):
        Config(
            test_command="",
            provider="gemini",
            base_url="https://api.openai.com/v1",
            model="m",
            model_high_risk="m",
            high_risk_paths=(),
            limits=Limits(),
        ).validate()


def test_missing_config_file_says_the_file_is_missing(tmp_path):
    """The reviewer's own repository had no .ai-review.yml for two weeks and nobody
    noticed: the job died at startup with "model must not be empty", which reads as
    a bad value rather than an absent file. Two seconds, exit 2, allow_failure -
    invisible. The message has to name the real cause."""
    missing = tmp_path / ".ai-review.yml"

    with pytest.raises(ConfigError) as exc:
        load_config(missing)

    assert str(missing) in str(exc.value)
    assert "not found" in str(exc.value)


def _config(**overrides):
    data = dict(
        test_command="", provider="openai", base_url="https://api.openai.com/v1",
        model="mini", model_high_risk="strong",
        high_risk_paths=("internal/service/auth_*", "*stripe*"), limits=Limits(),
    )
    data.update(overrides)
    return Config(**data)


# One high-risk file used to pull the whole diff to the strong model, so a merge
# request that touched auth_service.go plus eight unrelated files paid the strong
# model's output price ($30/M against $4.50/M) for all nine. Splitting the files
# lets each half be reviewed by the model its own risk deserves.
def test_split_by_risk_separates_the_two_halves():
    high, rest = _config().split_by_risk((
        "internal/service/auth_service.go",
        "internal/stats/pipeline_stats.go",
        "internal/service/stripe_webhook.go",
        "README.md",
    ))
    assert high == ("internal/service/auth_service.go", "internal/service/stripe_webhook.go")
    assert rest == ("internal/stats/pipeline_stats.go", "README.md")


def test_split_by_risk_keeps_order_and_handles_empty_sides():
    assert _config().split_by_risk(("a.go", "b.go")) == ((), ("a.go", "b.go"))
    assert _config().split_by_risk(("internal/service/auth_a.go",)) == (
        ("internal/service/auth_a.go",), (),
    )
    assert _config().split_by_risk(()) == ((), ())


# Without a separate strong model there is nothing to route to, and splitting
# would only double the fixed cost of a review for no gain.
def test_split_by_risk_is_a_no_op_without_a_distinct_strong_model():
    assert _config(model_high_risk="").split_by_risk(("internal/service/auth_a.go", "b.go")) == (
        (), ("internal/service/auth_a.go", "b.go"),
    )
    assert _config(model_high_risk="mini").split_by_risk(("internal/service/auth_a.go", "b.go")) == (
        (), ("internal/service/auth_a.go", "b.go"),
    )


# Файл бывает опасен не тем, где он лежит, а тем, что в нём написано.
#
# У jobbytime-go в high_risk_paths стоял `migrations/*`, и на дорогую модель
# уходила ЛЮБАЯ миграция. Замер 31.08: 20260831010000 добавляла одну колонку
# (`ADD COLUMN IF NOT EXISTS location text`) и стоила $0.1258 против $0.03 на
# базовой модели.
#
# Опасна не миграция, а сужение схемы: DROP COLUMN, RENAME, SET NOT NULL,
# ALTER COLUMN ... TYPE, DELETE FROM, TRUNCATE. Добавление колонки не ломает
# ни одного читателя, и у go для него уже есть отдельный гейт
# (scripts/migration-shape-gate.sh), различающий additive и narrowing строже,
# чем это сделала бы модель.
def _content_config(**overrides):
    data = dict(
        high_risk_paths=("internal/service/auth_*",),
        high_risk_content=(r"DROP\s+COLUMN", r"TRUNCATE\s"),
    )
    data.update(overrides)
    return _config(**data)


def test_path_alone_still_escalates():
    assert _content_config().is_high_risk("internal/service/auth_token.go")


def test_content_escalates_a_path_outside_the_list():
    assert _content_config().is_high_risk(
        "migrations/new/20260901_drop.sql",
        content="ALTER TABLE public.vacancies DROP COLUMN posted_date;",
    )


# Тот самый случай, который и оплачивался лишним.
def test_additive_migration_stays_on_the_cheap_model():
    assert not _content_config().is_high_risk(
        "migrations/new/20260831010000_vacancy_ingest_location.sql",
        content="ALTER TABLE public.vacancy_ingest\n"
                "    ADD COLUMN IF NOT EXISTS location text;",
    )


def test_content_is_matched_case_insensitively():
    assert _content_config().is_high_risk(
        "migrations/new/x.sql", content="alter table t drop column c;")


# Содержимое бывает недоступно: файл удалён, чтение упало. Тогда решает путь.
# Считать такой файл безопасным молча нельзя — дыра оказалась бы ровно там,
# где её труднее всего заметить.
def test_unavailable_content_falls_back_to_the_path():
    config = _content_config()
    assert config.is_high_risk("internal/service/auth_x.go", content=None)
    assert not config.is_high_risk("internal/models/vacancy.go", content=None)


# Репозиторий, не задавший high_risk_content, работает как раньше.
def test_no_content_patterns_means_paths_decide_alone():
    assert not _content_config(high_risk_content=()).is_high_risk(
        "migrations/new/x.sql", content="DROP COLUMN c;")


def test_split_by_risk_reads_content_when_a_reader_is_given():
    contents = {
        "migrations/new/add.sql": "ADD COLUMN IF NOT EXISTS x text;",
        "migrations/new/drop.sql": "DROP COLUMN y;",
    }
    high, rest = _content_config().split_by_risk(
        list(contents), read_content=contents.get)
    assert high == ("migrations/new/drop.sql",)
    assert rest == ("migrations/new/add.sql",)


# Без читателя деление остаётся тем же, что было до этой правки: только пути.
def test_split_by_risk_without_a_reader_ignores_content():
    high, rest = _content_config().split_by_risk(("migrations/new/drop.sql",))
    assert high == ()
    assert rest == ("migrations/new/drop.sql",)


# model_for отвечает на тот же вопрос одним словом и обязан отвечать так же.
def test_model_for_agrees_with_is_high_risk_on_content():
    config = _content_config()
    contents = {"migrations/new/drop.sql": "DROP COLUMN y;"}
    assert config.model_for(list(contents), read_content=contents.get) == "strong"
    assert config.model_for(("migrations/new/drop.sql",)) == "mini"
# Ревью документации стоит столько же, сколько ревью кода, и не даёт ничего:
# модель ищет ошибки в программе, а программы в диффе нет.
#
# Замер по последним 20 смёрженным MR jobbytime-go: 4 из 20 (20%) не трогали
# ничего, кроме .md. При средней цене прогона $0.03-0.13 это пятая часть счёта,
# потраченная на вычитку текста.
def test_skip_paths_are_empty_by_default():
    # Умолчание не угадывает. Репозиторий, где .md — это документация API или
    # содержимое сайта, ничего не должен потерять оттого, что обновил образ.
    assert _config().skip_paths == ()


def test_review_is_skipped_when_every_file_matches():
    config = _config(skip_paths=("*.md", "docs/**"))
    assert config.is_review_pointless(("README.md", "docs/api/v1.md"))


def test_one_code_file_is_enough_to_review():
    config = _config(skip_paths=("*.md", "docs/**"))
    assert not config.is_review_pointless(("README.md", "internal/api/handler.go"))


# Пустой дифф — не повод объявлять ревью бессмысленным: это состояние, которого
# быть не должно, и молча выходить нулём значило бы прятать его.
def test_an_empty_diff_is_not_a_skip():
    assert not _config(skip_paths=("*.md",)).is_review_pointless(())


def test_nothing_is_skipped_without_the_setting():
    assert not _config().is_review_pointless(("README.md",))


# Ревью !25 отметило замену `/**` -> `/*` как потерю рекурсивного смысла.
# Замерено — не так: fnmatch не уважает границы сегментов, его `*` матчит и
# слэши, поэтому обе формы ведут себя одинаково на любой глубине.
#
#     fnmatch("docs/a/b/deep.md", "docs/**")  -> True
#     fnmatch("docs/a/b/deep.md", "docs/*")   -> True
#
# Тест стоит здесь, чтобы этот вопрос не выясняли заново, и чтобы переход на
# матчер с настоящей семантикой ** (pathlib, wcmatch) не проехал молча: с ним
# `docs/*` перестанет ловить вложенные файлы, и это упадёт здесь.
def test_deep_paths_are_matched_by_both_glob_forms():
    for pattern in ("docs/**", "docs/*"):
        config = _config(skip_paths=(pattern,))
        assert config.is_review_pointless(("docs/a/b/deep.md",)), pattern
        assert config.is_review_pointless(("docs/api.md",)), pattern
        assert not config.is_review_pointless(("internal/api/handler.go",)), pattern


# --- generated_paths -----------------------------------------------------------
# A second list, separate from skip_paths on purpose. skip_paths answers "is
# this merge request worth a review at all", and a .md file in a diff that also
# has code still reaches the model by an explicit decision (see
# test_one_code_file_brings_the_model_back in test_cli.py): a README is often
# where it shows that the change is described wrongly.
#
# generated_paths answers a different question. The file is produced by a
# committed builder from committed inputs, and a test in the repository asserts
# it still is. There is no judgement for a reviewer to make about its contents,
# and eight thousand lines of it cost real money to send.
def test_a_generated_file_is_told_apart_from_the_rest():
    config = Config(
        test_command="", provider="openai", base_url="https://api.openai.com/v1",
        model="m", model_high_risk="m", high_risk_paths=(),
        generated_paths=("docs/shared/roles.json", "docs/generated/**"),
        limits=Limits(),
    )
    reviewable, generated = config.partition_generated(
        ("app/main.py", "docs/shared/roles.json", "docs/generated/x/y.json",
         "docs/shared/employment-vocabulary.json"))
    assert reviewable == ("app/main.py", "docs/shared/employment-vocabulary.json")
    assert generated == ("docs/shared/roles.json", "docs/generated/x/y.json")


def test_without_the_list_nothing_is_generated():
    config = Config(
        test_command="", provider="openai", base_url="https://api.openai.com/v1",
        model="m", model_high_risk="m", high_risk_paths=(), limits=Limits(),
    )
    assert config.partition_generated(("a.py", "b.json")) == (("a.py", "b.json"), ())


def test_generated_paths_are_read_from_the_file():
    config = load_config(FIXTURE)
    assert config.generated_paths == ("docs/shared/roles.json",)


# --- Claude provider and the group-level switch -----------------------------
#
# The switch lives in CI variables, not in .ai-review.yml: one variable on the
# GitLab group moves all three repositories to Claude and back without a merge
# request per repository. These tests hold what the variable may and may not do.

from reviewer.config import BUILTIN_PRICING, apply_env_overrides  # noqa: E402

OPENAI_CONFIG = Config(
    provider="openai",
    model="gpt-base",
    model_high_risk="gpt-strong",
    pricing={"gpt-base": {"input": 1.0, "output": 2.0}},
)


def test_anthropic_provider_is_accepted():
    Config(provider="anthropic", model="claude-sonnet-5-5").validate()


def test_no_switch_variable_leaves_the_repository_config_as_written():
    config = apply_env_overrides(OPENAI_CONFIG, {})
    assert config.provider == "openai"
    assert config.model == "gpt-base"
    assert config.model_high_risk == "gpt-strong"


def test_an_empty_switch_variable_is_the_same_as_none():
    # A variable deleted in the UI and one left with an empty value must behave
    # alike: returning to OpenAI on November 1 must not depend on which was done.
    config = apply_env_overrides(OPENAI_CONFIG, {"AI_REVIEW_PROVIDER": "  "})
    assert config.provider == "openai"
    assert config.model == "gpt-base"


def test_switching_to_anthropic_replaces_the_openai_model_names():
    # The repository's model names are OpenAI's. Sent to Anthropic they would be
    # a 404 on every run, so the switch brings its own models.
    config = apply_env_overrides(OPENAI_CONFIG, {"AI_REVIEW_PROVIDER": "anthropic"})
    assert config.provider == "anthropic"
    assert config.model == "claude-sonnet-5-5"
    assert config.model_high_risk == "claude-opus-5-5"


def test_model_variables_win_over_the_defaults():
    config = apply_env_overrides(
        OPENAI_CONFIG,
        {
            "AI_REVIEW_PROVIDER": "anthropic",
            "AI_REVIEW_MODEL": "claude-haiku-5-5",
            "AI_REVIEW_MODEL_HIGH_RISK": "claude-sonnet-5-5",
        },
    )
    assert config.model == "claude-haiku-5-5"
    assert config.model_high_risk == "claude-sonnet-5-5"


def test_the_switch_keeps_everything_that_is_not_about_the_model():
    base = Config(
        provider="openai",
        model="gpt-base",
        high_risk_paths=("reviewer/gate.py",),
        skip_paths=("docs/**",),
        limits=Limits(max_files=7, max_changed_lines=70),
    )
    config = apply_env_overrides(base, {"AI_REVIEW_PROVIDER": "anthropic"})
    assert config.high_risk_paths == ("reviewer/gate.py",)
    assert config.skip_paths == ("docs/**",)
    assert config.limits == Limits(max_files=7, max_changed_lines=70)


def test_an_unknown_provider_in_the_variable_is_rejected():
    with pytest.raises(ConfigError):
        apply_env_overrides(OPENAI_CONFIG, {"AI_REVIEW_PROVIDER": "gemini"})


def test_claude_prices_are_built_in_so_the_cost_line_is_not_unknown():
    # The repositories price only their OpenAI models. Without built-in prices a
    # switched run would print "стоимость неизвестна" for every Claude call, and
    # the whole point of the switch is to watch a fixed budget being spent.
    config = apply_env_overrides(OPENAI_CONFIG, {"AI_REVIEW_PROVIDER": "anthropic"})
    assert config.pricing["claude-sonnet-5-5"] == BUILTIN_PRICING["claude-sonnet-5-5"]
    assert config.pricing["claude-opus-5-5"] == BUILTIN_PRICING["claude-opus-5-5"]
    # The OpenAI prices stay: the fallback reports its tokens under them.
    assert config.pricing["gpt-base"] == {"input": 1.0, "output": 2.0}


def test_repository_prices_win_over_built_in_ones_key_by_key():
    # A key the repository wrote wins; a key it left out comes from the built-in
    # table (2026-10-22, decided for gpt-5.4-mini's missing cached_input).
    base = Config(
        provider="openai",
        model="gpt-base",
        pricing={"claude-sonnet-5-5": {"input": 9.0, "output": 9.0}},
    )
    config = apply_env_overrides(base, {"AI_REVIEW_PROVIDER": "anthropic"})
    assert config.pricing["claude-sonnet-5-5"] == {
        "input": 9.0, "output": 9.0, "cached_input": 0.10, "cache_write": 2.50,
    }


def test_a_price_key_missing_in_the_repository_is_filled_on_load(tmp_path):
    # All three repositories priced gpt-5.4-mini without cached_input, so its
    # cached tokens (~85% of the prompt) were billed at the full input rate in
    # the cost line. Filling the key here fixes all three without a merge
    # request each - every run reads its prices through load_config.
    path = tmp_path / ".ai-review.yml"
    path.write_text(
        "model: gpt-5.4-mini-2026-03-17\n"
        "pricing:\n"
        "  gpt-5.4-mini-2026-03-17:\n"
        "    input: 0.75\n"
        "    output: 4.50\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.pricing["gpt-5.4-mini-2026-03-17"] == {
        "input": 0.75, "output": 4.50, "cached_input": 0.075,
    }


def test_a_price_key_the_repository_wrote_is_kept_on_load(tmp_path):
    path = tmp_path / ".ai-review.yml"
    path.write_text(
        "model: gpt-5.4-mini-2026-03-17\n"
        "pricing:\n"
        "  gpt-5.4-mini-2026-03-17:\n"
        "    input: 0.75\n"
        "    output: 4.50\n"
        "    cached_input: 0.30\n",
        encoding="utf-8",
    )
    assert load_config(path).pricing["gpt-5.4-mini-2026-03-17"]["cached_input"] == 0.30


def test_a_model_the_repository_did_not_price_is_not_added_on_load(tmp_path):
    # Filling keys is not inventing entries: an OpenAI run's report lists only
    # the models that ran, and built-in Claude prices change nothing there.
    path = tmp_path / ".ai-review.yml"
    path.write_text("model: m\n", encoding="utf-8")
    assert "gpt-5.4-mini-2026-03-17" not in load_config(path).pricing


def test_built_in_openai_prices_match_the_published_model_pages():
    # developers.openai.com/api/docs/models/gpt-5.4-mini and .../gpt-5.5,
    # read 2026-10-22, standard tier, prompts up to 272K tokens.
    assert BUILTIN_PRICING["gpt-5.4-mini-2026-03-17"] == {
        "input": 0.75, "output": 4.50, "cached_input": 0.075,
    }
    assert BUILTIN_PRICING["gpt-5.5-2026-04-23"] == {
        "input": 5.00, "output": 30.00, "cached_input": 0.50,
    }


def test_built_in_claude_prices_match_the_published_table():
    # Copied from platform.claude.com/docs/en/about-claude/pricing on 2026-10-22,
    # USD per million tokens. A change of price is a change of this test.
    assert BUILTIN_PRICING["claude-sonnet-5-5"] == {
        "input": 2.00, "output": 10.00, "cached_input": 0.10, "cache_write": 2.50,
    }
    assert BUILTIN_PRICING["claude-opus-5-5"] == {
        "input": 4.00, "output": 20.00, "cached_input": 0.20, "cache_write": 5.00,
    }
