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
    with pytest.raises(ConfigError):
        Config(
            test_command="",
            provider="anthropic",
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
