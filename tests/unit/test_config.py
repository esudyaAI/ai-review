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
