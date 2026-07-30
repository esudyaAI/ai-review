from pathlib import Path

import pytest

from reviewer.config import Config, ConfigError, Limits, load_config

FIXTURE = Path(__file__).parent.parent / "fixtures" / "ai-review.yml"


def test_loads_values_from_file():
    config = load_config(FIXTURE)
    assert config.linter == "golangci-lint"
    assert config.test_command == "make test-unit"
    assert config.model == "review-model"
    assert config.model_high_risk == "review-model-strong"
    assert config.limits == Limits(max_files=40, max_changed_lines=2000)


def test_missing_file_yields_defaults():
    config = load_config(Path("/nonexistent/ai-review.yml"))
    assert config.linter == "none"
    assert config.limits == Limits()
    assert config.provider == "openai"


def test_high_risk_path_selects_stronger_model():
    config = load_config(FIXTURE)
    assert config.model_for(["internal/user/user.go"]) == "review-model"
    assert config.model_for(["internal/billing/subscription.go"]) == "review-model-strong"
    assert config.model_for(["migrations/new/001_init.sql"]) == "review-model-strong"


def test_unknown_linter_is_rejected():
    with pytest.raises(ConfigError):
        Config(
            linter="clang-tidy",
            test_command="",
            provider="openai",
            base_url="https://api.openai.com/v1",
            model="m",
            model_high_risk="m",
            high_risk_paths=(),
            limits=Limits(),
        ).validate()
