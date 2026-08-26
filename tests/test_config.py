"""설정.

위험한 항목에 기본값이 없는지, 잘못된 조합이 기동을 막는지 본다.
"""

from __future__ import annotations

import pytest

from babplanner_ai.config import MIN_SERVICE_TOKEN_LENGTH, Settings, get_settings
from tests.conftest import TEST_SERVICE_TOKEN


def _settings(**overrides: str) -> Settings:
    values = {
        "APP_ENV": "local",
        "WASTE_MODEL_MODE": "fixture",
        "AI_SERVICE_TOKEN": TEST_SERVICE_TOKEN,
    }
    values.update(overrides)
    return Settings(**values).validated()  # type: ignore[arg-type]


def test_service_token_has_no_default(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("APP_ENV", "local")
    clean_env.setenv("WASTE_MODEL_MODE", "fixture")

    with pytest.raises(Exception) as error:
        get_settings()

    assert "AI_SERVICE_TOKEN" in str(error.value)


def test_model_mode_has_no_default(clean_env: pytest.MonkeyPatch) -> None:
    # 기본값이 fixture라면 운영이 조용히 개발용 분석기를 쓰게 된다.
    clean_env.setenv("APP_ENV", "production")
    clean_env.setenv("AI_SERVICE_TOKEN", TEST_SERVICE_TOKEN)

    with pytest.raises(Exception) as error:
        get_settings()

    assert "WASTE_MODEL_MODE" in str(error.value)


def test_app_env_has_no_default(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("WASTE_MODEL_MODE", "fixture")
    clean_env.setenv("AI_SERVICE_TOKEN", TEST_SERVICE_TOKEN)

    with pytest.raises(Exception) as error:
        get_settings()

    assert "APP_ENV" in str(error.value)


def test_short_token_is_rejected() -> None:
    with pytest.raises(Exception):
        _settings(AI_SERVICE_TOKEN="short")


def test_blank_token_is_rejected() -> None:
    with pytest.raises(Exception):
        _settings(AI_SERVICE_TOKEN=" " * (MIN_SERVICE_TOKEN_LENGTH + 1))


def test_token_error_does_not_leak_the_value() -> None:
    secret = "x" * (MIN_SERVICE_TOKEN_LENGTH - 1)

    with pytest.raises(Exception) as error:
        _settings(AI_SERVICE_TOKEN=secret)

    assert secret not in str(error.value)


def test_fixture_estimator_is_rejected_in_production() -> None:
    with pytest.raises(Exception) as error:
        _settings(APP_ENV="production", WASTE_MODEL_MODE="fixture")

    assert "운영" in str(error.value)


def test_fixture_estimator_is_allowed_locally() -> None:
    settings = _settings(APP_ENV="local")

    assert settings.waste_model_mode.value == "fixture"


def test_unknown_model_mode_is_rejected() -> None:
    with pytest.raises(Exception):
        _settings(WASTE_MODEL_MODE="best-model-v3")


def test_image_limit_has_a_sane_default() -> None:
    settings = _settings()

    assert settings.max_image_bytes == 10 * 1024 * 1024


def test_image_limit_must_be_positive() -> None:
    with pytest.raises(Exception):
        _settings(AI_MAX_IMAGE_BYTES="0")
