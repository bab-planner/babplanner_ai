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
        "AI_SERVICE_TOKEN": TEST_SERVICE_TOKEN,
    }
    values.update(overrides)
    return Settings(**values).validated()  # type: ignore[arg-type]


def test_service_token_has_no_default(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("APP_ENV", "local")

    with pytest.raises(Exception) as error:
        get_settings()

    assert "AI_SERVICE_TOKEN" in str(error.value)


def test_app_env_has_no_default(clean_env: pytest.MonkeyPatch) -> None:
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


def test_removed_legacy_settings_are_ignored_not_required() -> None:
    # 예전 배포 환경에 남아 있을 수 있는 환경 변수는 무시된다. 기동을 막지도,
    # 어떤 기능을 켜지도 않는다.
    settings = _settings(
        WASTE_MODEL_MODE="fixture",
        DEMAND_MODEL_MODE="fixture",
        PROCUREMENT_MODEL_MODE="fixture",
        AI_MAX_IMAGE_BYTES="1",
    )

    for removed in (
        "waste_model_mode",
        "demand_model_mode",
        "procurement_model_mode",
        "max_image_bytes",
    ):
        assert not hasattr(settings, removed)


class TestDemandForecastMode:
    """합성 PoC 식수 예측 모델을 운영에서 쓰려면 명시적으로 골라야 한다."""

    def test_defaults_to_strict(self) -> None:
        from babplanner_ai.config import DemandForecastMode

        assert _settings().demand_forecast_mode is DemandForecastMode.STRICT

    def test_poc_mode_value_is_accepted(self) -> None:
        from babplanner_ai.config import DemandForecastMode

        settings = _settings(DEMAND_FORECAST_MODE="poc")

        assert settings.demand_forecast_mode is DemandForecastMode.POC

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(Exception):
            _settings(DEMAND_FORECAST_MODE="yes")

    @pytest.mark.parametrize(
        "env,mode,allowed",
        [
            ("local", "strict", True),
            ("dev", "strict", True),
            ("production", "strict", False),
            ("production", "poc", True),
        ],
    )
    def test_synthetic_model_gate(self, env: str, mode: str, allowed: bool) -> None:
        from babplanner_ai.demand_forecast.service import build_demand_forecast_service

        settings = Settings(
            APP_ENV=env,
            AI_SERVICE_TOKEN=TEST_SERVICE_TOKEN,
            DEMAND_FORECAST_MODE=mode,
        )  # type: ignore[arg-type]

        service = build_demand_forecast_service(settings)

        assert service._loader._allow_synthetic is allowed


class TestProductionConfiguration:
    """운영 구성: 합성 PoC 식수 예측은 DEMAND_FORECAST_MODE=poc로 명시한다."""

    def test_production_with_poc_demand_forecast_is_allowed(self) -> None:
        from babplanner_ai.config import AppEnv, DemandForecastMode

        settings = _settings(APP_ENV="production", DEMAND_FORECAST_MODE="poc")

        assert settings.app_env is AppEnv.PRODUCTION
        assert settings.demand_forecast_mode is DemandForecastMode.POC

    def test_production_without_poc_still_starts(self) -> None:
        # strict로 두면 합성 모델만 로드되지 않을 뿐 서비스는 뜬다(status unavailable).
        settings = _settings(APP_ENV="production")

        assert settings.validated() is settings
