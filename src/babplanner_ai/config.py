"""서비스 설정.

값은 환경 변수로만 들어온다. 기본값을 두면 운영에서 개발용 구성이 조용히
선택될 수 있어, 위험한 항목에는 기본값을 두지 않는다.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(str, Enum):
    LOCAL = "local"
    DEV = "dev"
    PRODUCTION = "production"


class DemandForecastMode(str, Enum):
    """식수 예측에 합성데이터 PoC 모델을 허용할지.

    - strict(기본): 운영(production)에서는 합성 모델을 로드하지 않는다.
    - poc: 합성 PoC 모델을 **명시적으로** 허용한다. 실데이터를 구할 수 없는
      프로젝트 단계에서 운영 환경이 PoC 예측을 내놓을 수 있게 하되, 응답과
      status가 합성 PoC임을 계속 밝힌다.

    기본값이 poc면 아무도 고르지 않았을 때 합성 모델이 운영에 나간다.
    """

    STRICT = "strict"
    POC = "poc"


class ConfigurationError(ValueError):
    """구성이 잘못됐다.

    메시지는 우리가 직접 만든다. pydantic 검증에 맡기면 오류 문자열에
    입력값이 그대로 실려 나가는데, 토큰이 그렇게 새면 안 된다.
    """


#: 서비스 토큰 최소 길이. 짧은 토큰은 사실상 없는 것과 같다.
MIN_SERVICE_TOKEN_LENGTH = 32

#: 식수 예측 artifact의 기본 위치. 저장소를 editable로 설치해 로컬에서 돌릴 때
#: research가 만든 파일을 그대로 쓰게 한다. 컨테이너에서는 환경 변수로
#: 마운트 경로를 준다. 파일이 없어도 기동은 하고, 수요 예측만 503이 된다.
_DEMAND_FORECASTING_ROOT = Path(__file__).resolve().parents[2] / "research" / "demand_forecasting"
DEFAULT_DEMAND_FORECAST_MODEL_PATH = _DEMAND_FORECASTING_ROOT / "models" / "meal_demand_model.joblib"
DEFAULT_DEMAND_FORECAST_METADATA_PATH = _DEMAND_FORECASTING_ROOT / "models" / "metadata.json"
DEFAULT_DEMAND_HISTORY_PATH = _DEMAND_FORECASTING_ROOT / "data" / "synthetic_meal_demand.csv"
DEFAULT_DEMAND_CCTV_LOG_PATH = _DEMAND_FORECASTING_ROOT / "data" / "synthetic_cctv_daily.csv"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    app_env: AppEnv = Field(alias="APP_ENV")
    demand_forecast_mode: DemandForecastMode = Field(
        default=DemandForecastMode.STRICT, alias="DEMAND_FORECAST_MODE"
    )
    service_token: str = Field(alias="AI_SERVICE_TOKEN")
    # 식수 예측(demand-forecast). 읽기만 한다. 서비스가 이 파일들을 쓰지 않는다.
    demand_forecast_model_path: Path = Field(
        default=DEFAULT_DEMAND_FORECAST_MODEL_PATH, alias="DEMAND_FORECAST_MODEL_PATH"
    )
    demand_forecast_metadata_path: Path = Field(
        default=DEFAULT_DEMAND_FORECAST_METADATA_PATH, alias="DEMAND_FORECAST_METADATA_PATH"
    )
    demand_history_path: Path = Field(
        default=DEFAULT_DEMAND_HISTORY_PATH, alias="DEMAND_HISTORY_PATH"
    )
    demand_cctv_log_path: Path = Field(
        default=DEFAULT_DEMAND_CCTV_LOG_PATH, alias="DEMAND_CCTV_LOG_PATH"
    )

    def validated(self) -> "Settings":
        """구성 자체가 위험한 조합인지 확인한다."""
        if len(self.service_token.strip()) < MIN_SERVICE_TOKEN_LENGTH:
            raise ConfigurationError(
                f"AI_SERVICE_TOKEN 은 최소 {MIN_SERVICE_TOKEN_LENGTH}자여야 합니다."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings().validated()  # type: ignore[call-arg]
