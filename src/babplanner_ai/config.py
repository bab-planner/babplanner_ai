"""서비스 설정.

값은 환경 변수로만 들어온다. 기본값을 두면 운영에서 개발용 구성이 조용히
선택될 수 있어, 위험한 항목에는 기본값을 두지 않는다.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(str, Enum):
    LOCAL = "local"
    DEV = "dev"
    PRODUCTION = "production"


class WasteModelMode(str, Enum):
    """분석기 구현 선택. 기본값을 두지 않는다."""

    FIXTURE = "fixture"


class ConfigurationError(ValueError):
    """구성이 잘못됐다.

    메시지는 우리가 직접 만든다. pydantic 검증에 맡기면 오류 문자열에
    입력값이 그대로 실려 나가는데, 토큰이 그렇게 새면 안 된다.
    """


#: 서비스 토큰 최소 길이. 짧은 토큰은 사실상 없는 것과 같다.
MIN_SERVICE_TOKEN_LENGTH = 32

#: 이미지 최대 크기. 서비스 백엔드에도 상한이 있지만 여기서도 스스로 막는다.
DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    app_env: AppEnv = Field(alias="APP_ENV")
    waste_model_mode: WasteModelMode = Field(alias="WASTE_MODEL_MODE")
    service_token: str = Field(alias="AI_SERVICE_TOKEN")
    max_image_bytes: int = Field(
        default=DEFAULT_MAX_IMAGE_BYTES, alias="AI_MAX_IMAGE_BYTES"
    )

    def validated(self) -> "Settings":
        """구성 자체가 위험한 조합인지 확인한다."""
        if len(self.service_token.strip()) < MIN_SERVICE_TOKEN_LENGTH:
            raise ConfigurationError(
                f"AI_SERVICE_TOKEN 은 최소 {MIN_SERVICE_TOKEN_LENGTH}자여야 합니다."
            )
        if self.max_image_bytes <= 0:
            raise ConfigurationError("AI_MAX_IMAGE_BYTES 는 양수여야 합니다.")
        # fixture는 이미지를 판독하지 않는다. 운영에서 선택되면 분석하지 않은
        # 결과가 승인으로 나가므로 기동 단계에서 막는다.
        if (
            self.app_env is AppEnv.PRODUCTION
            and self.waste_model_mode is WasteModelMode.FIXTURE
        ):
            raise ConfigurationError(
                "운영 환경에서는 fixture 분석기를 쓸 수 없습니다. "
                "실제 모델 어댑터를 등록하세요."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings().validated()  # type: ignore[call-arg]
