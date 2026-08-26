"""테스트 공용 fixture.

실제 토큰이나 실제 이미지를 쓰지 않는다. 여기 있는 값은 전부 테스트용
합성 데이터다.
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

#: 테스트 전용 가짜 토큰. 실제 서비스 토큰이 아니다.
TEST_SERVICE_TOKEN = "test-only-service-token-not-a-real-secret-0123456789"

TEST_ENV = {
    "APP_ENV": "local",
    "WASTE_MODEL_MODE": "fixture",
    "AI_SERVICE_TOKEN": TEST_SERVICE_TOKEN,
}

# main 모듈은 import 시점에 앱을 만들면서 설정을 읽는다. 잘못 뜬 서버가
# 정상으로 보이지 않게 하려는 의도이므로, 테스트도 import 전에 환경을 세운다.
os.environ.update(TEST_ENV)
os.environ.pop("AI_MAX_IMAGE_BYTES", None)


def png_bytes(size: tuple[int, int] = (8, 8)) -> bytes:
    """합성 PNG. 개인정보가 담긴 실제 사진을 쓰지 않는다."""
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 170, 120)).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes(size: tuple[int, int] = (8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 90, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """테스트 전체가 하나의 client를 공유한다.

    이 환경의 Windows socket 문제로 event loop를 새로 만들 때마다
    socketpair가 간헐적으로 실패한다. client를 한 번만 열면 loop도 하나만
    쓴다.
    """
    from babplanner_ai.config import get_settings
    from babplanner_ai.main import create_app

    os.environ.update(TEST_ENV)
    os.environ.pop("AI_MAX_IMAGE_BYTES", None)
    get_settings.cache_clear()

    with TestClient(create_app()) as test_client:
        yield test_client

    get_settings.cache_clear()


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """설정 관련 환경 변수를 모두 지운 상태."""
    from babplanner_ai.config import get_settings

    for key in (*TEST_ENV, "AI_MAX_IMAGE_BYTES"):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()
