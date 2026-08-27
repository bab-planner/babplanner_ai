"""테스트 공용 fixture.

실제 토큰이나 실제 이미지를 쓰지 않는다. 여기 있는 값은 전부 테스트용
합성 데이터다.
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterator
from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient
from PIL import Image

#: 테스트 전용 가짜 토큰. 실제 서비스 토큰이 아니다.
TEST_SERVICE_TOKEN = "test-only-service-token-not-a-real-secret-0123456789"

TEST_ENV = {
    "APP_ENV": "local",
    "WASTE_MODEL_MODE": "fixture",
    # 수요/발주는 기본이 disabled다. 경계를 확인하려면 켜 둬야 한다. 꺼진
    # 상태는 설정을 갈아 끼우는 테스트에서 따로 확인한다.
    "DEMAND_MODEL_MODE": "fixture",
    "PROCUREMENT_MODEL_MODE": "fixture",
    "AI_SERVICE_TOKEN": TEST_SERVICE_TOKEN,
}

# main 모듈은 import 시점에 앱을 만들면서 설정을 읽는다. 잘못 뜬 서버가
# 정상으로 보이지 않게 하려는 의도이므로, 테스트도 import 전에 환경을 세운다.
os.environ.update(TEST_ENV)
os.environ.pop("AI_MAX_IMAGE_BYTES", None)

#: 이 환경의 socket 생성이 간헐적으로 내는 오류. 다른 OSError는 다시
#: 시도하지 않는다.
WINDOWS_INVALID_POINTER = 10014
MAX_CLIENT_ATTEMPTS = 3


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

    with ExitStack() as stack:
        test_client = None
        # 이 환경은 socketpair 생성 자체가 간헐적으로 WinError 10014로
        # 실패한다. 여기서 한 번 실패하면 client를 쓰는 테스트가 전부
        # 무너지므로 그 오류에 한해 다시 시도한다.
        #
        # 다시 시도하는 범위는 client를 만드는 것뿐이다. 요청 결과나 단언
        # 실패는 다시 하지 않는다. 그건 감춰야 할 것이 아니라 봐야 할 것이다.
        for attempt in range(MAX_CLIENT_ATTEMPTS):
            try:
                test_client = stack.enter_context(TestClient(create_app()))
                break
            except OSError as error:
                if (
                    getattr(error, "winerror", None) != WINDOWS_INVALID_POINTER
                    or attempt == MAX_CLIENT_ATTEMPTS - 1
                ):
                    raise
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
