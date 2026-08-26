"""애플리케이션 진입점."""

from __future__ import annotations

from fastapi import FastAPI

from babplanner_ai.config import get_settings
from babplanner_ai.waste.api import router as waste_router


def create_app() -> FastAPI:
    # 기동 시점에 설정을 읽어 잘못된 구성이면 바로 실패하게 한다. 첫 요청이
    # 올 때까지 미루면 잘못 뜬 서버가 정상으로 보인다.
    get_settings()

    app = FastAPI(
        title="밥플래너 잔반 분석 서비스",
        version="0.1.0",
        description=(
            "서비스 백엔드 전용 내부 API. 사용자 앱이 직접 호출하는 endpoint는 없다."
        ),
    )
    app.include_router(waste_router)
    return app


app = create_app()
