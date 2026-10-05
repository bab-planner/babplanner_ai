"""애플리케이션 진입점."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from babplanner_ai.config import AppEnv, get_settings
from babplanner_ai.demand_forecast.api import router as demand_forecast_router
from babplanner_ai.demand_forecast.service import build_demand_forecast_service
from babplanner_ai.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # 식수 예측 artifact를 기동할 때 한 번 로드한다. 실패해도 서버는 뜬다.
    # 쓸 수 없다는 사실은 status가 알리고, predict는 503으로 답한다.
    app.state.demand_forecast_service.warm_up()
    yield


def create_app() -> FastAPI:
    # 기동 시점에 설정을 읽어 잘못된 구성이면 바로 실패하게 한다. 첫 요청이
    # 올 때까지 미루면 잘못 뜬 서버가 정상으로 보인다.
    settings = get_settings()

    app = FastAPI(
        title="밥플래너 AI 서비스",
        version="0.1.0",
        description=(
            "서비스 백엔드 전용 내부 AI 추론 API. 사용자 앱이 직접 호출하는 endpoint는 없다."
        ),
        lifespan=lifespan,
        # 운영에서는 내부 API 문서 화면을 열어 두지 않는다.
        docs_url=None if settings.app_env is AppEnv.PRODUCTION else "/docs",
        redoc_url=None if settings.app_env is AppEnv.PRODUCTION else "/redoc",
        openapi_url=None if settings.app_env is AppEnv.PRODUCTION else "/openapi.json",
    )
    app.state.demand_forecast_service = build_demand_forecast_service(settings)
    # 라우트는 항상 등록한다. 꺼진 기능은 없는 척하는 대신 503으로 답한다.
    # 404로 만들면 배포가 잘못됐는지 기능이 꺼졌는지 구분되지 않는다.
    app.include_router(health_router)
    app.include_router(demand_forecast_router)
    return app


app = create_app()
