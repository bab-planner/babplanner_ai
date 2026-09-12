"""식수 예측(demand-forecast) 라우트.

서비스 백엔드(Spring)만 부른다. 사용자 인증과 관리자 권한은 Spring이 맡고,
여기서는 서비스 간 토큰으로 호출자가 Spring인지만 확인한다.

- GET  /internal/v1/ai/demand-forecast/status   모델·이력 사용 가능 여부
- POST /internal/v1/ai/demand-forecast/predict  다음 날 중식 식수 예측

기존 /internal/v1/ai/demand/predict 는 그대로 둔다. 다른 계약이다.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from babplanner_ai.demand_forecast.history import InsufficientHistoryError
from babplanner_ai.demand_forecast.models import (
    DemandForecastRequest,
    DemandForecastResponse,
    DemandForecastStatus,
)
from babplanner_ai.demand_forecast.service import (
    DemandForecastService,
    ForecastUnavailableError,
)
from babplanner_ai.security import require_service_token

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/internal/v1/ai/demand-forecast",
    tags=["InternalAI"],
    # 꺼져 있는지, 어떤 모델이 떠 있는지도 아무나 알아낼 필요는 없다.
    dependencies=[Depends(require_service_token)],
)


def get_forecast_service(request: Request) -> DemandForecastService:
    return request.app.state.demand_forecast_service


@router.get(
    "/status",
    response_model=DemandForecastStatus,
    response_model_exclude_none=True,
    summary="식수 예측 모델 사용 가능 여부",
)
def forecast_status(
    service: DemandForecastService = Depends(get_forecast_service),
) -> DemandForecastStatus:
    return service.status()


@router.post(
    "/predict",
    response_model=DemandForecastResponse,
    summary="다음 날 중식 식수 예측",
    responses={
        422: {"description": "요청 검증 실패, 또는 예측일 이전 이력이 모자람"},
        503: {"description": "모델 artifact 또는 이력을 쓸 수 없음"},
    },
)
def predict_demand_forecast(
    body: DemandForecastRequest,
    service: DemandForecastService = Depends(get_forecast_service),
) -> DemandForecastResponse:
    # 동기 함수다. FastAPI가 스레드 풀에서 돌리므로 추론이 event loop를 막지 않는다.
    try:
        return service.predict(body)
    except InsufficientHistoryError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(error)) from error
    except ForecastUnavailableError as error:
        # 예측을 못 한 것이지 0명이라고 판정한 것이 아니다. 숫자로 위장하지 않는다.
        log.warning("식수 예측을 수행하지 못했습니다. date=%s reason=%s", body.date, error)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"식수 예측을 사용할 수 없습니다: {error}",
        ) from error
