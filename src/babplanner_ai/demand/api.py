"""수요 예측 라우트.

서비스 백엔드만 부른다. 사용자 앱이 직접 부르는 endpoint는 여기 없다.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from babplanner_ai.config import PredictiveModelMode, Settings, get_settings
from babplanner_ai.demand.estimator import DemandEstimateError, DemandEstimator
from babplanner_ai.demand.fixture_estimator import FixtureDemandEstimator
from babplanner_ai.demand.models import AIDemandRequest, AIDemandResponse
from babplanner_ai.demand.service import DemandPredictionService
from babplanner_ai.security import require_service_token

log = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/v1/ai/demand", tags=["InternalAI"])


def get_estimator(settings: Settings = Depends(get_settings)) -> DemandEstimator:
    """설정이 고른 예측기를 돌려준다.

    disabled면 아무것도 돌려주지 않고 503으로 끝낸다. 끈 상태에서 fixture로
    대신 답하면, 예측 기능이 없는 배포가 예측을 하고 있는 것처럼 보인다.
    """
    if settings.demand_model_mode is PredictiveModelMode.FIXTURE:
        return FixtureDemandEstimator()
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="수요 예측을 사용할 수 없습니다.",
    )


def get_service(
    estimator: DemandEstimator = Depends(get_estimator),
) -> DemandPredictionService:
    return DemandPredictionService(estimator)


@router.post(
    "/predict",
    response_model=AIDemandResponse,
    dependencies=[Depends(require_service_token)],
)
async def predict_demand(
    request: AIDemandRequest,
    service: DemandPredictionService = Depends(get_service),
) -> AIDemandResponse:
    try:
        return service.predict(request)
    except DemandEstimateError as error:
        # 예측을 못 한 것이지 0명이라고 판정한 것이 아니다. 숫자로 위장하지
        # 않는다. 호출자는 이걸 보고 실행을 실패로 끝낸다.
        log.warning("수요 예측을 수행하지 못했습니다. requestId=%s", request.requestId)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="예측기를 사용할 수 없습니다.",
        ) from error
