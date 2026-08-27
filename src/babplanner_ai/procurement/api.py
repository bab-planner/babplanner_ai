"""발주 추천 라우트."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from babplanner_ai.config import PredictiveModelMode, Settings, get_settings
from babplanner_ai.procurement.estimator import (
    ProcurementEstimateError,
    ProcurementRecommender,
)
from babplanner_ai.procurement.fixture_estimator import FixtureProcurementRecommender
from babplanner_ai.procurement.models import (
    AIProcurementRequest,
    AIProcurementResponse,
)
from babplanner_ai.procurement.service import ProcurementRecommendationService
from babplanner_ai.security import require_service_token

log = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/v1/ai/procurement", tags=["InternalAI"])


def get_recommender(
    settings: Settings = Depends(get_settings),
) -> ProcurementRecommender:
    """설정이 고른 추천기를 돌려준다. disabled면 503이다."""
    if settings.procurement_model_mode is PredictiveModelMode.FIXTURE:
        return FixtureProcurementRecommender()
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="발주 추천을 사용할 수 없습니다.",
    )


def get_service(
    recommender: ProcurementRecommender = Depends(get_recommender),
) -> ProcurementRecommendationService:
    return ProcurementRecommendationService(recommender)


@router.post(
    "/recommend",
    response_model=AIProcurementResponse,
    dependencies=[Depends(require_service_token)],
)
async def recommend_procurement(
    request: AIProcurementRequest,
    service: ProcurementRecommendationService = Depends(get_service),
) -> AIProcurementResponse:
    try:
        return service.recommend(request)
    except ProcurementEstimateError as error:
        log.warning("발주 추천을 수행하지 못했습니다. requestId=%s", request.requestId)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="추천기를 사용할 수 없습니다.",
        ) from error
