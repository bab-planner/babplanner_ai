"""발주 추천 흐름."""

from __future__ import annotations

from babplanner_ai.procurement.estimator import ProcurementRecommender
from babplanner_ai.procurement.models import (
    AIProcurementRequest,
    AIProcurementResponse,
)


class ProcurementRecommendationService:
    def __init__(self, recommender: ProcurementRecommender) -> None:
        self._recommender = recommender

    def recommend(self, request: AIProcurementRequest) -> AIProcurementResponse:
        """한 건을 추천한다. 아무것도 저장하지 않는다."""
        recommendations, model_version = self._recommender.recommend(request)
        return AIProcurementResponse(
            requestId=request.requestId,
            recommendations=recommendations,
            modelVersion=model_version,
        )
