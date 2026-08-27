"""발주 추천기 경계.

라우트 안에서 수량을 계산하지 않는다. 실제 모델이 붙을 때 바뀌는 것은 이
인터페이스의 구현 하나뿐이다.
"""

from __future__ import annotations

from typing import Protocol

from babplanner_ai.procurement.models import (
    AIProcurementRequest,
    ProcurementRecommendation,
)


class ProcurementEstimateError(RuntimeError):
    """추천 자체를 수행하지 못했다. 호출자에게 HTTP 오류로 나간다."""


class ProcurementRecommender(Protocol):
    def recommend(
        self, request: AIProcurementRequest
    ) -> tuple[list[ProcurementRecommendation], str]:
        """추천 목록과 모델 버전을 돌려준다."""
        ...
