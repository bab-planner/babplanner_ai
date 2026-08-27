"""수요 예측기 경계.

HTTP 계층과 실제 추론을 분리한다. 실제 모델이 붙을 때 바뀌는 것은 이
인터페이스의 구현 하나뿐이다. 라우트 안에서 숫자를 만들지 않는다.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from babplanner_ai.demand.models import DemandPredictionResult
from babplanner_ai.meal import MealType


class DemandEstimateError(RuntimeError):
    """예측 자체를 수행하지 못했다. 호출자에게 HTTP 오류로 나간다."""


class DemandEstimator(Protocol):
    def predict(
        self, site_id: int, service_date: date, meal_type: MealType
    ) -> DemandPredictionResult:
        """한 끼의 예상 인원을 돌려준다."""
        ...
