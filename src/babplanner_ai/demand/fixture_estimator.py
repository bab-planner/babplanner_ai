"""개발용 수요 예측기.

**돌려주는 값은 개발 fixture이지 예측이 아니다.** 과거 데이터를 보지 않고,
어떤 입력이 와도 같은 수를 돌려준다. 프로세스 경계가 실제로 연결됐는지
확인하는 용도다.

값을 입력에 따라 그럴듯하게 흔들지 않는다. 흔들면 화면에서 예측처럼 보이고,
누군가 그 숫자로 식재료를 주문하게 된다.
"""

from __future__ import annotations

from datetime import date

from babplanner_ai.demand.models import DemandPredictionResult
from babplanner_ai.meal import MealType

#: 고정 반환값. 예측 결과가 아니다.
FIXTURE_PREDICTED_PEOPLE = 100

#: 저장된 예측만 보고도 어느 구현이 낸 값인지 알 수 있어야 한다.
#: 실제 모델이 생기면 이 값은 모델 release 식별자로 바뀐다.
FIXTURE_MODEL_VERSION = "fastapi-fixture-demand-v1"


class FixtureDemandEstimator:
    """입력과 무관하게 같은 값을 돌려준다."""

    def predict(
        self, site_id: int, service_date: date, meal_type: MealType
    ) -> DemandPredictionResult:
        return DemandPredictionResult(
            predicted_people=FIXTURE_PREDICTED_PEOPLE,
            model_version=FIXTURE_MODEL_VERSION,
        )
