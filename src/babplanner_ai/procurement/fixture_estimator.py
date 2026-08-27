"""개발용 발주 추천기.

**아무것도 추천하지 않는다.** 들어온 계획 수량을 그대로 돌려주고 조정률은
0이다. 프로세스 경계가 실제로 연결됐는지 확인하는 용도다.

"20% 적게 사라" 같은 값을 만들지 않는 이유는 분명하다. 그런 숫자가 화면에
뜨면 관리자는 그대로 발주를 넣는다. 재고도 단가도 모르는 구현이 낼 수 있는
정직한 답은 "조정하지 않음"뿐이다.
"""

from __future__ import annotations

from babplanner_ai.procurement.models import (
    AIProcurementRequest,
    ProcurementRecommendation,
)

#: 조정하지 않았다는 사실을 사람이 읽을 수 있게 남긴다.
FIXTURE_REASON = "개발용 fixture입니다. 실제 추천이 아니며 계획 수량을 그대로 둡니다."

#: 저장된 추천만 보고도 어느 구현이 낸 값인지 알 수 있어야 한다.
FIXTURE_MODEL_VERSION = "fastapi-fixture-procurement-v1"


class FixtureProcurementRecommender:
    """계획 수량을 그대로 돌려준다. 순서도 그대로다."""

    def recommend(
        self, request: AIProcurementRequest
    ) -> tuple[list[ProcurementRecommendation], str]:
        recommendations = [
            ProcurementRecommendation(
                ingredientName=planned.ingredientName,
                unit=planned.unit,
                plannedQuantity=planned.plannedQuantity,
                recommendedQuantity=planned.plannedQuantity,
                adjustmentRate=0.0,
                reason=FIXTURE_REASON,
            )
            for planned in request.plannedIngredients
        ]
        return recommendations, FIXTURE_MODEL_VERSION
