"""수요 예측 흐름.

라우트는 요청을 받고 응답을 만드는 일만 한다. 예측 호출은 여기 있다.
"""

from __future__ import annotations

from babplanner_ai.demand.estimator import DemandEstimator
from babplanner_ai.demand.models import AIDemandRequest, AIDemandResponse


class DemandPredictionService:
    def __init__(self, estimator: DemandEstimator) -> None:
        self._estimator = estimator

    def predict(self, request: AIDemandRequest) -> AIDemandResponse:
        """한 건을 예측한다.

        아무것도 저장하지 않는다. 예측 이력의 보관은 서비스 백엔드가 맡는다.
        """
        result = self._estimator.predict(
            request.siteId, request.serviceDate, request.mealType
        )
        # requestId는 받은 값을 그대로 돌려준다. 호출자가 다른 요청의 결과를
        # 자기 것으로 저장하지 않으려면 이 값이 어긋나선 안 된다.
        return AIDemandResponse.of(request.requestId, result)
