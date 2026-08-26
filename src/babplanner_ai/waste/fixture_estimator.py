"""개발용 분석기.

**돌려주는 값은 개발 fixture이지 분석 결과가 아니다.** 이미지 내용을
판독하지 않는다. 이 구현이 운영에서 선택되지 않도록 설정 단계에서 막는다.
"""

from __future__ import annotations

from babplanner_ai.waste.models import WasteAnalysisDecision, WasteEstimate

#: 서비스 백엔드의 개발용 분석기와 같은 값을 쓴다. 두 곳의 개발 결과가
#: 달라 보이면 어느 쪽을 거쳐 왔는지 헷갈린다.
FIXTURE_WASTE_RATE = 0.15
FIXTURE_CONFIDENCE = 0.92
FIXTURE_MODEL_VERSION = "fixture-waste-v1"


class FixtureWasteEstimator:
    """디코딩된 이미지에 대해 늘 같은 값을 돌려준다."""

    def estimate(self, image: bytes, menu_id: int) -> WasteEstimate:
        return WasteEstimate(
            decision=WasteAnalysisDecision.APPROVED,
            waste_rate=FIXTURE_WASTE_RATE,
            confidence=FIXTURE_CONFIDENCE,
            model_version=FIXTURE_MODEL_VERSION,
        )
