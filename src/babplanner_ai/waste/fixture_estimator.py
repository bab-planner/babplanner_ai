"""개발용 분석기.

**돌려주는 값은 개발 fixture이지 분석 결과가 아니다.** 이미지 내용을
판독하지 않는다. 이 구현이 운영에서 선택되지 않도록 설정 단계에서 막는다.
"""

from __future__ import annotations

from babplanner_ai.waste.models import WasteAnalysisDecision, WasteEstimate

FIXTURE_WASTE_RATE = 0.15
FIXTURE_CONFIDENCE = 0.92

#: 서비스 백엔드에도 이름이 같은 개발용 분석기가 있다. 두 값이 같으면
#: 저장된 결과만 보고 어느 쪽을 거쳐 왔는지 알 수 없어, 설정이 잘못돼
#: HTTP 경로가 아닌 백엔드 내부 fixture를 쓰고 있어도 알아차리지 못한다.
#: 그래서 어느 프로세스가 낸 값인지 이름에 드러낸다.
#:
#: 실제 모델이 생기면 이 값은 모델 release 식별자로 바뀐다.
FIXTURE_MODEL_VERSION = "fastapi-fixture-waste-v1"


class FixtureWasteEstimator:
    """디코딩된 이미지에 대해 늘 같은 값을 돌려준다."""

    def estimate(self, image: bytes, menu_id: int) -> WasteEstimate:
        return WasteEstimate(
            decision=WasteAnalysisDecision.APPROVED,
            waste_rate=FIXTURE_WASTE_RATE,
            confidence=FIXTURE_CONFIDENCE,
            model_version=FIXTURE_MODEL_VERSION,
        )
