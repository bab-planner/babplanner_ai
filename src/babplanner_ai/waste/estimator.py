"""분석기 경계.

HTTP 계층과 실제 추론을 분리한다. 나중에 실제 모델을 붙일 때 바꾸는 것은
이 인터페이스의 구현 하나뿐이다.
"""

from __future__ import annotations

from typing import Protocol

from babplanner_ai.waste.models import WasteEstimate


class WasteEstimateError(RuntimeError):
    """분석 자체를 수행하지 못했다.

    입력을 승인하지 않은 것(REJECTED)과 구분한다. 이쪽은 호출자에게
    HTTP 오류로 나간다.
    """


class WasteEstimator(Protocol):
    def estimate(self, image: bytes, menu_id: int) -> WasteEstimate:
        """이미지 한 장을 판정한다."""
        ...
