"""식사 구분.

수요 예측과 발주 추천이 같은 값을 쓴다. 두 곳에 따로 적어 두면 한쪽만
늘어났을 때 서비스 백엔드와 조용히 어긋난다.
"""

from __future__ import annotations

from enum import Enum


class MealType(str, Enum):
    """OpenAPI MealType."""

    BREAKFAST = "BREAKFAST"
    LUNCH = "LUNCH"
    DINNER = "DINNER"
