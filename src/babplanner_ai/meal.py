"""식사 구분.

식수 예측 계약이 쓰는 값이다. 서비스 백엔드의 MealType과 같은 문자열을 쓴다.
"""

from __future__ import annotations

from enum import Enum


class MealType(str, Enum):
    """OpenAPI MealType."""

    BREAKFAST = "BREAKFAST"
    LUNCH = "LUNCH"
    DINNER = "DINNER"
