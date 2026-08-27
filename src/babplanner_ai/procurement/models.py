"""발주 추천 계약 모델.

OpenAPI `AIProcurementRequest` / `AIProcurementResponse` 와 1:1로 대응한다.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from babplanner_ai.meal import MealType


class PlannedIngredient(BaseModel):
    """계획된 식재료 한 줄.

    수량과 단위를 분리해서 받는다. "10kg" 같은 문자열은 파싱하는 쪽마다
    다르게 읽히고, 단위를 바꾸려면 문자열을 다시 뜯어야 한다.
    """

    model_config = ConfigDict(extra="forbid")

    ingredientName: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    plannedQuantity: float = Field(gt=0)


class AIProcurementRequest(BaseModel):
    """추천 요청.

    무엇을 얼마나 살 계획인지는 호출자가 명시한다. 여기서 식단 이름을 보고
    식재료를 추측하지 않는다. 메뉴 구성 항목은 조리된 음식 이름일 수 있고
    구매 단위도 수량도 없다.

    과거 잔반 집계 배열은 받지 않는다. 어느 기간의 무엇을 집계한 값인지
    정의된 적이 없기 때문이다. 실제 모델이 요구하게 되면 그때 구조를 정해
    추가한다.
    """

    model_config = ConfigDict(extra="forbid")

    requestId: str = Field(min_length=1)
    siteId: int = Field(ge=1)
    serviceDate: date
    mealType: MealType
    predictedPeople: int = Field(ge=0)
    plannedIngredients: list[PlannedIngredient] = Field(min_length=1)


class ProcurementRecommendation(BaseModel):
    """OpenAPI ProcurementItem."""

    ingredientName: str
    unit: str
    plannedQuantity: float
    recommendedQuantity: float = Field(ge=0)
    adjustmentRate: float | None = None
    reason: str | None = None


class AIProcurementResponse(BaseModel):
    """OpenAPI AIProcurementResponse.

    추천은 요청에 들어온 식재료 순서를 그대로 유지한다. 호출자가 입력과
    출력을 자리로 맞출 수 있어야 한다.
    """

    requestId: str
    recommendations: list[ProcurementRecommendation]
    modelVersion: str
