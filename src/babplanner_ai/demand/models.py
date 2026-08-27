"""수요 예측 계약 모델.

OpenAPI `AIDemandRequest` / `AIDemandResponse` 와 1:1로 대응한다.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from babplanner_ai.meal import MealType


class AIDemandRequest(BaseModel):
    """예측 요청.

    받는 것은 "무엇을 예측할지"뿐이다. 과거 관측치 배열을 받지 않는다.
    어느 기간의 무엇을 집계한 값인지 정의된 적이 없고, 정의되지 않은 배열은
    보내는 쪽과 받는 쪽이 서로 다른 뜻으로 채우게 된다.

    사용자 식별 정보는 들어오지 않는다. 정원 예측에 필요하지 않다.
    """

    model_config = ConfigDict(extra="forbid")

    requestId: str = Field(min_length=1)
    siteId: int = Field(ge=1)
    serviceDate: date
    mealType: MealType


class DemandPredictionResult(BaseModel):
    """예측기가 돌려주는 값."""

    predicted_people: int = Field(ge=0)
    model_version: str = Field(min_length=1)


class AIDemandResponse(BaseModel):
    """OpenAPI AIDemandResponse."""

    requestId: str
    predictedPeople: int = Field(ge=0)
    modelVersion: str

    @classmethod
    def of(
        cls, request_id: str, result: DemandPredictionResult
    ) -> "AIDemandResponse":
        return cls(
            requestId=request_id,
            predictedPeople=result.predicted_people,
            modelVersion=result.model_version,
        )
