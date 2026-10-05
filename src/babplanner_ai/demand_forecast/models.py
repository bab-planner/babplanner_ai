"""식수 예측(demand-forecast) 계약 모델.

기존 `/internal/v1/ai/demand/predict`(siteId·serviceDate·mealType만 받는 계약)와
별개의 계약이다. 이쪽은 날씨·학사·메뉴 조건을 받아 저장된 모델로 다음 날 중식
식수를 예측한다. 과거 식수와 과거 CCTV 값은 받지 않는다. AI 서버가 자기 이력
저장소에서 예측일 **이전** 기록만으로 만든다. 호출자가 계산 방식을 알 필요가
없고, 호출자마다 다르게 계산할 여지도 없다.

필드 이름은 서비스 백엔드(Spring)의 JSON 관례를 따라 camelCase다.
"""

from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from babplanner_ai.meal import MealType

DISCLAIMER_SYNTHETIC = "합성데이터 기반 PoC 예측값이며 실제 급식 운영 정확도를 의미하지 않습니다."
DISCLAIMER_REAL = "모델 예측값이며 실제 식수와 다를 수 있습니다."


class MenuCategory(str, Enum):
    KOREAN = "한식"
    CHINESE = "중식"
    WESTERN = "양식"
    JAPANESE = "일식"
    SNACK = "분식"
    OTHER = "기타"


MenuName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50)]


class DemandForecastRequest(BaseModel):
    """다음 날 중식 식수 예측 요청. 모든 필드가 필수다.

    기본값을 두지 않는다. 호출자가 빠뜨린 값을 서버가 조용히 채우면, 호출자는
    자기가 보낸 조건으로 예측했다고 믿는다.

    요일은 받지 않는다. date에서 계산한다. 둘을 따로 받으면 어긋날 수 있다.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "date": "2026-09-15",
                "semester": 2,
                "semesterWeek": 3,
                "examPeriod": False,
                "holiday": False,
                "campusEvent": False,
                "temperature": 24.0,
                "rainfall": 0.0,
                "humidity": 60.0,
                "menuCategory": "양식",
                "mainMenu": "돈까스",
                "specialMenu": False,
            }
        },
    )

    date: dt.date = Field(description="예측일(식사일). 학기 중 평일")
    semester: int = Field(ge=1, le=2, description="1 또는 2")
    semesterWeek: int = Field(ge=1, le=16, description="학기 주차 1~16")
    examPeriod: bool
    holiday: bool
    campusEvent: bool
    temperature: float = Field(ge=-30, le=45, allow_inf_nan=False, description="D-1 예보 기온(℃)")
    rainfall: float = Field(ge=0, le=500, allow_inf_nan=False, description="D-1 예보 강수량(mm)")
    humidity: float = Field(ge=0, le=100, allow_inf_nan=False, description="D-1 예보 습도(%)")
    menuCategory: MenuCategory
    mainMenu: MenuName
    specialMenu: bool

    @field_validator("date")
    @classmethod
    def _weekday_only(cls, value: dt.date) -> dt.date:
        # 모델은 학기 중 평일만 보고 학습했다. 주말에 숫자를 내면 근거 없는 값이다.
        if value.weekday() >= 5:
            raise ValueError("주말은 예측 대상이 아닙니다. 학기 중 평일만 예측합니다.")
        return value


class ForecastReference(BaseModel):
    """예측에 쓴 과거 식수. 예측일 이전 기록에서만 나온 값이다."""

    previousDay: int = Field(description="직전 운영일 식수 (lag_1)")
    sameWeekdayPreviousWeek: int = Field(description="직전 동일 요일 식수 (lag_7)")
    rolling7DayAverage: int = Field(description="직전 7개 운영일 평균 식수 (rolling_mean_7)")
    historyAsOf: dt.date = Field(description="사용한 이력의 마지막 날짜")


class ForecastConditions(BaseModel):
    weekday: str
    semester: int
    semesterWeek: int
    examPeriod: bool
    holiday: bool
    campusEvent: bool
    menuCategory: MenuCategory
    mainMenu: str
    specialMenu: bool
    temperature: float
    rainfall: float
    humidity: float


class ForecastModelInfo(BaseModel):
    name: str
    version: str = Field(description="artifact 파일 SHA-256 앞 12자리. 같은 값이면 같은 모델이다")
    syntheticData: bool
    trainedAt: str | None = None


class DemandForecastResponse(BaseModel):
    forecastDate: dt.date
    mealType: MealType = MealType.LUNCH
    predictedMealCount: int = Field(ge=0)
    reference: ForecastReference
    conditions: ForecastConditions
    model: ForecastModelInfo
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str


class DemandForecastStatus(BaseModel):
    """모델과 이력을 쓸 수 있는지. 쓸 수 없으면 reason에 이유가 있다.

    reason에는 파일 경로를 담지 않는다. 배포 구조를 알려줄 이유가 없다.
    """

    available: bool
    modelLoaded: bool
    historyAvailable: bool | None = None
    modelName: str | None = None
    modelVersion: str | None = None
    target: str | None = None
    syntheticData: bool | None = None
    #: 모델 종류 표지. 합성데이터 모델이면 "SYNTHETIC_POC"다. 실데이터 모델이
    #: 생기면 "TRAINED"가 된다. 모델이 없으면 없다.
    modelKind: str | None = None
    trainingPeriod: list[dt.date] | None = None
    reason: str | None = None
    warnings: list[str] | None = None
