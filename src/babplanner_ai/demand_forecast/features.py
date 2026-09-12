"""요청과 과거 이력을 모델 입력 한 행으로 만든다.

컬럼 이름과 값의 표현은 학습 데이터(research/demand_forecasting 의 CSV)와 같다.
요일은 "월"~"금" 한 글자, 불리언은 0/1, 학기는 정수다. 표현이 하나라도 다르면
모델의 one-hot 인코더는 그 값을 처음 보는 값으로 취급하고 조용히 무시한다.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence

from babplanner_ai.demand_forecast.history import HistoryFeatures
from babplanner_ai.demand_forecast.models import DemandForecastRequest

#: 학습 데이터의 weekday 값. date.weekday() 순서.
WEEKDAY_CODES = ("월", "화", "수", "목", "금", "토", "일")

REQUEST_FEATURES = (
    "weekday",
    "semester",
    "semester_week",
    "exam_period",
    "holiday",
    "campus_event",
    "temperature",
    "rainfall",
    "humidity",
    "menu_category",
    "main_menu",
    "menu_popularity",
    "special_menu",
)
HISTORY_FEATURES = (
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "prev_avg_occupancy",
    "prev_peak_occupancy",
    "same_weekday_prev_occupancy",
)

#: 이 서비스가 만들 수 있는 feature 전부. metadata가 이 밖의 feature를 요구하는
#: 모델은 로드하지 않는다. 만들 수 없는 입력을 요구하는 모델이다.
BUILDABLE_FEATURES = frozenset(REQUEST_FEATURES + HISTORY_FEATURES)


class FeatureSchemaError(RuntimeError):
    """모델이 기대하는 컬럼과 만든 컬럼이 어긋났다."""


def weekday_code(day: dt.date) -> str:
    return WEEKDAY_CODES[day.weekday()]


def build_feature_row(
    request: DemandForecastRequest, history: HistoryFeatures
) -> dict[str, object]:
    return {
        "weekday": weekday_code(request.date),
        "semester": request.semester,
        "semester_week": request.semesterWeek,
        "exam_period": int(request.examPeriod),
        "holiday": int(request.holiday),
        "campus_event": int(request.campusEvent),
        "temperature": request.temperature,
        "rainfall": request.rainfall,
        "humidity": request.humidity,
        "menu_category": request.menuCategory.value,
        "main_menu": request.mainMenu,
        "menu_popularity": request.menuPopularity,
        "special_menu": int(request.specialMenu),
        **history.as_model_inputs(),
    }


def order_for_model(row: Mapping[str, object], feature_order: Sequence[str]) -> dict[str, object]:
    """metadata의 feature 순서대로 뽑는다. 빠진 컬럼이 있으면 예측하지 않는다."""
    missing = [name for name in feature_order if name not in row]
    if missing:
        raise FeatureSchemaError(f"모델 입력에 없는 feature: {missing}")
    ordered = {name: row[name] for name in feature_order}
    if list(ordered) != list(feature_order):
        raise FeatureSchemaError("모델 입력 컬럼 순서가 metadata와 다릅니다.")
    return ordered
