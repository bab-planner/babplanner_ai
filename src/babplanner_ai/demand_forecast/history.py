"""과거 식수·CCTV 이력과, 거기서 만드는 과거 feature.

**예측일 당일과 그 이후 기록은 절대 쓰지 않는다.** 저장소는 예측일 이전
기록만 돌려주고, feature 계산은 받은 기록에 예측일 이후 것이 섞여 있으면
조용히 걸러내지 않고 실패한다. 걸러내면 저장소의 버그가 영영 드러나지 않는다.

계산 규칙은 research/demand_forecasting/src/common.py 의 history_features()와
같다. 학습 데이터의 lag 컬럼이 그 규칙으로 만들어졌으므로, 여기서 다르게
계산하면 모델은 학습 때와 다른 입력을 받는다. 두 구현이 같은 값을 내는지는
tests/test_demand_forecast_history.py 가 합성 데이터 전 행으로 대조한다.

이 모듈은 pandas를 쓰지 않는다. 이력 계산은 표준 라이브러리로 충분하고,
ML 런타임이 없는 환경에서도 이 부분은 검증되어야 한다.
"""

from __future__ import annotations

import csv
import datetime as dt
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

#: rolling_mean_7 에 필요한 과거 운영일 수.
ROLLING_WINDOW = 7


class HistoryUnavailableError(RuntimeError):
    """이력 저장소 자체를 쓸 수 없다(파일 없음, 형식 오류). 서버 쪽 문제다."""


class InsufficientHistoryError(ValueError):
    """예측일 이전 기록이 feature를 만들기에 모자란다. 요청한 날짜의 문제다."""


class FutureRecordError(AssertionError):
    """예측일 당일이나 이후 기록이 feature 계산에 들어왔다. leakage다."""


@dataclass(frozen=True)
class MealRecord:
    date: dt.date
    meal_count: int


@dataclass(frozen=True)
class CctvRecord:
    date: dt.date
    avg_occupancy: float
    peak_occupancy: float


@dataclass(frozen=True)
class HistoryFeatures:
    lag_1: float
    lag_7: float
    rolling_mean_7: float
    prev_avg_occupancy: float
    prev_peak_occupancy: float
    same_weekday_prev_occupancy: float
    meal_history_last_date: dt.date
    cctv_history_last_date: dt.date

    def as_model_inputs(self) -> dict[str, float]:
        return {
            "lag_1": self.lag_1,
            "lag_7": self.lag_7,
            "rolling_mean_7": self.rolling_mean_7,
            "prev_avg_occupancy": self.prev_avg_occupancy,
            "prev_peak_occupancy": self.prev_peak_occupancy,
            "same_weekday_prev_occupancy": self.same_weekday_prev_occupancy,
        }


def compute_history_features(
    meals: Sequence[MealRecord], cctv: Sequence[CctvRecord], target: dt.date
) -> HistoryFeatures:
    """예측일 이전 기록으로 과거 feature를 만든다.

    - lag_1: 직전 운영일 식수
    - lag_7: 직전 동일 요일 식수 (학기 중에는 7일 전)
    - rolling_mean_7: 직전 7개 운영일 평균 식수
    - prev_avg_occupancy / prev_peak_occupancy: 직전 운영일 CCTV 평균/최대 동시 인원
    - same_weekday_prev_occupancy: 직전 동일 요일 CCTV 평균 동시 인원
    """
    for record in (*meals, *cctv):
        if record.date >= target:
            raise FutureRecordError(
                f"예측일({target}) 당일 또는 이후 기록({record.date})이 들어왔습니다."
            )

    past_meals = sorted(meals, key=lambda r: r.date)
    past_cctv = sorted(cctv, key=lambda r: r.date)
    weekday = target.weekday()
    same_day_meals = [r for r in past_meals if r.date.weekday() == weekday]
    same_day_cctv = [r for r in past_cctv if r.date.weekday() == weekday]

    if len(past_meals) < ROLLING_WINDOW:
        raise InsufficientHistoryError(
            f"{target} 이전 식수 기록이 {len(past_meals)}일뿐입니다. 최소 {ROLLING_WINDOW}일이 필요합니다."
        )
    if not same_day_meals or not past_cctv or not same_day_cctv:
        raise InsufficientHistoryError(f"{target} 이전의 동일 요일 식수 또는 CCTV 기록이 없습니다.")

    window = past_meals[-ROLLING_WINDOW:]
    return HistoryFeatures(
        lag_1=float(past_meals[-1].meal_count),
        lag_7=float(same_day_meals[-1].meal_count),
        rolling_mean_7=sum(r.meal_count for r in window) / ROLLING_WINDOW,
        prev_avg_occupancy=past_cctv[-1].avg_occupancy,
        prev_peak_occupancy=past_cctv[-1].peak_occupancy,
        same_weekday_prev_occupancy=same_day_cctv[-1].avg_occupancy,
        meal_history_last_date=past_meals[-1].date,
        cctv_history_last_date=past_cctv[-1].date,
    )


class DemandHistoryRepository(Protocol):
    """과거 식수와 CCTV 이력의 출처.

    지금은 합성 CSV다. 실제 운영에서는 DB나 서비스 백엔드가 준 이력으로 바꾼다.
    어느 구현이든 **예측일 이전(date < target) 기록만** 돌려준다.
    """

    def meals_before(self, target: dt.date) -> Sequence[MealRecord]: ...

    def cctv_before(self, target: dt.date) -> Sequence[CctvRecord]: ...

    def check_available(self) -> None:
        """쓸 수 없으면 HistoryUnavailableError."""
        ...


class SyntheticDemandHistoryRepository:
    """research/demand_forecasting 이 만든 합성 CSV에서 이력을 읽는다.

    처음 쓸 때 한 번 읽어 메모리에 둔다. 파일을 고치면 서버를 다시 띄운다.
    """

    def __init__(self, meal_csv: Path, cctv_csv: Path) -> None:
        self._meal_csv = meal_csv
        self._cctv_csv = cctv_csv
        self._lock = threading.Lock()
        self._loaded: tuple[tuple[MealRecord, ...], tuple[CctvRecord, ...]] | None = None
        self._error: str | None = None

    def meals_before(self, target: dt.date) -> Sequence[MealRecord]:
        meals, _ = self._records()
        return [r for r in meals if r.date < target]

    def cctv_before(self, target: dt.date) -> Sequence[CctvRecord]:
        _, cctv = self._records()
        return [r for r in cctv if r.date < target]

    def check_available(self) -> None:
        self._records()

    def _records(self) -> tuple[tuple[MealRecord, ...], tuple[CctvRecord, ...]]:
        with self._lock:
            if self._loaded is None and self._error is None:
                try:
                    self._loaded = (self._read_meals(), self._read_cctv())
                except HistoryUnavailableError as error:
                    self._error = str(error)
            if self._error is not None:
                raise HistoryUnavailableError(self._error)
            assert self._loaded is not None
            return self._loaded

    def _read_meals(self) -> tuple[MealRecord, ...]:
        rows = _read_csv(self._meal_csv, {"date", "meal_count"}, "demand history")
        try:
            return tuple(
                MealRecord(dt.date.fromisoformat(r["date"]), int(r["meal_count"])) for r in rows
            )
        except ValueError as error:
            raise HistoryUnavailableError("demand history is malformed") from error

    def _read_cctv(self) -> tuple[CctvRecord, ...]:
        rows = _read_csv(self._cctv_csv, {"date", "avg_occupancy", "peak_occupancy"}, "cctv history")
        try:
            return tuple(
                CctvRecord(
                    dt.date.fromisoformat(r["date"]),
                    float(r["avg_occupancy"]),
                    float(r["peak_occupancy"]),
                )
                for r in rows
            )
        except ValueError as error:
            raise HistoryUnavailableError("cctv history is malformed") from error


def _read_csv(path: Path, required: set[str], label: str) -> list[dict[str, str]]:
    # 오류 메시지에 경로를 넣지 않는다. status 응답으로 나가는 문자열이다.
    if not path.is_file():
        raise HistoryUnavailableError(f"{label} not found")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or ()):
            raise HistoryUnavailableError(f"{label} is missing required columns")
        return list(reader)
