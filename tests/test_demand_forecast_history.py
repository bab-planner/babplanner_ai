"""식수 예측의 과거 feature.

가장 중요한 두 가지를 본다. 예측일 당일과 그 이후 기록이 절대 섞이지 않는지,
그리고 학습 데이터의 lag 컬럼과 같은 규칙으로 계산하는지.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import pytest

from babplanner_ai.config import DEFAULT_DEMAND_CCTV_LOG_PATH, DEFAULT_DEMAND_HISTORY_PATH
from babplanner_ai.demand_forecast.history import (
    CctvRecord,
    FutureRecordError,
    HistoryUnavailableError,
    InsufficientHistoryError,
    MealRecord,
    SyntheticDemandHistoryRepository,
    compute_history_features,
)


def weekdays(start: dt.date, count: int) -> list[dt.date]:
    days, day = [], start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


def history(days: list[dt.date]) -> tuple[list[MealRecord], list[CctvRecord]]:
    """식수 = 300 + 순번, CCTV 평균 = 순번 / 10. 어느 기록을 골랐는지 값으로 알 수 있다."""
    meals = [MealRecord(d, 300 + i) for i, d in enumerate(days)]
    cctv = [CctvRecord(d, i / 10, i / 10 + 20) for i, d in enumerate(days)]
    return meals, cctv


# 2026-08-31(월)부터 10 평일: 8/31 ~ 9/11. 예측일은 9/14(월).
DAYS = weekdays(dt.date(2026, 8, 31), 10)
TARGET = dt.date(2026, 9, 14)


class TestCompute:
    def test_lag_1_is_the_previous_operating_day(self) -> None:
        meals, cctv = history(DAYS)

        features = compute_history_features(meals, cctv, TARGET)

        # 월요일의 직전 운영일은 금요일(9/11, 순번 9)이다.
        assert features.lag_1 == 309
        assert features.meal_history_last_date == dt.date(2026, 9, 11)

    def test_lag_7_is_the_last_same_weekday(self) -> None:
        meals, cctv = history(DAYS)

        features = compute_history_features(meals, cctv, TARGET)

        # 직전 월요일은 9/7(순번 5).
        assert features.lag_7 == 305

    def test_rolling_mean_uses_the_last_seven_operating_days(self) -> None:
        meals, cctv = history(DAYS)

        features = compute_history_features(meals, cctv, TARGET)

        assert features.rolling_mean_7 == pytest.approx(sum(300 + i for i in range(3, 10)) / 7)

    def test_cctv_features_come_from_the_previous_day_and_weekday(self) -> None:
        meals, cctv = history(DAYS)

        features = compute_history_features(meals, cctv, TARGET)

        assert features.prev_avg_occupancy == pytest.approx(0.9)
        assert features.prev_peak_occupancy == pytest.approx(20.9)
        assert features.same_weekday_prev_occupancy == pytest.approx(0.5)

    def test_input_order_does_not_matter(self) -> None:
        meals, cctv = history(DAYS)

        shuffled = compute_history_features(meals[::-1], cctv[::-1], TARGET)

        assert shuffled == compute_history_features(meals, cctv, TARGET)

    def test_too_little_history_is_refused(self) -> None:
        meals, cctv = history(DAYS[:6])

        with pytest.raises(InsufficientHistoryError):
            compute_history_features(meals, cctv, dt.date(2026, 9, 8))

    def test_missing_same_weekday_is_refused(self) -> None:
        # 화~금만 두 주. 월요일 기록이 없다.
        days = [d for d in weekdays(dt.date(2026, 8, 31), 10) if d.weekday() != 0]
        meals, cctv = history(days)

        with pytest.raises(InsufficientHistoryError):
            compute_history_features(meals, cctv, TARGET)


class TestNoLeakage:
    def test_a_record_on_the_forecast_date_is_refused(self) -> None:
        meals, cctv = history(DAYS)
        meals.append(MealRecord(TARGET, 999))

        # 걸러서 쓰지 않는다. 걸러내면 저장소의 버그가 드러나지 않는다.
        with pytest.raises(FutureRecordError):
            compute_history_features(meals, cctv, TARGET)

    def test_a_future_cctv_record_is_refused(self) -> None:
        meals, cctv = history(DAYS)
        cctv.append(CctvRecord(TARGET + dt.timedelta(days=3), 99.0, 99.0))

        with pytest.raises(FutureRecordError):
            compute_history_features(meals, cctv, TARGET)

    def test_repository_never_returns_the_forecast_date_or_later(self, tmp_path: Path) -> None:
        repo = repository_with(tmp_path, weekdays(dt.date(2026, 8, 31), 20))

        assert all(r.date < TARGET for r in repo.meals_before(TARGET))
        assert all(r.date < TARGET for r in repo.cctv_before(TARGET))

    def test_future_values_do_not_change_the_features(self, tmp_path: Path) -> None:
        past = DAYS
        future = weekdays(TARGET, 10)
        truncated = repository_with(tmp_path / "past", past)
        # 예측일 이후에 말도 안 되는 값을 넣는다. 조금이라도 새면 결과가 바뀐다.
        with_future = repository_with(tmp_path / "all", past + future, future_value=99999)

        def features(repo: SyntheticDemandHistoryRepository):
            return compute_history_features(repo.meals_before(TARGET), repo.cctv_before(TARGET), TARGET)

        assert features(with_future) == features(truncated)


def repository_with(
    root: Path, days: list[dt.date], future_value: int | None = None
) -> SyntheticDemandHistoryRepository:
    root.mkdir(parents=True, exist_ok=True)
    meal_csv, cctv_csv = root / "meals.csv", root / "cctv.csv"
    with meal_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "weekday", "meal_count"])
        for i, d in enumerate(days):
            value = future_value if future_value is not None and d >= TARGET else 300 + i
            writer.writerow([d.isoformat(), "x", value])
    with cctv_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "avg_occupancy", "peak_occupancy"])
        for i, d in enumerate(days):
            value = future_value if future_value is not None and d >= TARGET else i / 10
            writer.writerow([d.isoformat(), value, value + 20])
    return SyntheticDemandHistoryRepository(meal_csv, cctv_csv)


class TestRepository:
    def test_missing_file_is_reported_without_the_path(self, tmp_path: Path) -> None:
        repo = SyntheticDemandHistoryRepository(tmp_path / "nope.csv", tmp_path / "nope2.csv")

        with pytest.raises(HistoryUnavailableError) as failure:
            repo.check_available()

        assert str(failure.value) == "demand history not found"
        assert str(tmp_path) not in str(failure.value)

    def test_missing_column_is_reported(self, tmp_path: Path) -> None:
        (tmp_path / "m.csv").write_text("date,count\n2026-09-01,1\n", encoding="utf-8")
        (tmp_path / "c.csv").write_text("date,avg_occupancy,peak_occupancy\n", encoding="utf-8")

        with pytest.raises(HistoryUnavailableError, match="missing required columns"):
            SyntheticDemandHistoryRepository(tmp_path / "m.csv", tmp_path / "c.csv").check_available()

    def test_malformed_value_is_reported(self, tmp_path: Path) -> None:
        (tmp_path / "m.csv").write_text("date,meal_count\n2026-09-01,many\n", encoding="utf-8")
        (tmp_path / "c.csv").write_text("date,avg_occupancy,peak_occupancy\n", encoding="utf-8")

        with pytest.raises(HistoryUnavailableError, match="malformed"):
            SyntheticDemandHistoryRepository(tmp_path / "m.csv", tmp_path / "c.csv").check_available()

    def test_files_are_read_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        repo = repository_with(tmp_path, DAYS)
        reads = []
        original = repo._read_meals
        monkeypatch.setattr(repo, "_read_meals", lambda: reads.append(1) or original())

        for _ in range(3):
            repo.meals_before(TARGET)
            repo.cctv_before(TARGET)

        assert len(reads) == 1


needs_synthetic_history = pytest.mark.skipif(
    not (DEFAULT_DEMAND_HISTORY_PATH.is_file() and DEFAULT_DEMAND_CCTV_LOG_PATH.is_file()),
    reason="합성 이력이 없다. research/demand_forecasting 에서 generate_synthetic_data.py 를 먼저 실행한다.",
)


@needs_synthetic_history
def test_matches_the_training_data_on_every_row() -> None:
    """학습 CSV의 lag·CCTV 컬럼을 서빙 쪽 계산이 전 행에서 그대로 재현한다.

    학습 CSV의 컬럼은 research 쪽이 shift/rolling으로 만들고 history_features()로
    검증한 값이다. 여기서 한 행이라도 다르면, 모델은 서빙 때 학습 때와 다른
    입력을 받는다.
    """
    repo = SyntheticDemandHistoryRepository(DEFAULT_DEMAND_HISTORY_PATH, DEFAULT_DEMAND_CCTV_LOG_PATH)
    with DEFAULT_DEMAND_HISTORY_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    # CSV는 lag 생성 때문에 앞 7일이 빠져 있다. 그 7일을 이력으로 쓰는 처음 행들은
    # CSV만으로 재현할 수 없으므로 제외한다.
    checked = 0
    for row in rows[7:]:
        target = dt.date.fromisoformat(row["date"])
        f = compute_history_features(repo.meals_before(target), repo.cctv_before(target), target)

        assert f.lag_1 == float(row["lag_1"]), row["date"]
        assert f.lag_7 == float(row["lag_7"]), row["date"]
        assert f.rolling_mean_7 == pytest.approx(float(row["rolling_mean_7"]), abs=0.006), row["date"]
        assert f.prev_avg_occupancy == pytest.approx(float(row["prev_avg_occupancy"])), row["date"]
        assert f.prev_peak_occupancy == pytest.approx(float(row["prev_peak_occupancy"])), row["date"]
        assert f.same_weekday_prev_occupancy == pytest.approx(
            float(row["same_weekday_prev_occupancy"])
        ), row["date"]
        checked += 1

    assert checked == len(rows) - 7 > 500
