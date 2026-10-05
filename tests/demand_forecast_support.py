"""식수 예측 테스트용 가짜 부품.

실제 모델 대신 쓰는 예측기, 임시 artifact, 메모리 이력이다. 모두 테스트용
합성 값이다. 실제 artifact를 쓰는 smoke test는 test_demand_forecast_api.py 에 따로 있다.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from babplanner_ai.demand_forecast.artifact import ModelArtifactLoader
from babplanner_ai.demand_forecast.features import BUILDABLE_FEATURES
from babplanner_ai.demand_forecast.history import CctvRecord, MealRecord
from babplanner_ai.demand_forecast.service import DemandForecastService

#: research가 저장하는 순서와 같은 feature 목록.
FEATURES = [
    "lag_1", "lag_7", "rolling_mean_7", "weekday", "semester", "semester_week",
    "exam_period", "holiday", "campus_event", "temperature", "rainfall", "humidity",
    "menu_category", "main_menu", "special_menu",
    "prev_avg_occupancy", "prev_peak_occupancy", "same_weekday_prev_occupancy",
]

FAKE_MODEL_BYTES = b"not-a-real-model; the fake predictor never unpickles this"


def metadata(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "synthetic_data": True,
        "model_name": "fake_model",
        "target": "meal_count",
        "features": FEATURES,
        "training_date": "2026-09-12T00:00:00",
        "training_period": ["2023-03-13", "2026-09-11"],
        "categorical_levels": {"main_menu": ["돈까스", "제육볶음"]},
        "versions": {},
        "model_sha256": hashlib.sha256(FAKE_MODEL_BYTES).hexdigest(),
    }
    body.update(overrides)
    return body


def write_artifact(root: Path, meta: dict[str, object] | None = None) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    model_path, metadata_path = root / "model.joblib", root / "metadata.json"
    model_path.write_bytes(FAKE_MODEL_BYTES)
    metadata_path.write_text(json.dumps(meta or metadata(), ensure_ascii=False), encoding="utf-8")
    return model_path, metadata_path


class FakePredictor:
    """정해진 값을 돌려주고, 받은 입력을 기록한다."""

    def __init__(self, value: float = 428.0) -> None:
        self.value = value
        self.calls: list[tuple[dict[str, object], list[str]]] = []

    def predict_one(self, row: Mapping[str, object], feature_order: Sequence[str]) -> float:
        self.calls.append((dict(row), list(feature_order)))
        return self.value


class CountingPredictorFactory:
    def __init__(self, predictor: FakePredictor) -> None:
        self.predictor = predictor
        self.loads = 0

    def __call__(self, path: Path, features: Sequence[str]) -> FakePredictor:
        self.loads += 1
        return self.predictor


def fake_loader(
    root: Path,
    predictor: FakePredictor | None = None,
    meta: dict[str, object] | None = None,
    allow_synthetic: bool = True,
) -> tuple[ModelArtifactLoader, CountingPredictorFactory]:
    model_path, metadata_path = write_artifact(root, meta)
    factory = CountingPredictorFactory(predictor or FakePredictor())
    loader = ModelArtifactLoader(
        model_path,
        metadata_path,
        buildable_features=BUILDABLE_FEATURES,
        allow_synthetic=allow_synthetic,
        load_predictor=factory,
        version_lookup=lambda package: None,
    )
    return loader, factory


def weekdays(start: dt.date, count: int) -> list[dt.date]:
    days, day = [], start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


class InMemoryHistory:
    """예측일 이전 기록만 돌려주는 메모리 이력. leaky=True면 일부러 전부 돌려준다."""

    def __init__(self, days: list[dt.date], leaky: bool = False) -> None:
        self.meals = [MealRecord(d, 380 + i) for i, d in enumerate(days)]
        self.cctv = [CctvRecord(d, 40 + i / 10, 70 + i / 10) for i, d in enumerate(days)]
        self.leaky = leaky

    def meals_before(self, target: dt.date) -> list[MealRecord]:
        return list(self.meals) if self.leaky else [r for r in self.meals if r.date < target]

    def cctv_before(self, target: dt.date) -> list[CctvRecord]:
        return list(self.cctv) if self.leaky else [r for r in self.cctv if r.date < target]

    def check_available(self) -> None:
        return None


#: 2026-08-31(월) ~ 2026-09-14(월) 11 평일. 9/15(화) 예측의 직전 운영일은 9/14다.
HISTORY_DAYS = weekdays(dt.date(2026, 8, 31), 11)


def fake_service(root: Path, predictor: FakePredictor | None = None, **kwargs) -> DemandForecastService:
    loader, _ = fake_loader(root, predictor, **kwargs)
    return DemandForecastService(loader, InMemoryHistory(HISTORY_DAYS))
