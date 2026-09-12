"""식수 예측(demand-forecast) 라우트.

대부분은 가짜 예측기로 경계와 계약을 본다. 마지막 두 묶음은 저장된 실제
artifact를 쓴다. 하나는 기동 smoke test이고, 하나는 서빙 쪽이 만든 입력 행이
학습 데이터의 행과 같은지 예측값으로 대조한다.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from babplanner_ai.config import (
    DEFAULT_DEMAND_CCTV_LOG_PATH,
    DEFAULT_DEMAND_FORECAST_METADATA_PATH,
    DEFAULT_DEMAND_FORECAST_MODEL_PATH,
    DEFAULT_DEMAND_HISTORY_PATH,
)
from babplanner_ai.demand_forecast.api import get_forecast_service
from babplanner_ai.demand_forecast.artifact import ModelArtifactLoader
from babplanner_ai.demand_forecast.features import BUILDABLE_FEATURES
from babplanner_ai.demand_forecast.models import DISCLAIMER_SYNTHETIC
from babplanner_ai.demand_forecast.service import DemandForecastService
from tests.conftest import TEST_SERVICE_TOKEN
from tests.demand_forecast_support import (
    FEATURES,
    HISTORY_DAYS,
    FakePredictor,
    InMemoryHistory,
    fake_loader,
    fake_service,
    weekdays,
)

PREDICT = "/internal/v1/ai/demand-forecast/predict"
STATUS = "/internal/v1/ai/demand-forecast/status"
AUTH = {"X-Service-Token": TEST_SERVICE_TOKEN}


def body(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
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
        "menuPopularity": 0.9,
        "specialMenu": False,
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def use_service(client: TestClient) -> Iterator:
    """앱의 식수 예측 서비스를 갈아 끼운다. TestClient는 새로 만들지 않는다."""

    def install(service: DemandForecastService) -> DemandForecastService:
        client.app.dependency_overrides[get_forecast_service] = lambda: service
        return service

    yield install
    client.app.dependency_overrides.pop(get_forecast_service, None)


@pytest.fixture
def predictor(tmp_path: Path, use_service) -> FakePredictor:
    fake = FakePredictor(428.0)
    use_service(fake_service(tmp_path, fake))
    return fake


def test_app_imports() -> None:
    from babplanner_ai.main import app

    assert app.title


class TestHealth:
    def test_health_is_200_without_a_token(self, client: TestClient) -> None:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_stays_200_when_the_model_is_missing(
        self, client: TestClient, tmp_path: Path, use_service
    ) -> None:
        loader = ModelArtifactLoader(tmp_path / "x", tmp_path / "y",
                                     buildable_features=BUILDABLE_FEATURES, allow_synthetic=True)
        use_service(DemandForecastService(loader, InMemoryHistory(HISTORY_DAYS)))

        # 모델 파일 하나 때문에 오케스트레이터가 프로세스를 재시작하면 안 된다.
        assert client.get("/health").status_code == 200


class TestServiceToken:
    @pytest.mark.parametrize("method,path", [("get", STATUS), ("post", PREDICT)])
    def test_token_is_required(self, client: TestClient, predictor, method: str, path: str) -> None:
        kwargs = {"json": body()} if method == "post" else {}

        missing = getattr(client, method)(path, **kwargs)
        wrong = getattr(client, method)(path, headers={"X-Service-Token": "wrong-token"}, **kwargs)

        assert missing.status_code == wrong.status_code == 401
        assert missing.json() == wrong.json()

    def test_token_is_never_echoed(self, client: TestClient, predictor) -> None:
        assert TEST_SERVICE_TOKEN not in client.post(PREDICT, json=body(), headers=AUTH).text


class TestStatus:
    def test_available(self, client: TestClient, predictor) -> None:
        data = client.get(STATUS, headers=AUTH).json()

        assert data["available"] is True
        assert data["modelLoaded"] is True
        assert data["historyAvailable"] is True
        assert data["modelName"] == "fake_model"
        assert data["target"] == "meal_count"
        assert data["syntheticData"] is True
        assert "reason" not in data

    def test_missing_artifact(self, client: TestClient, tmp_path: Path, use_service) -> None:
        loader = ModelArtifactLoader(tmp_path / "x", tmp_path / "y",
                                     buildable_features=BUILDABLE_FEATURES, allow_synthetic=True)
        use_service(DemandForecastService(loader, InMemoryHistory(HISTORY_DAYS)))

        response = client.get(STATUS, headers=AUTH)

        # 서버는 살아 있고, 모델을 쓸 수 없다는 사실만 알린다.
        assert response.status_code == 200
        assert response.json() == {
            "available": False,
            "modelLoaded": False,
            "historyAvailable": True,
            "reason": "model artifact not found",
        }


class TestValidation:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("menuCategory", "퓨전"),
            ("menuCategory", ""),
            ("humidity", 101),
            ("humidity", -1),
            ("semester", 3),
            ("semester", 0),
            ("semesterWeek", 0),
            ("semesterWeek", 17),
            ("rainfall", -0.1),
            ("temperature", 80),
            ("menuPopularity", 1.2),
            ("mainMenu", "   "),
            ("date", "2026-13-45"),
        ],
    )
    def test_out_of_contract_values_are_rejected(
        self, client: TestClient, predictor, field: str, value: object
    ) -> None:
        response = client.post(PREDICT, json=body(**{field: value}), headers=AUTH)

        assert response.status_code == 422, (field, value)
        assert predictor.calls == []

    @pytest.mark.parametrize("field", list(body()))
    def test_every_field_is_required(self, client: TestClient, predictor, field: str) -> None:
        payload = body()
        del payload[field]

        assert client.post(PREDICT, json=payload, headers=AUTH).status_code == 422

    @pytest.mark.parametrize("field", ["weekday", "lag_1", "prevAvgOccupancy", "userId", "studentId"])
    def test_unknown_fields_are_refused(self, client: TestClient, predictor, field: str) -> None:
        # 과거 feature를 보내도 쓰지 않는다. 조용히 무시하면 보낸 쪽은 쓰였다고 믿는다.
        response = client.post(PREDICT, json=body(**{field: 1}), headers=AUTH)

        assert response.status_code == 422

    @pytest.mark.parametrize("day", ["2026-09-12", "2026-09-13"])
    def test_weekends_are_rejected(self, client: TestClient, predictor, day: str) -> None:
        response = client.post(PREDICT, json=body(date=day), headers=AUTH)

        assert response.status_code == 422
        assert "주말" in response.text


class TestPrediction:
    def test_contract(self, client: TestClient, predictor) -> None:
        response = client.post(PREDICT, json=body(), headers=AUTH)

        assert response.status_code == 200
        data = response.json()
        assert data["forecastDate"] == "2026-09-15"
        assert data["mealType"] == "LUNCH"
        assert data["predictedMealCount"] == 428
        assert isinstance(data["predictedMealCount"], int)
        assert data["model"]["syntheticData"] is True
        assert data["model"]["name"] == "fake_model"
        assert data["disclaimer"] == DISCLAIMER_SYNTHETIC
        assert set(data) == {"forecastDate", "mealType", "predictedMealCount", "reference",
                             "conditions", "model", "warnings", "disclaimer"}

    def test_no_confidence_interval_is_claimed(self, client: TestClient, predictor) -> None:
        text = client.post(PREDICT, json=body(), headers=AUTH).text.lower()

        # 보정된 예측 구간이 없다. 있는 것처럼 보이는 필드를 두지 않는다.
        for word in ("confidence", "interval", "95%", "lower", "upper"):
            assert word not in text

    def test_weekday_is_computed_from_the_date(self, client: TestClient, predictor) -> None:
        data = client.post(PREDICT, json=body(date="2026-09-15"), headers=AUTH).json()

        assert data["conditions"]["weekday"] == "화요일"
        row, _ = predictor.calls[-1]
        assert row["weekday"] == "화"

    def test_model_receives_the_metadata_feature_order(self, client: TestClient, predictor) -> None:
        client.post(PREDICT, json=body(), headers=AUTH)

        row, order = predictor.calls[-1]
        assert order == FEATURES
        assert list(row) == FEATURES

    def test_model_inputs_use_the_training_encoding(self, client: TestClient, predictor) -> None:
        client.post(PREDICT, json=body(examPeriod=True, specialMenu=True), headers=AUTH)

        row, _ = predictor.calls[-1]
        assert row["exam_period"] == 1 and row["special_menu"] == 1
        assert row["holiday"] == 0 and row["campus_event"] == 0
        assert row["semester"] == 2 and row["menu_category"] == "양식"

    def test_history_comes_from_before_the_forecast_date(self, client: TestClient, predictor) -> None:
        data = client.post(PREDICT, json=body(date="2026-09-15"), headers=AUTH).json()

        # InMemoryHistory: 식수 = 380 + 순번. 9/14(월)는 순번 10, 직전 화요일 9/8은 순번 6.
        assert data["reference"] == {
            "previousDay": 390,
            "sameWeekdayPreviousWeek": 386,
            "rolling7DayAverage": round(sum(380 + i for i in range(4, 11)) / 7),
            "historyAsOf": "2026-09-14",
        }
        row, _ = predictor.calls[-1]
        assert row["lag_1"] == 390

    @pytest.mark.parametrize("raw,expected", [(-12.3, 0), (-0.4, 0), (427.5, 428), (427.49, 427)])
    def test_count_is_rounded_and_never_negative(
        self, client: TestClient, tmp_path: Path, use_service, raw: float, expected: int
    ) -> None:
        use_service(fake_service(tmp_path, FakePredictor(raw)))

        data = client.post(PREDICT, json=body(), headers=AUTH).json()

        assert data["predictedMealCount"] == expected

    def test_non_finite_output_is_not_disguised_as_a_number(
        self, client: TestClient, tmp_path: Path, use_service
    ) -> None:
        use_service(fake_service(tmp_path, FakePredictor(math.nan)))

        assert client.post(PREDICT, json=body(), headers=AUTH).status_code == 503


class TestWarnings:
    def test_unknown_menu_is_warned(self, client: TestClient, predictor) -> None:
        data = client.post(PREDICT, json=body(mainMenu="마라탕"), headers=AUTH).json()

        assert any("마라탕" in w for w in data["warnings"])

    def test_missing_previous_day_is_warned(self, client: TestClient, predictor) -> None:
        # 이력은 9/14까지. 9/16(수)의 직전 운영일 9/15 기록이 없다.
        data = client.post(PREDICT, json=body(date="2026-09-16"), headers=AUTH).json()

        assert any("2026-09-15" in w and "2026-09-14" in w for w in data["warnings"])

    def test_forecast_inside_the_training_period_is_warned(self, client: TestClient, predictor) -> None:
        data = client.post(PREDICT, json=body(date="2026-09-10"), headers=AUTH).json()

        assert any("학습 데이터 기간" in w for w in data["warnings"])


class TestUnavailable:
    def test_missing_artifact_is_503(self, client: TestClient, tmp_path: Path, use_service) -> None:
        loader = ModelArtifactLoader(tmp_path / "x", tmp_path / "y",
                                     buildable_features=BUILDABLE_FEATURES, allow_synthetic=True)
        use_service(DemandForecastService(loader, InMemoryHistory(HISTORY_DAYS)))

        response = client.post(PREDICT, json=body(), headers=AUTH)

        assert response.status_code == 503
        assert "model artifact not found" in response.json()["detail"]

    def test_insufficient_history_is_422(self, client: TestClient, predictor) -> None:
        response = client.post(PREDICT, json=body(date="2026-09-02"), headers=AUTH)

        assert response.status_code == 422
        assert predictor.calls == []

    def test_a_leaky_history_source_is_refused(
        self, client: TestClient, tmp_path: Path, use_service
    ) -> None:
        fake = FakePredictor()
        loader, _ = fake_loader(tmp_path, fake)
        # 예측일 이후 기록까지 돌려주는 잘못된 저장소. 걸러 쓰지 않고 멈춰야 한다.
        use_service(DemandForecastService(loader, InMemoryHistory(HISTORY_DAYS, leaky=True)))

        response = client.post(PREDICT, json=body(date="2026-09-10"), headers=AUTH)

        assert response.status_code == 503
        assert fake.calls == []


def test_prediction_writes_nothing(client: TestClient, predictor) -> None:
    root = Path(__file__).resolve().parents[1]
    before = {p for p in root.rglob("*") if "__pycache__" not in p.parts}

    client.post(PREDICT, json=body(), headers=AUTH)
    client.get(STATUS, headers=AUTH)

    after = {p for p in root.rglob("*") if "__pycache__" not in p.parts}
    assert before == after


# ---------------------------------------------------------------------------
# 저장된 실제 artifact
# ---------------------------------------------------------------------------
REAL_FILES = (
    DEFAULT_DEMAND_FORECAST_MODEL_PATH,
    DEFAULT_DEMAND_FORECAST_METADATA_PATH,
    DEFAULT_DEMAND_HISTORY_PATH,
    DEFAULT_DEMAND_CCTV_LOG_PATH,
)
needs_real_artifact = pytest.mark.skipif(
    not all(p.is_file() for p in REAL_FILES),
    reason="저장된 artifact나 합성 이력이 없다. research/demand_forecasting 파이프라인을 먼저 실행한다.",
)


@needs_real_artifact
class TestRealArtifactSmoke:
    """앱이 기동하며 로드한 실제 artifact로 끝까지 돈다. 서비스를 갈아 끼우지 않는다."""

    def test_status_reports_the_real_model(self, client: TestClient) -> None:
        pytest.importorskip("sklearn")

        data = client.get(STATUS, headers=AUTH).json()

        assert data["available"] is True, data
        assert data["modelLoaded"] is True
        assert data["syntheticData"] is True
        assert data["target"] == "meal_count"
        assert data["modelName"] in {"lightgbm", "random_forest", "xgboost", "hist_gradient_boosting"}

    def test_predict_with_the_real_model(self, client: TestClient) -> None:
        pytest.importorskip("sklearn")
        with DEFAULT_DEMAND_HISTORY_PATH.open(encoding="utf-8-sig", newline="") as handle:
            last = list(csv.DictReader(handle))[-1]

        response = client.post(PREDICT, json=body(), headers=AUTH)

        assert response.status_code == 200, response.text
        data = response.json()
        assert isinstance(data["predictedMealCount"], int)
        assert 0 <= data["predictedMealCount"] < 1000
        assert data["model"]["syntheticData"] is True
        assert data["disclaimer"] == DISCLAIMER_SYNTHETIC
        # 9/15의 과거 feature는 합성 이력의 마지막 날(9/11)에서 나온다.
        assert data["reference"]["historyAsOf"] == last["date"]
        assert data["reference"]["previousDay"] == int(last["meal_count"])


@needs_real_artifact
def test_served_inputs_reproduce_the_training_rows(client: TestClient) -> None:
    """API가 만든 입력 행이 학습 CSV의 행과 같은지, 예측값으로 대조한다.

    같은 모델에 학습 CSV의 행을 직접 넣은 값과 API 응답이 같아야 한다. 요일 표현,
    0/1 인코딩, 과거 feature 계산, 컬럼 순서 중 하나라도 어긋나면 달라진다.
    """
    joblib = pytest.importorskip("joblib")
    pd = pytest.importorskip("pandas")
    import json

    pipeline = joblib.load(DEFAULT_DEMAND_FORECAST_MODEL_PATH)
    features = json.loads(DEFAULT_DEMAND_FORECAST_METADATA_PATH.read_text(encoding="utf-8"))["features"]
    frame = pd.read_csv(DEFAULT_DEMAND_HISTORY_PATH, encoding="utf-8-sig")
    # 앞 7행은 CSV 밖의 이력을 쓴다(history 테스트 참고). 그 뒤에서 고르게 뽑는다.
    sample = frame.iloc[7::20]
    expected = pipeline.predict(sample[features])

    for (_, row), want in zip(sample.iterrows(), expected):
        request = body(
            date=str(row["date"]),
            semester=int(row["semester"]),
            semesterWeek=int(row["semester_week"]),
            examPeriod=bool(row["exam_period"]),
            holiday=bool(row["holiday"]),
            campusEvent=bool(row["campus_event"]),
            temperature=float(row["temperature"]),
            rainfall=float(row["rainfall"]),
            humidity=float(row["humidity"]),
            menuCategory=str(row["menu_category"]),
            mainMenu=str(row["main_menu"]),
            menuPopularity=float(row["menu_popularity"]),
            specialMenu=bool(row["special_menu"]),
        )
        response = client.post(PREDICT, json=request, headers=AUTH)

        assert response.status_code == 200, (row["date"], response.text)
        assert response.json()["predictedMealCount"] == max(0, math.floor(want + 0.5)), row["date"]

    assert len(sample) > 20
