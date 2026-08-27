"""수요 예측 라우트.

확인하려는 것은 예측의 품질이 아니다. 그런 것은 아직 없다. 경계가 맞는지,
그리고 꺼져 있을 때 숫자를 지어내지 않는지를 본다.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from babplanner_ai.config import (
    AppEnv,
    ConfigurationError,
    PredictiveModelMode,
    Settings,
    get_settings,
)
from babplanner_ai.demand.fixture_estimator import (
    FIXTURE_MODEL_VERSION,
    FIXTURE_PREDICTED_PEOPLE,
)
from tests.conftest import TEST_SERVICE_TOKEN

PATH = "/internal/v1/ai/demand/predict"


def payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "requestId": "demand-11111111-2222-3333-4444-555555555555",
        "siteId": 1,
        "serviceDate": "2026-09-15",
        "mealType": "LUNCH",
    }
    body.update(overrides)
    return body


def post(client: TestClient, body: dict[str, object], token: str | None):
    headers = {} if token is None else {"X-Service-Token": token}
    return client.post(PATH, json=body, headers=headers)


@pytest.fixture
def disabled(client: TestClient) -> Iterator[None]:
    """수요 예측이 꺼진 설정으로 갈아 끼운다.

    TestClient를 새로 만들지 않는다. 이 환경은 socket 생성이 간헐적으로
    실패해서, 클라이언트를 늘릴수록 관계없는 이유로 무너진다.
    """
    settings = Settings(
        APP_ENV=AppEnv.LOCAL,
        WASTE_MODEL_MODE="fixture",
        DEMAND_MODEL_MODE=PredictiveModelMode.DISABLED,
        PROCUREMENT_MODEL_MODE=PredictiveModelMode.FIXTURE,
        AI_SERVICE_TOKEN=TEST_SERVICE_TOKEN,
    )
    client.app.dependency_overrides[get_settings] = lambda: settings
    yield
    client.app.dependency_overrides.pop(get_settings, None)


class TestServiceToken:
    def test_missing_token_is_rejected(self, client: TestClient) -> None:
        assert post(client, payload(), None).status_code == 401

    def test_wrong_token_is_rejected(self, client: TestClient) -> None:
        assert post(client, payload(), "wrong-token-value").status_code == 401

    def test_missing_and_wrong_look_the_same(self, client: TestClient) -> None:
        # 어느 쪽인지 알려주면 유효한 헤더 이름을 찾는 데 쓸 수 있다.
        missing = post(client, payload(), None)
        wrong = post(client, payload(), "wrong-token-value")

        assert missing.json() == wrong.json()

    def test_correct_token_is_accepted(self, client: TestClient) -> None:
        assert post(client, payload(), TEST_SERVICE_TOKEN).status_code == 200

    def test_token_is_never_echoed(self, client: TestClient) -> None:
        response = post(client, payload(), TEST_SERVICE_TOKEN)

        assert TEST_SERVICE_TOKEN not in response.text


class TestContract:
    def test_request_id_is_echoed_exactly(self, client: TestClient) -> None:
        body = payload(requestId="demand-abc-123")

        data = post(client, body, TEST_SERVICE_TOKEN).json()

        # 어긋나면 호출자가 다른 요청의 결과를 자기 것으로 저장하게 된다.
        assert data["requestId"] == "demand-abc-123"

    def test_predicted_people_is_non_negative(self, client: TestClient) -> None:
        data = post(client, payload(), TEST_SERVICE_TOKEN).json()

        assert isinstance(data["predictedPeople"], int)
        assert data["predictedPeople"] >= 0

    def test_model_version_names_the_fixture(self, client: TestClient) -> None:
        data = post(client, payload(), TEST_SERVICE_TOKEN).json()

        assert data["modelVersion"] == FIXTURE_MODEL_VERSION
        assert data["modelVersion"] == "fastapi-fixture-demand-v1"

    def test_fixture_is_deterministic(self, client: TestClient) -> None:
        first = post(client, payload(), TEST_SERVICE_TOKEN).json()
        second = post(
            client, payload(siteId=2, serviceDate="2026-12-31"), TEST_SERVICE_TOKEN
        ).json()

        # 입력에 따라 그럴듯하게 흔들리면 예측처럼 보인다. 예측이 아니다.
        assert first["predictedPeople"] == FIXTURE_PREDICTED_PEOPLE
        assert second["predictedPeople"] == FIXTURE_PREDICTED_PEOPLE

    def test_every_meal_type_is_accepted(self, client: TestClient) -> None:
        for meal in ("BREAKFAST", "LUNCH", "DINNER"):
            response = post(client, payload(mealType=meal), TEST_SERVICE_TOKEN)

            assert response.status_code == 200, meal

    def test_unknown_meal_type_is_rejected(self, client: TestClient) -> None:
        assert post(client, payload(mealType="BRUNCH"), TEST_SERVICE_TOKEN).status_code == 422

    def test_malformed_date_is_rejected(self, client: TestClient) -> None:
        assert (
            post(client, payload(serviceDate="2026-13-45"), TEST_SERVICE_TOKEN).status_code
            == 422
        )

    def test_blank_request_id_is_rejected(self, client: TestClient) -> None:
        assert post(client, payload(requestId=""), TEST_SERVICE_TOKEN).status_code == 422

    def test_missing_fields_are_rejected(self, client: TestClient) -> None:
        for field in ("requestId", "siteId", "serviceDate", "mealType"):
            body = payload()
            del body[field]

            assert post(client, body, TEST_SERVICE_TOKEN).status_code == 422, field

    def test_ambiguous_history_arrays_are_refused(self, client: TestClient) -> None:
        # 어느 기간의 무엇을 집계한 값인지 정의된 적이 없다. 조용히 무시하면
        # 보내는 쪽은 모델이 그 값을 쓰고 있다고 믿는다.
        for field in ("recentPeopleCounts", "recentWasteRates", "recentUsageCounts"):
            body = payload(**{field: [1, 2, 3]})

            assert post(client, body, TEST_SERVICE_TOKEN).status_code == 422, field

    def test_user_identifiers_are_refused(self, client: TestClient) -> None:
        for field in ("userId", "phone", "studentId", "name"):
            body = payload(**{field: "whatever"})

            assert post(client, body, TEST_SERVICE_TOKEN).status_code == 422, field

    def test_response_carries_nothing_else(self, client: TestClient) -> None:
        data = post(client, payload(), TEST_SERVICE_TOKEN).json()

        assert set(data) == {"requestId", "predictedPeople", "modelVersion"}


class TestDisabled:
    def test_disabled_returns_503(self, client: TestClient, disabled: None) -> None:
        # 끈 기능이 fixture로 대신 답하면, 예측이 없는 배포가 예측을 하고
        # 있는 것처럼 보인다.
        assert post(client, payload(), TEST_SERVICE_TOKEN).status_code == 503

    def test_disabled_still_checks_the_token(
        self, client: TestClient, disabled: None
    ) -> None:
        # 꺼져 있다는 사실도 아무나 알아낼 필요는 없다.
        assert post(client, payload(), None).status_code == 401

    def test_disabled_is_the_default(self, clean_env: pytest.MonkeyPatch) -> None:
        # 환경을 비우고 본다. 테스트 환경이 켜 둔 값이 기본값처럼 보이면
        # 안 된다.
        clean_env.delenv("DEMAND_MODEL_MODE", raising=False)
        clean_env.delenv("PROCUREMENT_MODEL_MODE", raising=False)

        settings = Settings(
            APP_ENV=AppEnv.LOCAL,
            WASTE_MODEL_MODE="fixture",
            AI_SERVICE_TOKEN=TEST_SERVICE_TOKEN,
        )

        assert settings.demand_model_mode is PredictiveModelMode.DISABLED
        assert settings.procurement_model_mode is PredictiveModelMode.DISABLED


class TestProductionSafety:
    def test_production_rejects_the_demand_fixture(self) -> None:
        settings = Settings(
            APP_ENV=AppEnv.PRODUCTION,
            WASTE_MODEL_MODE="fixture",
            DEMAND_MODEL_MODE=PredictiveModelMode.FIXTURE,
            AI_SERVICE_TOKEN=TEST_SERVICE_TOKEN,
        )

        with pytest.raises(ConfigurationError) as failure:
            settings.validated()

        assert "DEMAND_MODEL_MODE" in str(failure.value)

    def test_production_names_every_offending_mode(self) -> None:
        settings = Settings(
            APP_ENV=AppEnv.PRODUCTION,
            WASTE_MODEL_MODE="fixture",
            DEMAND_MODEL_MODE=PredictiveModelMode.FIXTURE,
            PROCUREMENT_MODEL_MODE=PredictiveModelMode.FIXTURE,
            AI_SERVICE_TOKEN=TEST_SERVICE_TOKEN,
        )

        with pytest.raises(ConfigurationError) as failure:
            settings.validated()

        # 하나씩 알려주면 고칠 때마다 다시 기동해 봐야 남은 것을 안다.
        message = str(failure.value)
        assert "WASTE_MODEL_MODE" in message
        assert "DEMAND_MODEL_MODE" in message
        assert "PROCUREMENT_MODEL_MODE" in message

    def test_production_allows_disabled(self) -> None:
        settings = Settings(
            APP_ENV=AppEnv.PRODUCTION,
            WASTE_MODEL_MODE="fixture",
            DEMAND_MODEL_MODE=PredictiveModelMode.DISABLED,
            PROCUREMENT_MODEL_MODE=PredictiveModelMode.DISABLED,
            AI_SERVICE_TOKEN=TEST_SERVICE_TOKEN,
        )

        # waste fixture 때문에 막히는 것이지 demand/procurement 때문이 아니다.
        with pytest.raises(ConfigurationError) as failure:
            settings.validated()

        assert "DEMAND_MODEL_MODE" not in str(failure.value)
        assert "PROCUREMENT_MODEL_MODE" not in str(failure.value)

    def test_failure_message_never_carries_the_token(self) -> None:
        settings = Settings(
            APP_ENV=AppEnv.PRODUCTION,
            WASTE_MODEL_MODE="fixture",
            DEMAND_MODEL_MODE=PredictiveModelMode.FIXTURE,
            AI_SERVICE_TOKEN=TEST_SERVICE_TOKEN,
        )

        with pytest.raises(ConfigurationError) as failure:
            settings.validated()

        assert TEST_SERVICE_TOKEN not in str(failure.value)


class TestStatelessness:
    def test_prediction_writes_nothing(self, client: TestClient) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        before = {p for p in root.rglob("*") if "__pycache__" not in p.parts}

        post(client, payload(), TEST_SERVICE_TOKEN)

        after = {p for p in root.rglob("*") if "__pycache__" not in p.parts}
        assert before == after

    def test_module_touches_no_storage_or_network(self) -> None:
        from pathlib import Path

        demand = Path(__file__).resolve().parents[1] / "src" / "babplanner_ai" / "demand"
        for path in demand.glob("*.py"):
            text = path.read_text("utf-8")

            for forbidden in ("sqlite", "redis", "requests", "httpx", "urllib", "open("):
                assert forbidden not in text, f"{path.name}: {forbidden}"
