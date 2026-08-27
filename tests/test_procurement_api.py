"""발주 추천 라우트.

fixture는 아무것도 추천하지 않는다. 여기서 확인하는 것은 그 사실이
유지되는지다. 계획 수량을 슬쩍 깎아 내려주기 시작하면 그 숫자로 실제
발주가 나간다.
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
from babplanner_ai.procurement.fixture_estimator import (
    FIXTURE_MODEL_VERSION,
    FIXTURE_REASON,
)
from tests.conftest import TEST_SERVICE_TOKEN

PATH = "/internal/v1/ai/procurement/recommend"


def ingredient(name: str = "돼지고기", unit: str = "kg", quantity: float = 12.5):
    return {"ingredientName": name, "unit": unit, "plannedQuantity": quantity}


def payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "requestId": "procurement-11111111-2222-3333-4444-555555555555",
        "siteId": 1,
        "serviceDate": "2026-09-15",
        "mealType": "LUNCH",
        "predictedPeople": 120,
        "plannedIngredients": [ingredient()],
    }
    body.update(overrides)
    return body


def post(client: TestClient, body: dict[str, object], token: str | None):
    headers = {} if token is None else {"X-Service-Token": token}
    return client.post(PATH, json=body, headers=headers)


@pytest.fixture
def disabled(client: TestClient) -> Iterator[None]:
    settings = Settings(
        APP_ENV=AppEnv.LOCAL,
        WASTE_MODEL_MODE="fixture",
        DEMAND_MODEL_MODE=PredictiveModelMode.FIXTURE,
        PROCUREMENT_MODEL_MODE=PredictiveModelMode.DISABLED,
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
        missing = post(client, payload(), None)
        wrong = post(client, payload(), "wrong-token-value")

        assert missing.json() == wrong.json()

    def test_correct_token_is_accepted(self, client: TestClient) -> None:
        assert post(client, payload(), TEST_SERVICE_TOKEN).status_code == 200

    def test_token_is_never_echoed(self, client: TestClient) -> None:
        assert TEST_SERVICE_TOKEN not in post(client, payload(), TEST_SERVICE_TOKEN).text


class TestFixtureBehaviour:
    def test_recommends_exactly_what_was_planned(self, client: TestClient) -> None:
        body = payload(plannedIngredients=[ingredient("양파", "kg", 8.0)])

        item = post(client, body, TEST_SERVICE_TOKEN).json()["recommendations"][0]

        # 재고도 단가도 모르는 구현이 낼 수 있는 정직한 답은 이것뿐이다.
        assert item["recommendedQuantity"] == 8.0
        assert item["plannedQuantity"] == 8.0
        assert item["adjustmentRate"] == 0.0

    def test_reason_says_it_is_a_fixture(self, client: TestClient) -> None:
        item = post(client, payload(), TEST_SERVICE_TOKEN).json()["recommendations"][0]

        assert item["reason"] == FIXTURE_REASON
        assert "fixture" in item["reason"]

    def test_model_version_names_the_fixture(self, client: TestClient) -> None:
        data = post(client, payload(), TEST_SERVICE_TOKEN).json()

        assert data["modelVersion"] == FIXTURE_MODEL_VERSION
        assert data["modelVersion"] == "fastapi-fixture-procurement-v1"

    def test_predicted_people_does_not_change_the_answer(
        self, client: TestClient
    ) -> None:
        few = post(client, payload(predictedPeople=1), TEST_SERVICE_TOKEN).json()
        many = post(client, payload(predictedPeople=5000), TEST_SERVICE_TOKEN).json()

        # 인원에 따라 수량이 움직이면 추천처럼 보인다. 추천이 아니다.
        assert few["recommendations"] == many["recommendations"]

    def test_order_is_preserved(self, client: TestClient) -> None:
        planned = [
            ingredient("돼지고기", "kg", 12.5),
            ingredient("양파", "kg", 8.0),
            ingredient("간장", "L", 2.0),
        ]

        data = post(client, payload(plannedIngredients=planned), TEST_SERVICE_TOKEN).json()

        # 호출자가 입력과 출력을 자리로 맞춘다.
        assert [item["ingredientName"] for item in data["recommendations"]] == [
            "돼지고기",
            "양파",
            "간장",
        ]

    def test_every_ingredient_comes_back_once(self, client: TestClient) -> None:
        planned = [ingredient("쌀", "kg", 40.0), ingredient("쌀", "포", 3.0)]

        data = post(client, payload(plannedIngredients=planned), TEST_SERVICE_TOKEN).json()

        assert len(data["recommendations"]) == 2
        assert [item["unit"] for item in data["recommendations"]] == ["kg", "포"]

    def test_units_are_returned_unchanged(self, client: TestClient) -> None:
        body = payload(plannedIngredients=[ingredient("간장", "L", 2.0)])

        item = post(client, body, TEST_SERVICE_TOKEN).json()["recommendations"][0]

        # 단위 변환 정책은 없다. kg/g를 마음대로 바꾸지 않는다.
        assert item["unit"] == "L"

    def test_request_id_is_echoed_exactly(self, client: TestClient) -> None:
        data = post(client, payload(requestId="procurement-xyz"), TEST_SERVICE_TOKEN).json()

        assert data["requestId"] == "procurement-xyz"


class TestValidation:
    def test_empty_ingredient_list_is_rejected(self, client: TestClient) -> None:
        assert (
            post(client, payload(plannedIngredients=[]), TEST_SERVICE_TOKEN).status_code
            == 422
        )

    def test_blank_ingredient_name_is_rejected(self, client: TestClient) -> None:
        body = payload(plannedIngredients=[ingredient(name="")])

        assert post(client, body, TEST_SERVICE_TOKEN).status_code == 422

    def test_blank_unit_is_rejected(self, client: TestClient) -> None:
        body = payload(plannedIngredients=[ingredient(unit="")])

        assert post(client, body, TEST_SERVICE_TOKEN).status_code == 422

    def test_non_positive_quantity_is_rejected(self, client: TestClient) -> None:
        for quantity in (0, -1.5):
            body = payload(plannedIngredients=[ingredient(quantity=quantity)])

            assert post(client, body, TEST_SERVICE_TOKEN).status_code == 422, quantity

    def test_quantity_embedded_in_a_string_is_rejected(self, client: TestClient) -> None:
        # "10kg" 같은 값은 파싱하는 쪽마다 다르게 읽힌다.
        body = payload(plannedIngredients=[{"ingredientName": "쌀", "unit": "kg", "plannedQuantity": "10kg"}])

        assert post(client, body, TEST_SERVICE_TOKEN).status_code == 422

    def test_negative_predicted_people_is_rejected(self, client: TestClient) -> None:
        assert (
            post(client, payload(predictedPeople=-1), TEST_SERVICE_TOKEN).status_code == 422
        )

    def test_missing_fields_are_rejected(self, client: TestClient) -> None:
        for field in (
            "requestId",
            "siteId",
            "serviceDate",
            "mealType",
            "predictedPeople",
            "plannedIngredients",
        ):
            body = payload()
            del body[field]

            assert post(client, body, TEST_SERVICE_TOKEN).status_code == 422, field

    def test_menu_waste_history_is_refused(self, client: TestClient) -> None:
        # 어느 기간의 어떤 집계인지 정의된 적이 없다. 실제 모델이 요구하게
        # 되면 그때 구조를 정해 추가한다.
        body = payload(menuWasteHistory=[{"menuId": 1, "averageWasteRate": 0.2}])

        assert post(client, body, TEST_SERVICE_TOKEN).status_code == 422

    def test_user_identifiers_are_refused(self, client: TestClient) -> None:
        for field in ("userId", "phone", "studentId", "name"):
            body = payload(**{field: "whatever"})

            assert post(client, body, TEST_SERVICE_TOKEN).status_code == 422, field

    def test_unknown_ingredient_field_is_refused(self, client: TestClient) -> None:
        planned = [{**ingredient(), "supplierAccount": "12345"}]

        assert (
            post(client, payload(plannedIngredients=planned), TEST_SERVICE_TOKEN).status_code
            == 422
        )


class TestDisabled:
    def test_disabled_returns_503(self, client: TestClient, disabled: None) -> None:
        assert post(client, payload(), TEST_SERVICE_TOKEN).status_code == 503

    def test_disabled_still_checks_the_token(
        self, client: TestClient, disabled: None
    ) -> None:
        assert post(client, payload(), None).status_code == 401


class TestProductionSafety:
    def test_production_rejects_the_procurement_fixture(self) -> None:
        settings = Settings(
            APP_ENV=AppEnv.PRODUCTION,
            WASTE_MODEL_MODE="fixture",
            PROCUREMENT_MODEL_MODE=PredictiveModelMode.FIXTURE,
            AI_SERVICE_TOKEN=TEST_SERVICE_TOKEN,
        )

        with pytest.raises(ConfigurationError) as failure:
            settings.validated()

        assert "PROCUREMENT_MODEL_MODE" in str(failure.value)


class TestStatelessness:
    def test_module_touches_no_storage_or_network(self) -> None:
        from pathlib import Path

        package = (
            Path(__file__).resolve().parents[1] / "src" / "babplanner_ai" / "procurement"
        )
        for path in package.glob("*.py"):
            text = path.read_text("utf-8")

            for forbidden in ("sqlite", "redis", "requests", "httpx", "urllib", "open("):
                assert forbidden not in text, f"{path.name}: {forbidden}"

    def test_no_saving_rate_is_claimed(self) -> None:
        from pathlib import Path

        package = (
            Path(__file__).resolve().parents[1] / "src" / "babplanner_ai" / "procurement"
        )
        source = "\n".join(path.read_text("utf-8") for path in package.glob("*.py"))

        # 절감률을 지어내는 상수가 슬쩍 들어오지 않게 한다.
        for forbidden in ("0.8", "0.9", "0.85", "* 0.", "savings", "discount"):
            assert forbidden not in source, forbidden
