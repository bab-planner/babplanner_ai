"""분석 endpoint."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from babplanner_ai.security import SERVICE_TOKEN_HEADER
from babplanner_ai.waste import api, fixture_estimator, service
from babplanner_ai.waste.estimator import WasteEstimateError
from babplanner_ai.waste.fixture_estimator import (
    FIXTURE_CONFIDENCE,
    FIXTURE_MODEL_VERSION,
    FIXTURE_WASTE_RATE,
)
from babplanner_ai.waste.models import WasteAnalysisDecision, WasteEstimate
from tests.conftest import TEST_SERVICE_TOKEN, jpeg_bytes, png_bytes

PATH = "/internal/v1/ai/waste/analyze"
AUTH = {SERVICE_TOKEN_HEADER: TEST_SERVICE_TOKEN}


def analyze(
    client: TestClient,
    *,
    request_id: str = "waste-42",
    menu_id: str = "3",
    image: bytes | None = None,
    filename: str = "a.png",
    content_type: str = "image/png",
):
    return client.post(
        PATH,
        data={"requestId": request_id, "menuId": menu_id},
        files={"image": (filename, png_bytes() if image is None else image, content_type)},
        headers=AUTH,
    )


def test_multipart_request_succeeds(client: TestClient) -> None:
    assert analyze(client).status_code == 200


def test_request_id_is_echoed(client: TestClient) -> None:
    body = analyze(client, request_id="waste-4242").json()

    assert body["requestId"] == "waste-4242"


def test_menu_id_reaches_the_estimator(client: TestClient) -> None:
    seen: list[int] = []

    class Recording:
        def estimate(self, image: bytes, menu_id: int) -> WasteEstimate:
            seen.append(menu_id)
            return WasteEstimate(
                decision=WasteAnalysisDecision.APPROVED,
                waste_rate=0.1,
                confidence=0.9,
                model_version="fastapi-fixture-waste-v1",
            )

    client.app.dependency_overrides[api.get_estimator] = lambda: Recording()
    try:
        analyze(client, menu_id="77")
    finally:
        client.app.dependency_overrides.clear()

    assert seen == [77]


def test_image_bytes_reach_the_estimator(client: TestClient) -> None:
    seen: list[bytes] = []
    sent = png_bytes((16, 16))

    class Recording:
        def estimate(self, image: bytes, menu_id: int) -> WasteEstimate:
            seen.append(image)
            return WasteEstimate(
                decision=WasteAnalysisDecision.REJECTED, model_version="fastapi-fixture-waste-v1"
            )

    client.app.dependency_overrides[api.get_estimator] = lambda: Recording()
    try:
        analyze(client, image=sent)
    finally:
        client.app.dependency_overrides.clear()

    assert seen == [sent]


def test_png_is_accepted(client: TestClient) -> None:
    assert analyze(client, image=png_bytes()).status_code == 200


def test_jpeg_is_accepted(client: TestClient) -> None:
    # 형식을 새로 제한하지 않는다. 디코딩되면 받는다.
    response = analyze(client, image=jpeg_bytes(), filename="a.jpg", content_type="image/jpeg")

    assert response.status_code == 200


def test_empty_image_is_unprocessable(client: TestClient) -> None:
    assert analyze(client, image=b"").status_code == 422


def test_undecodable_image_is_unprocessable(client: TestClient) -> None:
    assert analyze(client, image=b"not really an image").status_code == 422


def test_oversized_image_is_unprocessable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from babplanner_ai.config import get_settings

    monkeypatch.setenv("AI_MAX_IMAGE_BYTES", "10")
    get_settings.cache_clear()
    try:
        assert analyze(client, image=png_bytes((64, 64))).status_code == 422
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def test_blank_request_id_is_unprocessable(client: TestClient) -> None:
    assert analyze(client, request_id="   ").status_code == 422


def test_missing_fields_are_unprocessable(client: TestClient) -> None:
    response = client.post(PATH, data={"requestId": "waste-1"}, headers=AUTH)

    assert response.status_code == 422


def test_fixture_returns_approved(client: TestClient) -> None:
    body = analyze(client).json()

    assert body["decision"] == "APPROVED"


def test_fixture_rate_is_within_range(client: TestClient) -> None:
    body = analyze(client).json()

    assert body["wasteRate"] == FIXTURE_WASTE_RATE
    assert 0 <= body["wasteRate"] <= 1


def test_fixture_confidence_is_within_range(client: TestClient) -> None:
    body = analyze(client).json()

    assert body["confidence"] == FIXTURE_CONFIDENCE
    assert 0 <= body["confidence"] <= 1


def test_model_version_marks_itself_as_a_fixture(client: TestClient) -> None:
    body = analyze(client).json()

    assert body["modelVersion"] == FIXTURE_MODEL_VERSION
    assert "fixture" in body["modelVersion"]
    # 실제 모델처럼 보이는 이름을 쓰지 않는다.
    for misleading in ("production", "best-model", "prod"):
        assert misleading not in body["modelVersion"]


def test_model_version_names_this_process(client: TestClient) -> None:
    # 서비스 백엔드 안의 개발용 분석기와 구분돼야, 저장된 결과만 보고도
    # 어느 쪽을 거쳐 왔는지 알 수 있다.
    body = analyze(client).json()

    assert body["modelVersion"] == "fastapi-fixture-waste-v1"
    assert body["modelVersion"].startswith("fastapi-")


def test_backend_fixture_name_is_not_used_here() -> None:
    source = "".join(
        inspect.getsource(module) for module in (fixture_estimator, api, service)
    )

    # 이 이름은 서비스 백엔드 쪽 fixture 것이다.
    assert "spring-fixture-waste-v1" not in source


def test_approved_must_carry_rate_and_confidence(client: TestClient) -> None:
    body = analyze(client).json()

    assert body["decision"] == "APPROVED"
    assert body["wasteRate"] is not None
    assert body["confidence"] is not None


def test_rejected_may_omit_rate_and_confidence(client: TestClient) -> None:
    class Rejecting:
        def estimate(self, image: bytes, menu_id: int) -> WasteEstimate:
            return WasteEstimate(
                decision=WasteAnalysisDecision.REJECTED, model_version="fastapi-fixture-waste-v1"
            )

    client.app.dependency_overrides[api.get_estimator] = lambda: Rejecting()
    try:
        body = analyze(client).json()
    finally:
        client.app.dependency_overrides.clear()

    assert body["decision"] == "REJECTED"
    assert body["wasteRate"] is None
    assert body["confidence"] is None


def test_estimator_failure_is_not_disguised_as_rejected(client: TestClient) -> None:
    # 분석을 못 한 것과 승인하지 않은 것은 다른 사건이다.
    class Broken:
        def estimate(self, image: bytes, menu_id: int) -> WasteEstimate:
            raise WasteEstimateError("모델을 사용할 수 없습니다.")

    client.app.dependency_overrides[api.get_estimator] = lambda: Broken()
    try:
        response = analyze(client)
    finally:
        client.app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "REJECTED" not in response.text


def test_out_of_range_estimate_is_rejected_by_the_model() -> None:
    with pytest.raises(Exception):
        WasteEstimate(
            decision=WasteAnalysisDecision.APPROVED,
            waste_rate=1.5,
            confidence=0.9,
            model_version="fastapi-fixture-waste-v1",
        )


def test_response_carries_no_reward_fields(client: TestClient) -> None:
    body = analyze(client).json()

    for forbidden in ("earnedPoints", "couponIssued", "points", "coupon"):
        assert forbidden not in body


def test_response_carries_no_user_data(client: TestClient) -> None:
    body = analyze(client).json()

    assert set(body) == {"requestId", "decision", "wasteRate", "confidence", "modelVersion"}


def test_response_keys_match_the_contract(client: TestClient) -> None:
    body = analyze(client).json()

    for required in ("requestId", "decision", "modelVersion"):
        assert body[required] is not None


def test_service_does_not_persist_images(tmp_path: Path, client: TestClient) -> None:
    before = set(tmp_path.iterdir())

    analyze(client, image=png_bytes((32, 32)))

    assert set(tmp_path.iterdir()) == before
    source = inspect.getsource(service) + inspect.getsource(api)
    for forbidden in ("open(", "Path(", "shutil", "os.makedirs", "write_bytes"):
        assert forbidden not in source.replace("Image.open(", "").replace(
            "io.BytesIO(", ""
        )


def test_no_outbound_network_clients_in_sources() -> None:
    source = "".join(
        inspect.getsource(module)
        for module in (api, service, fixture_estimator)
    )

    for forbidden in ("httpx", "requests.", "urllib", "socket", "openai", "anthropic"):
        assert forbidden not in source


def test_no_user_endpoints_are_exposed(client: TestClient) -> None:
    paths = list(client.app.openapi()["paths"])

    assert paths == [PATH]
