"""AI 서비스의 runtime 범위를 소스로 고정한다.

최종 AI 기능은 정확히 두 개다.

1. CCTV 혼잡도: 영상 수집기(demo/cctv_collector, research/cctv_counting)가 Backend로
   push한다. 이 FastAPI 서비스를 거치지 않으므로 여기에 라우트가 없다.
2. 다음 운영일 중식 식수 인원 예측: `demand_forecast`.

이 서비스가 제공하는 runtime 라우트는 `/health`와 식수 예측 status/predict뿐이다.
아래는 한 번 정리한 legacy 경로가 다시 들어오는 것을 막는다.

- 잔반 분석 AI (잔반 인증은 Backend의 관리자 승인 기능이다)
- 일반 수요 예측 fixture (`/internal/v1/ai/demand/predict`)
- 발주/주문 추천 AI (`/internal/v1/ai/procurement/recommend`)
- 추천 AI

`research/` 아래의 오프라인 연구 코드, 그리고 서비스에 연결되지 않은 오프라인 CLI
도구(`collection`, `dataset`)는 허용한다. 서비스가 그것들을 import하지 않는지만 본다.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "babplanner_ai"

#: 서비스가 제공하는 경로 전부. 늘리려면 이 목록과 문서를 함께 바꿔야 한다.
FINAL_RUNTIME_PATHS = {
    "/health",
    "/internal/v1/ai/demand-forecast/status",
    "/internal/v1/ai/demand-forecast/predict",
}

#: 서비스 패키지 안에서 허용하는 하위 패키지. collection/dataset은 서비스에 연결되지
#: 않은 오프라인 CLI 도구다(잔반 사진 수집/검증용 기록). 서비스가 쓰지 않는다.
ALLOWED_SUBPACKAGES = {"demand_forecast", "collection", "dataset"}
OFFLINE_SUBPACKAGES = {"collection", "dataset"}

LEGACY_PACKAGES = {"waste", "demand", "procurement", "order", "orders", "recommendation"}

#: runtime 소스에 다시 나타나면 안 되는 말.
LEGACY_TOKENS = (
    "fixture_estimator",
    "FixtureWaste",
    "FixtureDemand",
    "FixtureProcurement",
    "ai/waste",
    "ai/demand/predict",
    "ai/procurement",
    "WASTE_MODEL_MODE",
    "DEMAND_MODEL_MODE",
    "PROCUREMENT_MODEL_MODE",
    "AI_MAX_IMAGE_BYTES",
)


def _runtime_sources() -> list[tuple[str, str]]:
    """서비스가 실제로 로드하는 소스(오프라인 도구 제외)."""
    found = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        parts = path.relative_to(PACKAGE_ROOT).parts
        if "__pycache__" in parts or (parts and parts[0] in OFFLINE_SUBPACKAGES):
            continue
        found.append((str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"), path.read_text("utf-8")))
    return found


def test_runtime_routes_are_exactly_the_final_scope() -> None:
    from babplanner_ai.main import app

    paths = set(app.openapi()["paths"])

    assert paths == FINAL_RUNTIME_PATHS


def test_no_legacy_ai_route_is_exposed() -> None:
    from babplanner_ai.main import app

    paths = " ".join(app.openapi()["paths"])

    for legacy in ("waste", "procurement", "order", "recommend"):
        assert legacy not in paths
    # 일반 수요 예측(/demand/predict)과 식수 예측(/demand-forecast/...)은 다른 것이다.
    assert "/demand/predict" not in paths


def test_service_package_has_only_the_allowed_subpackages() -> None:
    subpackages = {
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__" and any(path.rglob("*.py"))
    }

    assert subpackages <= ALLOWED_SUBPACKAGES, subpackages - ALLOWED_SUBPACKAGES
    assert not (subpackages & LEGACY_PACKAGES)


def test_no_fixture_estimator_exists_in_the_service() -> None:
    fixtures = [
        str(path.relative_to(PROJECT_ROOT))
        for path in PACKAGE_ROOT.rglob("*fixture*.py")
        if "__pycache__" not in path.parts
    ]

    assert fixtures == []


def test_runtime_source_has_no_legacy_references() -> None:
    offenders = [
        (name, token)
        for name, text in _runtime_sources()
        for token in LEGACY_TOKENS
        if token in text
    ]

    assert offenders == []


def test_main_registers_only_health_and_demand_forecast() -> None:
    main = (PACKAGE_ROOT / "main.py").read_text("utf-8")
    included = [line.strip() for line in main.splitlines() if "include_router(" in line]

    assert sorted(included) == [
        "app.include_router(demand_forecast_router)",
        "app.include_router(health_router)",
    ]


def test_service_does_not_import_the_offline_tools() -> None:
    offenders = [
        (name, offline)
        for name, text in _runtime_sources()
        for offline in OFFLINE_SUBPACKAGES
        if f"babplanner_ai.{offline}" in text or f"from babplanner_ai import {offline}" in text
    ]

    assert offenders == []


def test_settings_have_no_legacy_ai_switches() -> None:
    from babplanner_ai.config import Settings

    fields = set(Settings.model_fields)

    for removed in ("waste_model_mode", "demand_model_mode", "procurement_model_mode", "max_image_bytes"):
        assert removed not in fields
