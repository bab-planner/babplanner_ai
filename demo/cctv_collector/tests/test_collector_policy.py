"""수집기의 정책과 계약.

여기서 지키는 것은 셋이다.

    1. 인원수를 단계로 바꾸는 규칙이 **설정에서만** 온다.
    2. 백엔드로 나가는 본문에 계약에 없는 필드가 없다.
    3. 토큰이 파일에도, 인자에도, 로그에도 없다.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cctv_collector.config import (
    BACKEND_BASE_URL_ENV,
    INTERNAL_SERVICE_TOKEN_ENV,
    MIN_SERVICE_TOKEN_LENGTH,
    BackendSettings,
    CollectorConfigError,
    adopted_research_policy,
    camera_config_from_dict,
    load_backend_settings,
)
from cctv_collector.congestion_policy import (
    Calibration,
    CongestionLevel,
    CongestionPolicyError,
    CountThresholdPolicy,
)
from cctv_collector.schemas import (
    SPRING_REQUEST_FIELDS,
    SPRING_REQUIRED_FIELDS,
    CongestionObservation,
    LocalObservationRecord,
    SchemaError,
)

TOKEN = "t" * MIN_SERVICE_TOKEN_LENGTH
NOW = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone(timedelta(hours=9)))


def camera_payload(**overrides) -> dict:
    payload = {
        "site_id": 1,
        "device_id": "demo-cctv-01",
        "model_version": "cctv-fasterrcnn-mobilenetv3-demo-v1",
        "frame": {"width": 384, "height": 288},
        "roi": {"enabled": True, "x1": 0.0, "y1": 0.20, "x2": 1.0, "y2": 0.95},
        "line": {
            "start": [0.5, 0.20],
            "end": [0.5, 0.95],
            "in_direction": "POSITIVE",
            "anchor": "BOTTOM_CENTER",
        },
        "observation": {"interval_seconds": 1.0, "aggregation": "latest"},
        "congestion_policy": {
            "type": "count_thresholds",
            "calibration": "DEMO_ONLY_UNCALIBRATED",
            "comfortable_max": 5,
            "normal_max": 12,
            "crowded_max": 20,
        },
    }
    payload.update(overrides)
    return payload


def policy(**overrides) -> CountThresholdPolicy:
    values = {
        "comfortable_max": 5,
        "normal_max": 12,
        "crowded_max": 20,
        "calibration": Calibration.DEMO_ONLY_UNCALIBRATED,
    }
    values.update(overrides)
    return CountThresholdPolicy(**values)


# --- 혼잡도 단계 정책 -------------------------------------------------------


def test_comfortable_boundary_is_inclusive():
    assert policy().level_for(0) is CongestionLevel.COMFORTABLE
    assert policy().level_for(5) is CongestionLevel.COMFORTABLE


def test_normal_boundary_is_inclusive():
    assert policy().level_for(6) is CongestionLevel.NORMAL
    assert policy().level_for(12) is CongestionLevel.NORMAL


def test_crowded_boundary_is_inclusive():
    assert policy().level_for(13) is CongestionLevel.CROWDED
    assert policy().level_for(20) is CongestionLevel.CROWDED


def test_above_crowded_is_very_crowded():
    assert policy().level_for(21) is CongestionLevel.VERY_CROWDED
    assert policy().level_for(500) is CongestionLevel.VERY_CROWDED


def test_thresholds_must_be_strictly_increasing():
    with pytest.raises(CongestionPolicyError):
        policy(normal_max=5)
    with pytest.raises(CongestionPolicyError):
        policy(comfortable_max=13)
    with pytest.raises(CongestionPolicyError):
        policy(crowded_max=12)


def test_thresholds_reject_negative_and_non_integers():
    with pytest.raises(CongestionPolicyError):
        policy(comfortable_max=-1)
    with pytest.raises(CongestionPolicyError):
        policy(normal_max=7.5)


def test_level_rejects_a_negative_count():
    with pytest.raises(CongestionPolicyError):
        policy().level_for(-1)


def test_level_strings_match_the_backend_enum():
    """Flutter가 이 문자열을 그대로 읽는다. 하나라도 어긋나면 표시가 깨진다."""
    assert {level.value for level in CongestionLevel} == {
        "COMFORTABLE",
        "NORMAL",
        "CROWDED",
        "VERY_CROWDED",
    }


def test_demo_thresholds_declare_that_they_are_uncalibrated():
    assert policy().is_calibrated is False
    assert policy().describe()["calibration"] == "DEMO_ONLY_UNCALIBRATED"


# --- Spring 본문 ------------------------------------------------------------


def observation(**overrides) -> CongestionObservation:
    values = {
        "site_id": 1,
        "device_id": "demo-cctv-01",
        "people_count": 8,
        "level": CongestionLevel.NORMAL,
        "observed_at": NOW,
        "model_version": "cctv-demo-v1",
    }
    values.update(overrides)
    return CongestionObservation(**values)


def test_people_count_becomes_the_payload():
    payload = observation(people_count=8).to_payload()
    assert payload["peopleCount"] == 8
    assert payload["level"] == "NORMAL"
    assert payload["siteId"] == 1
    assert payload["deviceId"] == "demo-cctv-01"


def test_payload_carries_only_contract_fields():
    """계약에 없는 이름을 몰래 보내지 않는다."""
    payload = observation().to_payload()
    assert set(payload) <= SPRING_REQUEST_FIELDS
    assert SPRING_REQUIRED_FIELDS <= set(payload)


def test_payload_does_not_carry_entry_counts():
    """백엔드에 입장 인원을 받을 자리가 없다. 몰래 넣지 않는다."""
    payload = observation().to_payload()
    for name in ("entriesTotal", "exitsTotal", "entries", "exits"):
        assert name not in payload


def test_payload_does_not_carry_source():
    """출처는 백엔드가 기록한다. 보내는 쪽이 정하면 신뢰할 수 없다."""
    assert "source" not in observation().to_payload()


def test_payload_has_no_personal_or_frame_data():
    payload = json.dumps(observation().to_payload()).lower()
    for word in (
        "bbox",
        "trackid",
        "track_id",
        "face",
        "image",
        "frame",
        "name",
        "userid",
        "phone",
        "studentid",
        "embedding",
    ):
        assert word not in payload


def test_observed_at_must_carry_a_timezone():
    with pytest.raises(SchemaError):
        observation(observed_at=datetime(2026, 8, 28, 12, 0, 0))


def test_observed_at_is_serialised_with_an_offset():
    payload = observation().to_payload()
    assert payload["observedAt"] == "2026-08-28T12:00:00.000+09:00"


def test_blank_model_version_is_refused_before_the_backend_sees_it():
    with pytest.raises(SchemaError):
        observation(model_version="   ")


def test_model_version_can_be_omitted():
    assert "modelVersion" not in observation(model_version=None).to_payload()


def test_negative_people_count_is_refused():
    with pytest.raises(SchemaError):
        observation(people_count=-1)


def test_local_record_keeps_entry_counts_and_both_time_axes():
    """로컬 기록에는 남긴다. 두 시간축이 나란히 보여야 한다."""
    record = LocalObservationRecord(
        video_timestamp_seconds=12.0,
        observed_at=NOW,
        site_id=1,
        device_id="demo-cctv-01",
        people_count=8,
        level=CongestionLevel.NORMAL,
        samples=25,
        entries_total=14,
        exits_total=3,
        send_status="SUCCESS",
        http_status=202,
    )
    payload = record.to_json()
    assert payload["entriesTotal"] == 14
    assert payload["exitsTotal"] == 3
    assert payload["videoTimestampSeconds"] == 12.0
    assert payload["observedAt"].startswith("2026-08-28T12:00:00")


# --- 설정 -------------------------------------------------------------------


def test_site_id_and_device_id_come_from_config():
    camera = camera_config_from_dict(camera_payload(site_id=7, device_id="cam-b"))
    assert camera.site_id == 7
    assert camera.device_id == "cam-b"


def test_site_id_must_be_a_positive_integer():
    with pytest.raises(CollectorConfigError):
        camera_config_from_dict(camera_payload(site_id=0))
    with pytest.raises(CollectorConfigError):
        camera_config_from_dict(camera_payload(site_id="1"))


def test_device_id_must_not_be_blank():
    with pytest.raises(CollectorConfigError):
        camera_config_from_dict(camera_payload(device_id="  "))


def test_camera_config_refuses_secret_looking_keys():
    """토큰을 설정 파일에 적는 습관이 생기지 않게 막는다."""
    for key in ("token", "service_token", "serviceToken", "password", "secret"):
        with pytest.raises(CollectorConfigError):
            camera_config_from_dict(camera_payload(**{key: "x"}))


def test_camera_config_requires_roi_and_line_per_camera():
    payload = camera_payload()
    del payload["line"]
    with pytest.raises(CollectorConfigError):
        camera_config_from_dict(payload)


def test_camera_config_requires_the_frame_size():
    payload = camera_payload()
    del payload["frame"]
    with pytest.raises(CollectorConfigError):
        camera_config_from_dict(payload)


def test_congestion_policy_is_optional():
    payload = camera_payload()
    del payload["congestion_policy"]
    camera = camera_config_from_dict(payload)
    assert camera.congestion_policy is None


def test_observation_interval_must_be_positive():
    with pytest.raises(CollectorConfigError):
        camera_config_from_dict(
            camera_payload(observation={"interval_seconds": 0, "aggregation": "latest"})
        )


def test_unknown_aggregation_is_refused():
    with pytest.raises(CollectorConfigError):
        camera_config_from_dict(
            camera_payload(
                observation={"interval_seconds": 1.0, "aggregation": "adaptive"}
            )
        )


def test_example_camera_config_is_marked_demo_only():
    """저장소에 있는 예시가 운영 기준처럼 읽히면 안 된다."""
    path = Path(__file__).resolve().parents[1] / "cameras" / "demo_camera.example.yaml"
    text = path.read_text(encoding="utf-8")
    assert "DEMO_ONLY_UNCALIBRATED" in text
    for word in ("token", "password", "secret"):
        assert f"{word}:" not in text.lower()


def test_example_camera_config_has_no_absolute_local_path():
    path = Path(__file__).resolve().parents[1] / "cameras" / "demo_camera.example.yaml"
    text = path.read_text(encoding="utf-8")
    assert "C:\\Users" not in text
    assert "/home/" not in text


# --- 연구 결과를 복사하지 않는다 --------------------------------------------


def test_collector_reads_the_adopted_policy_from_the_research_freeze():
    """A6가 채택한 것은 baseline이다. 개선 tracker는 승격되지 않았다."""
    policy_, name = adopted_research_policy()
    assert name == "P0_baseline"
    assert policy_.occupancy_threshold == pytest.approx(0.7)
    assert policy_.tracker.value == "baseline_greedy"


def test_adopted_policy_fails_loudly_when_the_freeze_is_missing(tmp_path):
    with pytest.raises(CollectorConfigError):
        adopted_research_policy(tmp_path / "nope.json")


# --- 토큰 ------------------------------------------------------------------


def test_backend_settings_come_from_the_environment_only():
    settings = load_backend_settings(
        {
            BACKEND_BASE_URL_ENV: "http://127.0.0.1:8080",
            INTERNAL_SERVICE_TOKEN_ENV: TOKEN,
        }
    )
    assert settings.base_url == "http://127.0.0.1:8080"
    assert settings.token == TOKEN


def test_missing_token_refuses_to_build_settings():
    with pytest.raises(CollectorConfigError):
        load_backend_settings({BACKEND_BASE_URL_ENV: "http://127.0.0.1:8080"})


def test_missing_base_url_refuses_to_build_settings():
    with pytest.raises(CollectorConfigError):
        load_backend_settings({INTERNAL_SERVICE_TOKEN_ENV: TOKEN})


def test_short_token_is_refused_like_the_backend_does():
    with pytest.raises(CollectorConfigError):
        BackendSettings(base_url="http://127.0.0.1:8080", token="short")


def test_base_url_must_have_a_scheme():
    with pytest.raises(CollectorConfigError):
        BackendSettings(base_url="127.0.0.1:8080", token=TOKEN)


def test_token_is_never_in_the_repr():
    """객체가 로그에 실려도 토큰이 나오지 않아야 한다."""
    settings = BackendSettings(base_url="http://127.0.0.1:8080", token=TOKEN)
    assert TOKEN not in repr(settings)
    assert "redacted" in repr(settings)


def test_token_is_not_in_configuration_error_messages():
    try:
        BackendSettings(base_url="", token=TOKEN)
    except CollectorConfigError as error:
        assert TOKEN not in str(error)
    else:  # pragma: no cover - 위에서 반드시 터진다
        pytest.fail("설정 오류가 나야 합니다.")


def test_base_url_override_does_not_accept_a_token():
    """CLI 인자로 토큰을 받지 않는다. 명령줄은 히스토리에 남는다."""
    import inspect

    from cctv_collector import cli

    source = inspect.getsource(cli.build_parser)
    assert "--token" not in source
    assert "--service-token" not in source
