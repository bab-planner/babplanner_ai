"""수집기 흐름과 Spring 전송.

detector는 가짜다. 모델도 opencv도 없이 전체 흐름을 그대로 돌린다. 이
파일이 지키는 것은 계수 정확도가 아니라 **파이프라인의 규약**이다.

    detector를 프레임당 한 번만 부른다
    표본을 간격대로 모아서 보낸다
    observedAt은 영상 시각이 아니라 실제 시각이다
    dry-run은 아무것도 보내지 않는다
    전송 실패를 추론 실패로 적지 않는다
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from cctv_counting.detection import COCO_PERSON_CLASS_ID, Detection
from cctv_counting.geometry import BoundingBox

from cctv_collector.collector import Collector, CollectorError, Sample, aggregate
from cctv_collector.config import (
    MIN_SERVICE_TOKEN_LENGTH,
    BackendSettings,
    adopted_research_policy,
    camera_config_from_dict,
)
from cctv_collector.congestion_policy import CongestionLevel
from cctv_collector.schemas import CongestionObservation
from cctv_collector.spring_client import (
    EXPECTED_STATUS,
    INGEST_PATH,
    SERVICE_TOKEN_HEADER,
    SendStatus,
    SpringCongestionClient,
)

TOKEN = "t" * MIN_SERVICE_TOKEN_LENGTH
FPS = 25.0
START = datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone(timedelta(hours=9)))


def camera(**overrides):
    payload = {
        "site_id": 1,
        "device_id": "demo-cctv-01",
        "model_version": "cctv-demo-v1",
        "frame": {"width": 384, "height": 288},
        "roi": {"enabled": True, "x1": 0.0, "y1": 0.20, "x2": 1.0, "y2": 0.95},
        "line": {"start": [0.5, 0.20], "end": [0.5, 0.95], "in_direction": "POSITIVE"},
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
    return camera_config_from_dict(payload)


class CountingDetector:
    """프레임마다 정해진 상자를 낸다. 호출 횟수를 센다."""

    def __init__(self, per_frame: list[list[Detection]]) -> None:
        self.per_frame = per_frame
        self.calls = 0

    def detect(self, frame: object) -> list[Detection]:
        boxes = self.per_frame[self.calls] if self.calls < len(self.per_frame) else []
        self.calls += 1
        return list(boxes)


def person(x: float, confidence: float = 0.95) -> Detection:
    return Detection(
        BoundingBox(x, 120.0, x + 30.0, 220.0), confidence, COCO_PERSON_CLASS_ID
    )


def crowd(count: int, confidence: float = 0.95) -> list[Detection]:
    return [person(20.0 + index * 34.0, confidence) for index in range(count)]


def frames(script: list[list[Detection]], fps: float = FPS):
    for index in range(len(script)):
        yield index, index / fps, object()


def fixed_clock(step_seconds: float = 0.0):
    state = {"n": 0}

    def clock() -> datetime:
        value = START + timedelta(seconds=step_seconds * state["n"])
        state["n"] += 1
        return value

    return clock


# --- 표본 집계 --------------------------------------------------------------


def samples(counts: list[int]) -> list[Sample]:
    return [
        Sample(
            frame_index=index,
            video_timestamp_seconds=index / FPS,
            occupancy_count=count,
            entries_total=0,
            exits_total=0,
        )
        for index, count in enumerate(counts)
    ]


def test_latest_takes_the_last_sample():
    assert aggregate(samples([3, 9, 4]), "latest") == 4


def test_median_is_an_integer_and_does_not_invent_people():
    """짝수 개일 때 올림으로 사람을 만들어 내지 않는다."""
    assert aggregate(samples([2, 4]), "median") == 2
    assert aggregate(samples([1, 5, 9]), "median") == 5


def test_aggregation_refuses_an_empty_window():
    with pytest.raises(CollectorError):
        aggregate([], "latest")


def test_unknown_aggregation_is_refused():
    with pytest.raises(CollectorError):
        aggregate(samples([1]), "adaptive")


# --- 공유 detector ----------------------------------------------------------


def test_detector_runs_exactly_once_per_frame():
    """추론을 두 번 돌리면 서비스에서 비용이 그대로 두 배가 된다."""
    script = [crowd(3) for _ in range(50)]
    detector = CountingDetector(script)
    collector = Collector(
        camera=camera(),
        policy=adopted_research_policy()[0],
        detector=detector,
        clock=fixed_clock(),
    )
    list(collector.run(frames(script)))

    assert detector.calls == 50
    assert collector.detector_calls == 50
    assert collector.stats.frames == 50
    assert collector.stats.detector_calls == 50


def test_both_branches_consume_the_same_inference():
    """occupancy가 값을 내고 entry 갈래도 살아 있어야 공유의 의미가 있다."""
    script = []
    for index in range(60):
        # 오른쪽에서 왼쪽으로 걸어간다. 선(x=0.5)을 넘는다.
        script.append([person(300.0 - index * 4.0)])
    detector = CountingDetector(script)
    collector = Collector(
        camera=camera(),
        policy=adopted_research_policy()[0],
        detector=detector,
        clock=fixed_clock(),
    )
    records = list(collector.run(frames(script)))

    assert detector.calls == 60
    assert any(record.people_count > 0 for record in records)
    assert records[-1].entries_total >= 1


# --- 표본 간격 --------------------------------------------------------------


def test_one_observation_per_interval_not_per_frame():
    """25fps 영상이라고 초당 25번 관측을 만들지 않는다."""
    script = [crowd(2) for _ in range(75)]  # 3초
    collector = Collector(
        camera=camera(),
        policy=adopted_research_policy()[0],
        detector=CountingDetector(script),
        clock=fixed_clock(),
    )
    records = list(collector.run(frames(script)))
    assert len(records) == 3
    assert all(record.samples > 1 for record in records)


def test_a_shorter_interval_makes_more_observations():
    script = [crowd(2) for _ in range(50)]  # 2초
    config = camera(
        observation={"interval_seconds": 0.5, "aggregation": "latest"}
    )
    collector = Collector(
        camera=config,
        policy=adopted_research_policy()[0],
        detector=CountingDetector(script),
        clock=fixed_clock(),
    )
    assert len(list(collector.run(frames(script)))) == 4


def test_a_trailing_partial_window_still_produces_an_observation():
    """짧은 데모 영상에서 관측이 하나도 안 나가는 일을 막는다."""
    script = [crowd(1) for _ in range(10)]  # 0.4초. 창보다 짧다.
    collector = Collector(
        camera=camera(),
        policy=adopted_research_policy()[0],
        detector=CountingDetector(script),
        clock=fixed_clock(),
    )
    records = list(collector.run(frames(script)))
    assert len(records) == 1
    assert records[0].samples == 10


# --- 두 시간축 --------------------------------------------------------------


def test_observed_at_is_wall_clock_not_the_video_timestamp():
    """녹화 영상의 시각을 보내면 백엔드가 오래된 관측으로 버린다."""
    script = [crowd(2) for _ in range(50)]
    collector = Collector(
        camera=camera(),
        policy=adopted_research_policy()[0],
        detector=CountingDetector(script),
        clock=fixed_clock(step_seconds=1.0),
    )
    records = list(collector.run(frames(script)))

    assert records[0].observed_at == START
    assert records[1].observed_at == START + timedelta(seconds=1)
    # 영상 시각은 따로 남는다. 둘이 같은 값이 아니다.
    assert records[0].video_timestamp_seconds != records[0].observed_at.timestamp()


def test_video_timestamp_is_kept_as_local_diagnostic_only():
    script = [crowd(2) for _ in range(50)]
    collector = Collector(
        camera=camera(),
        policy=adopted_research_policy()[0],
        detector=CountingDetector(script),
        clock=fixed_clock(),
    )
    record = list(collector.run(frames(script)))[0]

    # 창은 간격에 도달한 프레임까지 포함하고 닫힌다. 25fps에서 1초 창의
    # 마지막 표본은 프레임 25(t=1.0)다.
    assert record.video_timestamp_seconds == pytest.approx(1.0)
    assert record.samples == 26
    payload = CongestionObservation(
        site_id=record.site_id,
        device_id=record.device_id,
        people_count=record.people_count,
        level=record.level,
        observed_at=record.observed_at,
    ).to_payload()
    assert "videoTimestampSeconds" not in payload


# --- dry-run과 live ---------------------------------------------------------


def test_dry_run_sends_nothing():
    script = [crowd(2) for _ in range(50)]
    collector = Collector(
        camera=camera(),
        policy=adopted_research_policy()[0],
        detector=CountingDetector(script),
        sender=None,
        clock=fixed_clock(),
    )
    records = list(collector.run(frames(script)))

    assert all(record.send_status == SendStatus.SKIPPED_DRY_RUN.value for record in records)
    assert collector.stats.sent == 0
    assert collector.stats.send_failed == 0


def test_dry_run_works_without_a_congestion_policy():
    """임계값이 없어도 인원수는 볼 수 있다. 단계만 비어 있다."""
    payload = {
        "site_id": 1,
        "device_id": "demo-cctv-01",
        "frame": {"width": 384, "height": 288},
        "line": {"start": [0.5, 0.20], "end": [0.5, 0.95]},
    }
    config = camera_config_from_dict(payload)
    script = [crowd(2) for _ in range(30)]
    collector = Collector(
        camera=config,
        policy=adopted_research_policy()[0],
        detector=CountingDetector(script),
        clock=fixed_clock(),
    )
    records = list(collector.run(frames(script)))
    assert records[0].level is None
    assert records[0].people_count == 2


def test_live_send_requires_a_congestion_policy():
    """임계값 없이 단계를 만들려면 코드가 숫자를 지어내야 한다. 거부한다."""
    payload = {
        "site_id": 1,
        "device_id": "demo-cctv-01",
        "frame": {"width": 384, "height": 288},
        "line": {"start": [0.5, 0.20], "end": [0.5, 0.95]},
    }
    with pytest.raises(CollectorError):
        Collector(
            camera=camera_config_from_dict(payload),
            policy=adopted_research_policy()[0],
            detector=CountingDetector([]),
            sender=lambda observation: None,
        )


def test_cli_refuses_live_send_without_thresholds(tmp_path, capsys):
    from cctv_collector import cli

    config = tmp_path / "camera.json"
    config.write_text(
        json.dumps(
            {
                "site_id": 1,
                "device_id": "demo-cctv-01",
                "frame": {"width": 384, "height": 288},
                "line": {"start": [0.5, 0.20], "end": [0.5, 0.95]},
            }
        ),
        encoding="utf-8",
    )
    code = cli.main(
        ["--video", str(tmp_path / "none.mp4"), "--config", str(config), "--send"]
    )
    assert code == 2
    assert "임계값" in capsys.readouterr().err


def test_cli_default_is_dry_run():
    from cctv_collector import cli

    args = cli.build_parser().parse_args(["--video", "a.mp4", "--config", "b.yaml"])
    assert args.send is False


# --- Spring 전송 ------------------------------------------------------------


def settings() -> BackendSettings:
    return BackendSettings(base_url="http://127.0.0.1:8080", token=TOKEN)


def observation() -> CongestionObservation:
    return CongestionObservation(
        site_id=1,
        device_id="demo-cctv-01",
        people_count=8,
        level=CongestionLevel.NORMAL,
        observed_at=START,
        model_version="cctv-demo-v1",
    )


def mock_spring(status: int = EXPECTED_STATUS, seen: list | None = None):
    """백엔드 계약을 그대로 흉내 내는 가짜 서버."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        assert request.url.path == INGEST_PATH
        if request.headers.get(SERVICE_TOKEN_HEADER) != TOKEN:
            return httpx.Response(401)
        return httpx.Response(status)

    return httpx.MockTransport(handler)


def test_successful_send_posts_the_contract_request():
    seen: list[httpx.Request] = []
    client = SpringCongestionClient(settings=settings(), transport=mock_spring(seen=seen))
    result = client.send(observation())

    assert result.status is SendStatus.SUCCESS
    assert result.http_status == 202
    assert result.attempts == 1

    request = seen[0]
    assert request.method == "POST"
    assert str(request.url) == "http://127.0.0.1:8080" + INGEST_PATH
    assert request.headers[SERVICE_TOKEN_HEADER] == TOKEN
    assert json.loads(request.content.decode()) == {
        "siteId": 1,
        "deviceId": "demo-cctv-01",
        "peopleCount": 8,
        "level": "NORMAL",
        "observedAt": "2026-08-28T12:00:00.000+09:00",
        "modelVersion": "cctv-demo-v1",
    }


def test_a_rejected_token_is_a_backend_failure_not_an_inference_failure():
    bad = BackendSettings(base_url="http://127.0.0.1:8080", token="x" * 40)
    client = SpringCongestionClient(settings=bad, transport=mock_spring())
    result = client.send(observation())

    assert result.status is SendStatus.BACKEND_SEND_FAILED
    assert result.http_status == 401
    assert "INFERENCE" not in (result.error or "")


def test_error_messages_never_carry_the_token():
    bad = BackendSettings(base_url="http://127.0.0.1:8080", token="x" * 40)
    client = SpringCongestionClient(settings=bad, transport=mock_spring())
    result = client.send(observation())
    assert "x" * 40 not in (result.error or "")
    assert TOKEN not in (result.error or "")


def test_client_error_is_not_retried():
    """4xx는 다시 보내도 같다. 백엔드를 두 번 때리지 않는다."""
    seen: list[httpx.Request] = []
    client = SpringCongestionClient(
        settings=settings(), transport=mock_spring(400, seen=seen), max_retries=3
    )
    result = client.send(observation())
    assert result.status is SendStatus.BACKEND_SEND_FAILED
    assert len(seen) == 1


def test_server_error_is_retried_but_bounded():
    seen: list[httpx.Request] = []
    client = SpringCongestionClient(
        settings=settings(), transport=mock_spring(503, seen=seen), max_retries=1
    )
    result = client.send(observation())

    assert result.status is SendStatus.BACKEND_SEND_FAILED
    assert result.attempts == 2
    assert len(seen) == 2  # 무한 재시도가 아니다


def test_a_timeout_is_reported_as_a_backend_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timeout", request=request)

    client = SpringCongestionClient(
        settings=settings(), transport=httpx.MockTransport(handler), max_retries=0
    )
    result = client.send(observation())

    assert result.status is SendStatus.BACKEND_SEND_FAILED
    assert "시간 초과" in (result.error or "")
    assert TOKEN not in (result.error or "")


def test_a_dead_backend_does_not_raise_into_the_video_loop():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = SpringCongestionClient(
        settings=settings(), transport=httpx.MockTransport(handler), max_retries=0
    )
    result = client.send(observation())
    assert result.status is SendStatus.BACKEND_SEND_FAILED
    assert TOKEN not in (result.error or "")


def test_timeouts_are_configured_explicitly():
    client = SpringCongestionClient(settings=settings())
    assert client.connect_timeout > 0
    assert client.read_timeout > 0
    with pytest.raises(ValueError):
        SpringCongestionClient(settings=settings(), connect_timeout=0)
    with pytest.raises(ValueError):
        SpringCongestionClient(settings=settings(), max_retries=-1)


def test_collector_records_backend_failures_without_stopping():
    """백엔드가 죽어도 영상 처리는 끝까지 간다."""
    script = [crowd(2) for _ in range(75)]
    client = SpringCongestionClient(
        settings=settings(), transport=mock_spring(500), max_retries=0
    )
    collector = Collector(
        camera=camera(),
        policy=adopted_research_policy()[0],
        detector=CountingDetector(script),
        sender=client.send,
        clock=fixed_clock(),
    )
    records = list(collector.run(frames(script)))

    assert len(records) == 3
    assert collector.stats.frames == 75
    assert collector.stats.send_failed == 3
    assert collector.stats.sent == 0
    assert all(
        record.send_status == SendStatus.BACKEND_SEND_FAILED.value for record in records
    )


def test_collector_end_to_end_against_a_mock_spring():
    """가짜 백엔드로 전체 흐름을 한 번 통과시킨다."""
    seen: list[httpx.Request] = []
    script = [crowd(8) for _ in range(75)]
    client = SpringCongestionClient(settings=settings(), transport=mock_spring(seen=seen))
    collector = Collector(
        camera=camera(),
        policy=adopted_research_policy()[0],
        detector=CountingDetector(script),
        sender=client.send,
        clock=fixed_clock(step_seconds=1.0),
    )
    records = list(collector.run(frames(script)))

    assert collector.stats.sent == 3
    assert [record.level for record in records] == [CongestionLevel.NORMAL] * 3
    bodies = [json.loads(request.content.decode()) for request in seen]
    assert [body["peopleCount"] for body in bodies] == [8, 8, 8]
    assert all(body["level"] == "NORMAL" for body in bodies)
    assert all("entriesTotal" not in body for body in bodies)
