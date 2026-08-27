"""영상 파일만 빼고 전부 관통하는 통합 테스트.

opencv가 없어도 detector -> 추적 -> 계수 -> CSV/JSON까지는 그대로 돈다.
실제 영상 디코딩과 겹쳐 그리기만 빠진다. 결과 파일의 모양이 깨지는 것을
설치 여부와 무관하게 잡으려는 것이다.
"""

from __future__ import annotations

import json

import pytest

from cctv_counting.config import CctvConfig
from cctv_counting.observations import read_observations, write_observations
from cctv_counting.pipeline import CountingPipeline
from cctv_counting.summary import build_summary, write_json
from cctv_counting.windows import aggregate_windows

from conftest import ScriptedDetector, walker

FPS = 25.0
WIDTH = 1280
HEIGHT = 720


def _run(config: CctvConfig, script) -> tuple[CountingPipeline, list]:
    pipeline = CountingPipeline(
        detector=ScriptedDetector(script=script),
        config=config,
        frame_width=WIDTH,
        frame_height=HEIGHT,
    )
    observations = [
        pipeline.process_frame(index, index / FPS, object()).observation
        for index in range(len(script))
    ]
    return pipeline, observations


def _config(**overrides) -> CctvConfig:
    payload = {
        "name": "e2e",
        "detector": {"confidence_threshold": 0.3},
        "tracker": {"min_hits": 2, "max_age": 10},
        "line": {
            "enabled": True,
            "start": [0.0, 0.5],
            "end": [1.0, 0.5],
            "in_direction": "NEGATIVE",
            "cooldown_frames": 10,
        },
        "window": {"enabled": True, "seconds": 1.0},
    }
    payload.update(overrides)
    return CctvConfig.model_validate(payload)


def _one_person_walking_in(frames: int = 40):
    """한 사람이 화면 아래(y=0.9)에서 위(y=0.15)로 걸어 들어간다."""
    return walker(
        frames,
        start_y=HEIGHT * 0.9,
        end_y=HEIGHT * 0.15,
        x=WIDTH * 0.5,
        width=70.0,
        height=180.0,
    )


def test_tracking_baseline_writes_every_output(tmp_path):
    pipeline, observations = _run(_config(), _one_person_walking_in())

    csv_path = tmp_path / "observations.csv"
    write_observations(csv_path, observations)

    summary = build_summary(
        video="demo.mp4",
        observations=observations,
        duration_seconds=len(observations) / FPS,
        processing_fps=12.5,
        naive_line_band_detections=pipeline.naive_line_band_detections,
    )
    summary_path = tmp_path / "summary.json"
    write_json(summary_path, summary.to_dict())

    assert csv_path.exists() and summary_path.exists()

    # CSV가 다시 읽히고 내용이 보존된다.
    assert read_observations(csv_path) == observations

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["video"] == "demo.mp4"
    assert payload["framesProcessed"] == len(observations)
    assert payload["peakOccupancy"] == 1
    assert payload["entriesTotal"] == 1
    assert payload["exitsTotal"] == 0
    assert payload["naiveLineBandDetections"] is None


def test_detection_only_baseline_writes_empty_entry_columns(tmp_path):
    pipeline, observations = _run(
        _config(tracker={"enabled": False}), _one_person_walking_in()
    )

    csv_path = tmp_path / "observations.csv"
    write_observations(csv_path, observations)
    body = csv_path.read_text(encoding="utf-8").splitlines()[1]
    assert body.split(",")[3] == ""

    summary = build_summary(
        video="demo.mp4",
        observations=observations,
        duration_seconds=len(observations) / FPS,
        processing_fps=30.0,
        naive_line_band_detections=pipeline.naive_line_band_detections,
    )
    payload = summary.to_dict()
    assert payload["entriesTotal"] is None
    assert payload["naiveLineBandDetections"] is not None

    # 이 값은 입장 인원이 아니다. 실제로 지나간 사람은 한 명이다.
    assert payload["naiveLineBandDetections"] >= 1


def test_two_baselines_agree_on_who_is_in_the_room():
    """occupancy는 둘 다 낼 수 있다. entry는 한쪽만 낸다."""
    script = _one_person_walking_in()
    _, tracked = _run(_config(), script)
    _, detected = _run(_config(tracker={"enabled": False}), script)

    assert max(o.occupancy_count for o in tracked) == 1
    assert max(o.occupancy_count for o in detected) == 1
    assert tracked[-1].entries_total == 1
    assert detected[-1].entries_total is None


def test_window_output_is_derived_from_the_same_observations():
    _, observations = _run(_config(), _one_person_walking_in(50))
    windows = aggregate_windows(observations, 1.0)
    assert windows
    assert sum(w.frame_count for w in windows) == len(observations)
    assert all(w.max_occupancy <= 1 for w in windows)


def test_summary_survives_a_video_where_nobody_appears(tmp_path):
    """빈 식당도 유효한 결과다. 다만 낮은 신뢰로 표시된다."""
    _, observations = _run(_config(), [[] for _ in range(20)])
    summary = build_summary(
        video="empty.mp4",
        observations=observations,
        duration_seconds=20 / FPS,
        processing_fps=40.0,
    )
    assert summary.peak_occupancy == 0
    assert summary.low_reliability_frame_ratio == pytest.approx(1.0)
    write_json(tmp_path / "summary.json", summary.to_dict())


def test_running_two_videos_needs_a_reset_between_them():
    """번호와 누적 인원이 영상 경계를 넘지 않는다."""
    pipeline, first = _run(_config(), _one_person_walking_in())
    assert first[-1].entries_total == 1

    pipeline.reset()
    pipeline.detector = ScriptedDetector(script=_one_person_walking_in())
    second = [
        pipeline.process_frame(index, index / FPS, object()).observation
        for index in range(40)
    ]
    assert second[0].entries_total == 0
    assert second[-1].entries_total == 1
