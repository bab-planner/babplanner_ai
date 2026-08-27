"""detector -> 추적 -> 계수 전체 흐름.

실제 모델 없이 가짜 detector로 돌린다. 계수 로직이 맞는지는 모델이 무엇을
보는지와 무관하게 검증되어야 한다.
"""

from __future__ import annotations

import pytest

from cctv_counting.config import CctvConfig
from cctv_counting.pipeline import CountingPipeline, PipelineError
from cctv_counting.reliability import ReliabilityFlag

from conftest import ScriptedDetector, person, walker

WIDTH = 1000
HEIGHT = 1000


def _config(**overrides) -> CctvConfig:
    payload: dict = {
        "name": "test",
        "detector": {"confidence_threshold": 0.3},
        "tracker": {"min_hits": 1, "max_age": 5},
        "line": {
            "enabled": True,
            "start": [0.0, 0.6],
            "end": [1.0, 0.6],
            "in_direction": "NEGATIVE",
            "band": 0.01,
            "min_confirm_distance": 0.03,
            "cooldown_seconds": 0.2,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return CctvConfig.model_validate(payload)


def _run(script, config: CctvConfig) -> tuple[CountingPipeline, list]:
    pipeline = CountingPipeline(
        detector=ScriptedDetector(script=script),
        config=config,
        frame_width=WIDTH,
        frame_height=HEIGHT,
    )
    results = [
        pipeline.process_frame(index, index / 25.0, object())
        for index in range(len(script))
    ]
    return pipeline, results


def test_zero_detections_give_zero_occupancy():
    _, results = _run([[], [], []], _config())
    assert [r.observation.occupancy_count for r in results] == [0, 0, 0]
    assert results[0].observation.reliability_flag is ReliabilityFlag.NO_DETECTION


def test_multiple_people_are_counted():
    frame = [
        person(100, 400, 160, 700),
        person(300, 400, 360, 700),
        person(600, 400, 660, 700),
    ]
    _, results = _run([frame, frame], _config())
    assert results[-1].observation.occupancy_count == 3


def test_occupancy_is_not_cumulative():
    """지금 있는 사람 수와 지금까지 들어온 사람 수를 절대 섞지 않는다."""
    script = walker(20, start_y=900.0, end_y=200.0)
    _, results = _run(script, _config())
    assert max(r.observation.occupancy_count for r in results) == 1
    assert results[-1].observation.entries_total == 1


def test_walking_across_the_line_registers_one_entry():
    script = walker(20, start_y=900.0, end_y=200.0)
    _, results = _run(script, _config())
    final = results[-1].observation
    assert final.entries_total == 1
    assert final.exits_total == 0


def test_walking_the_other_way_registers_one_exit():
    script = walker(20, start_y=200.0, end_y=900.0)
    _, results = _run(script, _config())
    final = results[-1].observation
    assert final.entries_total == 0
    assert final.exits_total == 1


def test_entries_are_cumulative_and_monotonic():
    script = walker(20, start_y=900.0, end_y=200.0)
    _, results = _run(script, _config())
    totals = [r.observation.entries_total for r in results]
    assert totals == sorted(totals)
    assert totals[0] == 0


def test_a_person_who_never_reaches_the_line_is_not_counted():
    script = walker(20, start_y=950.0, end_y=700.0)
    _, results = _run(script, _config())
    assert results[-1].observation.entries_total == 0


def test_roi_excludes_people_outside_it():
    """창밖을 지나가는 사람이 식당 혼잡도에 들어오면 안 된다."""
    outside = [person(100, 50, 160, 200)]  # 발이 y=200 -> 0.2
    config = _config(roi={"enabled": True, "x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 1.0})
    _, results = _run([outside, outside, outside], config)
    assert all(r.observation.occupancy_count == 0 for r in results)


def test_roi_includes_people_inside_it():
    inside = [person(100, 400, 160, 700)]  # 발이 y=700 -> 0.7
    config = _config(roi={"enabled": True, "x1": 0.0, "y1": 0.5, "x2": 1.0, "y2": 1.0})
    _, results = _run([inside, inside], config)
    assert results[-1].observation.occupancy_count == 1


def test_jitter_near_the_line_does_not_inflate_entries():
    """한 사람이 선 위에서 흔들려도 입장은 한 번이다."""
    script = []
    for y in [900.0, 850.0, 800.0, 700.0, 640.0]:
        script.append([person(470, y - 160, 530, y)])
    for _ in range(12):
        script.append([person(470, 601 - 160, 530, 601)])
        script.append([person(470, 599 - 160, 530, 599)])
    for y in [560.0, 500.0, 400.0]:
        script.append([person(470, y - 160, 530, y)])
    _, results = _run(script, _config())
    assert results[-1].observation.entries_total == 1


def test_detection_only_baseline_reports_no_entries():
    """추적 없이는 '넘었다'는 사건이 정의되지 않는다."""
    script = walker(20, start_y=900.0, end_y=200.0)
    pipeline, results = _run(script, _config(tracker={"enabled": False}))
    final = results[-1].observation
    assert final.entries_total is None
    assert final.exits_total is None
    assert final.occupancy_count == 1


def test_detection_only_baseline_records_the_naive_proxy():
    """이 값이 실제 입장 인원 1보다 크다는 것이 추적이 하는 일이다."""
    script = walker(60, start_y=900.0, end_y=200.0)
    pipeline, _ = _run(script, _config(tracker={"enabled": False}))
    assert pipeline.naive_line_band_detections is not None
    assert pipeline.naive_line_band_detections >= 1


def test_tracking_baseline_has_no_naive_proxy():
    script = walker(10, start_y=900.0, end_y=200.0)
    pipeline, _ = _run(script, _config())
    assert pipeline.naive_line_band_detections is None


def test_low_confidence_frames_are_flagged():
    weak = [person(100, 400, 160, 700, confidence=0.35)]
    config = _config(
        detector={"confidence_threshold": 0.3},
        tracker={"min_hits": 1, "high_threshold": 0.3},
        reliability={"min_mean_confidence": 0.6},
    )
    _, results = _run([weak, weak], config)
    assert results[-1].observation.reliability_flag is ReliabilityFlag.LOW_CONFIDENCE


def test_confidence_threshold_changes_what_is_counted():
    """임계값이 실험 축이 되려면 설정만 바꿔도 결과가 달라져야 한다."""
    frame = [person(100, 400, 160, 700, confidence=0.35)]
    strict = _config(detector={"confidence_threshold": 0.5})
    loose = _config(
        detector={"confidence_threshold": 0.2},
        tracker={"min_hits": 1, "high_threshold": 0.3},
    )
    _, strict_results = _run([frame, frame], strict)
    _, loose_results = _run([frame, frame], loose)
    assert strict_results[-1].observation.occupancy_count == 0
    assert loose_results[-1].observation.occupancy_count == 1


def test_timestamps_are_carried_through():
    _, results = _run([[], []], _config())
    assert results[1].observation.timestamp_seconds == pytest.approx(0.04)


def test_reset_starts_the_next_video_from_scratch():
    """track 번호도 누적 인원도 영상 경계를 넘지 않는다."""
    script = walker(20, start_y=900.0, end_y=200.0)
    pipeline, _ = _run(script, _config())
    pipeline.reset()
    fresh = pipeline.process_frame(0, 0.0, object())
    assert fresh.observation.entries_total == 0
    assert fresh.observation.occupancy_count == 0


def test_frame_results_keep_track_ids_out_of_the_observation():
    """관측(파일로 나가는 것)에는 번호가 없어야 한다."""
    script = walker(6, start_y=900.0, end_y=800.0)
    _, results = _run(script, _config())
    assert results[-1].tracked  # 추적은 되고 있다
    assert not hasattr(results[-1].observation, "track_id")


def test_bad_frame_size_is_rejected():
    with pytest.raises(PipelineError):
        CountingPipeline(
            detector=ScriptedDetector(script=[]),
            config=_config(),
            frame_width=0,
            frame_height=100,
        )


def test_non_person_detections_never_reach_counting():
    chairs = [person(100, 400, 160, 700, class_id=62)]
    _, results = _run([chairs, chairs], _config())
    assert results[-1].observation.occupancy_count == 0


def test_warmup_frames_do_not_count_detections_yet():
    """min_hits를 채우기 전에는 상자가 보여도 occupancy는 0이다.

    실제 영상 첫 프레임에서 관찰한 동작이다. 겹쳐 그리기가 이때 원시 상자를
    굵게 그리면 "상자 4개 + CURRENT 0"이 되어 계수가 고장 난 것처럼 보인다.
    """
    frame = [person(100, 400, 160, 700)]
    config = _config(tracker={"min_hits": 3})
    _, results = _run([frame, frame, frame], config)

    assert results[0].observation.occupancy_count == 0
    assert results[0].detections, "detector는 상자를 냈다"
    assert results[0].tracked == [], "아직 확정된 track은 없다"
    assert results[0].observation.reliability_flag is ReliabilityFlag.NO_DETECTION
    # 세 번째 프레임에서 확정된다.
    assert results[2].observation.occupancy_count == 1


def test_pending_detections_are_distinguishable_from_counted_ones():
    """겹쳐 그리기가 둘을 구분할 수 있도록 결과가 분리되어 있어야 한다."""
    frame = [person(100, 400, 160, 700)]
    _, results = _run([frame, frame, frame], _config(tracker={"min_hits": 3}))

    warmup = results[0]
    pending = [d.bbox for d in warmup.detections if d.bbox not in [p.bbox for p in warmup.tracked]]
    assert len(pending) == 1

    settled = results[2]
    still_pending = [
        d.bbox for d in settled.detections if d.bbox not in [p.bbox for p in settled.tracked]
    ]
    assert still_pending == [], "확정된 뒤에는 대기 중인 상자가 없어야 한다"
