"""CSV 스키마, JSON 요약, 시간 창."""

from __future__ import annotations

import json

import pytest

from cctv_counting.observations import (
    FORBIDDEN_FIELD_TOKENS,
    OBSERVATION_FIELDS,
    FrameObservation,
    ObservationError,
    read_observations,
    write_observations,
)
from cctv_counting.reliability import ReliabilityFlag
from cctv_counting.summary import (
    SummaryError,
    build_summary,
    reject_non_finite,
    write_json,
)
from cctv_counting.windows import WindowError, aggregate_windows


def observation(
    frame_index: int = 0,
    timestamp: float = 0.0,
    occupancy: int = 0,
    entries: int | None = 0,
    exits: int | None = 0,
    mean_confidence: float | None = 0.9,
    flag: ReliabilityFlag = ReliabilityFlag.OK,
) -> FrameObservation:
    return FrameObservation(
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        occupancy_count=occupancy,
        entries_total=entries,
        exits_total=exits,
        mean_detection_confidence=mean_confidence,
        min_detection_confidence=mean_confidence,
        active_track_count=occupancy,
        reliability_flag=flag,
    )


# --- CSV 스키마 ---------------------------------------------------------


def test_csv_columns_are_fixed():
    assert OBSERVATION_FIELDS == (
        "frame_index",
        "timestamp_seconds",
        "occupancy_count",
        "entries_total",
        "exits_total",
        "mean_detection_confidence",
        "min_detection_confidence",
        "active_track_count",
        "reliability_flag",
    )


def test_csv_schema_carries_no_identifying_column():
    """track_id를 시각과 함께 파일로 남기면 그것은 사람별 동선 기록이 된다."""
    for field in OBSERVATION_FIELDS:
        for token in FORBIDDEN_FIELD_TOKENS:
            assert token not in field, f"{field}에 {token}이 들어 있다"


def test_csv_round_trip(tmp_path):
    rows = [
        observation(0, 0.0, 3, 1, 0),
        observation(1, 0.04, 4, 2, 1, flag=ReliabilityFlag.LOW_CONFIDENCE),
    ]
    path = tmp_path / "observations.csv"
    write_observations(path, rows)
    assert read_observations(path) == rows


def test_csv_writes_header_even_with_no_rows(tmp_path):
    path = tmp_path / "observations.csv"
    write_observations(path, [])
    assert path.read_text(encoding="utf-8").strip() == ",".join(OBSERVATION_FIELDS)


def test_missing_entry_counts_become_empty_not_zero(tmp_path):
    """0은 '아무도 안 들어왔다'이고, 빈 칸은 '재지 않았다'이다."""
    path = tmp_path / "observations.csv"
    write_observations(path, [observation(entries=None, exits=None)])
    line = path.read_text(encoding="utf-8").splitlines()[1]
    assert line.split(",")[3] == ""
    assert line.split(",")[4] == ""
    assert read_observations(path)[0].entries_total is None


def test_missing_confidence_becomes_empty(tmp_path):
    path = tmp_path / "observations.csv"
    write_observations(path, [observation(mean_confidence=None, flag=ReliabilityFlag.NO_DETECTION)])
    assert read_observations(path)[0].mean_detection_confidence is None


def test_reading_a_file_with_wrong_columns_fails(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("frame_index,occupancy_count\n0,1\n", encoding="utf-8")
    with pytest.raises(ObservationError):
        read_observations(path)


def test_observation_rejects_negative_occupancy():
    with pytest.raises(ObservationError):
        observation(occupancy=-1)


def test_observation_rejects_nan_timestamp():
    with pytest.raises(ObservationError):
        observation(timestamp=float("nan"))


def test_observation_rejects_confidence_out_of_range():
    with pytest.raises(ObservationError):
        observation(mean_confidence=1.4)


# --- summary.json -------------------------------------------------------


def test_summary_peak_is_the_maximum():
    rows = [observation(i, i * 0.04, occupancy=count) for i, count in enumerate([2, 7, 3])]
    summary = build_summary(
        video="a.mp4", observations=rows, duration_seconds=0.12, processing_fps=25.0
    )
    assert summary.peak_occupancy == 7


def test_summary_average_is_the_mean():
    rows = [observation(i, i * 0.04, occupancy=count) for i, count in enumerate([2, 7, 3])]
    summary = build_summary(
        video="a.mp4", observations=rows, duration_seconds=0.12, processing_fps=25.0
    )
    assert summary.average_occupancy == pytest.approx(4.0)


def test_summary_entries_come_from_the_last_frame():
    """누적값이므로 마지막 프레임이 곧 전체 이용 인원이다."""
    rows = [observation(0, 0.0, entries=0), observation(1, 0.04, entries=9, exits=4)]
    summary = build_summary(
        video="a.mp4", observations=rows, duration_seconds=0.08, processing_fps=None
    )
    assert summary.entries_total == 9
    assert summary.exits_total == 4


def test_summary_reports_low_reliability_ratio():
    rows = [
        observation(0, 0.0),
        observation(1, 0.04, flag=ReliabilityFlag.LOW_CONFIDENCE),
        observation(2, 0.08, flag=ReliabilityFlag.NO_DETECTION, mean_confidence=None),
        observation(3, 0.12),
    ]
    summary = build_summary(
        video="a.mp4", observations=rows, duration_seconds=0.16, processing_fps=None
    )
    assert summary.low_reliability_frame_ratio == pytest.approx(0.5)


def test_summary_refuses_to_describe_a_video_it_never_read():
    """0명, 0명이 나가면 못 읽은 영상이 빈 식당으로 보인다."""
    with pytest.raises(SummaryError):
        build_summary(
            video="a.mp4", observations=[], duration_seconds=0.0, processing_fps=None
        )


def test_summary_keys_are_camel_case_for_the_backend():
    summary = build_summary(
        video="a.mp4",
        observations=[observation()],
        duration_seconds=1.0,
        processing_fps=30.0,
    )
    payload = summary.to_dict()
    assert set(payload) == {
        "video",
        "durationSeconds",
        "framesProcessed",
        "peakOccupancy",
        "averageOccupancy",
        "entriesTotal",
        "exitsTotal",
        "processingFps",
        "lowReliabilityFrameRatio",
        "naiveLineBandDetections",
    }


def test_json_write_round_trips(tmp_path):
    summary = build_summary(
        video="a.mp4",
        observations=[observation(occupancy=3)],
        duration_seconds=1.0,
        processing_fps=30.0,
    )
    path = tmp_path / "summary.json"
    write_json(path, summary.to_dict())
    assert json.loads(path.read_text(encoding="utf-8"))["peakOccupancy"] == 3


def test_nan_is_rejected_before_it_reaches_the_file():
    with pytest.raises(SummaryError) as error:
        reject_non_finite({"averageOccupancy": float("nan")})
    assert "averageOccupancy" in str(error.value)


def test_infinity_is_rejected():
    with pytest.raises(SummaryError):
        reject_non_finite({"processingFps": float("inf")})


def test_nan_nested_in_a_list_is_found():
    with pytest.raises(SummaryError) as error:
        reject_non_finite({"windows": [{"mean": 1.0}, {"mean": float("nan")}]})
    assert "[1]" in str(error.value)


def test_write_json_refuses_non_finite(tmp_path):
    with pytest.raises(SummaryError):
        write_json(tmp_path / "bad.json", {"value": float("nan")})


def test_none_is_allowed_because_unknown_is_not_broken():
    reject_non_finite({"entriesTotal": None})


# --- 시간 창 ------------------------------------------------------------


def test_windows_group_by_absolute_time():
    rows = [observation(i, i * 0.5, occupancy=i) for i in range(8)]
    windows = aggregate_windows(rows, window_seconds=2.0)
    assert [(w.start_seconds, w.end_seconds) for w in windows] == [(0.0, 2.0), (2.0, 4.0)]


def test_window_statistics():
    rows = [
        observation(0, 0.0, occupancy=2),
        observation(1, 0.5, occupancy=8),
        observation(2, 1.0, occupancy=2),
    ]
    window = aggregate_windows(rows, window_seconds=2.0)[0]
    assert window.frame_count == 3
    assert window.mean_occupancy == pytest.approx(4.0)
    assert window.median_occupancy == pytest.approx(2.0)
    assert window.max_occupancy == 8


def test_windows_with_no_observations_are_not_invented():
    """영상이 끊긴 구간이 '아무도 없던 구간'으로 둔갑하면 안 된다."""
    rows = [observation(0, 0.0, occupancy=3), observation(1, 10.0, occupancy=4)]
    windows = aggregate_windows(rows, window_seconds=2.0)
    assert len(windows) == 2
    assert windows[1].start_seconds == pytest.approx(10.0)


def test_window_low_reliability_ratio():
    rows = [
        observation(0, 0.0),
        observation(1, 0.5, flag=ReliabilityFlag.LOW_CONFIDENCE),
    ]
    assert aggregate_windows(rows, 2.0)[0].low_reliability_ratio == pytest.approx(0.5)


def test_windows_without_tracking_have_no_entry_count():
    rows = [observation(0, 0.0, entries=None, exits=None)]
    assert aggregate_windows(rows, 2.0)[0].entries_in_window is None


def test_empty_observations_yield_no_windows():
    assert aggregate_windows([], 2.0) == []


def test_window_seconds_must_be_positive():
    with pytest.raises(WindowError):
        aggregate_windows([observation()], 0.0)


def test_window_entries_are_the_increase_within_that_window():
    """창 값은 누적이 아니라 그 창에서 늘어난 양이다."""
    rows = [
        observation(0, 0.0, entries=0),
        observation(1, 1.5, entries=3),
        observation(2, 2.5, entries=3),
        observation(3, 3.5, entries=7),
        observation(4, 4.5, entries=9),
    ]
    windows = aggregate_windows(rows, window_seconds=2.0)
    assert [w.entries_in_window for w in windows] == [3, 4, 2]
    assert sum(w.entries_in_window for w in windows) == rows[-1].entries_total
