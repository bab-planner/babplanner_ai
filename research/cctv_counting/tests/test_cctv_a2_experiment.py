"""Stage A2 실험 자산이 스스로 무너지지 않게 고정한다.

실제 모델 추론은 여기서 돌리지 않는다. torch가 깔린 환경에서만 도는 테스트를
기본 pytest에 넣으면, 서비스 환경에서 전체 테스트가 통과하지 못하게 된다.
여기서 지키는 것은 설정/정답/결과 파일의 **모양**이다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cctv_counting.config import CctvConfig
from cctv_counting.detectors.torchvision_detector import (
    DetectorLoadError,
    resolve_device,
)
from cctv_counting.groundtruth import (
    Direction,
    read_crossing_ground_truth,
    read_occupancy_ground_truth,
)
from cctv_counting.metrics import score_occupancy

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
EVAL = ROOT / "evaluation"
EXPERIMENT = ROOT / "experiments" / "a2_local_demo"
VIDEO_ID = "a2-demo-vtest"
WINDOW_END_SECONDS = 29.0


def _load(name: str) -> CctvConfig:
    yaml = pytest.importorskip("yaml")
    return CctvConfig.model_validate(
        yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))
    )


# --- detector 어댑터 경계 -------------------------------------------------


def test_unknown_device_is_refused_before_torch_is_touched():
    """오타 난 device 이름이 torch import까지 가지 않고 걸린다."""
    with pytest.raises(DetectorLoadError) as error:
        resolve_device("gpu")
    assert "gpu" in str(error.value)


def test_device_resolution_needs_torch_and_says_so():
    """torch가 없는 환경에서는 무엇이 없는지 알려주고 끝나야 한다.

    서비스 환경(.venv)에는 torch가 없다. 그때 ImportError가 그대로 올라오면
    설정이 틀린 것인지 설치가 안 된 것인지 구분되지 않는다.
    """
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(DetectorLoadError) as error:
            resolve_device("cpu")
        assert "torch" in str(error.value)
    else:
        assert resolve_device("cpu") == "cpu"


# --- 실험 설정이 비교 가능한가 -------------------------------------------


def test_a2_baselines_differ_only_in_tracking():
    """A와 B의 차이를 추적 유무 탓으로 돌리려면 나머지가 같아야 한다."""
    a = _load("a2_demo_detection")
    b = _load("a2_demo_tracking")
    assert a.detector == b.detector
    assert a.roi == b.roi
    assert a.line == b.line
    assert a.tracker.enabled is False
    assert b.tracker.enabled is True
    assert a.counts_entries is False
    assert b.counts_entries is True


def test_fps_diagnostic_differs_only_in_tracker_ages():
    """진단 실험은 max_age/min_hits 외에는 주 baseline과 같아야 한다."""
    b = _load("a2_demo_tracking")
    fix = _load("a2_demo_tracking_fpsfix")
    assert b.detector == fix.detector
    assert b.roi == fix.roi
    assert b.line == fix.line
    assert (fix.tracker.max_age, fix.tracker.min_hits) != (
        b.tracker.max_age,
        b.tracker.min_hits,
    )
    changed = {
        name
        for name in type(b.tracker).model_fields
        if getattr(b.tracker, name) != getattr(fix.tracker, name)
    }
    assert changed == {"max_age", "min_hits"}


def test_a2_configs_use_the_same_counting_line_as_the_ground_truth():
    """정답은 x=0.72 세로선 기준으로 셌다. 설정이 달라지면 정답이 무효가 된다."""
    for name in ("a2_demo_detection", "a2_demo_tracking", "a2_demo_tracking_fpsfix"):
        config = _load(name)
        assert config.line.start == (0.72, 0.22)
        assert config.line.end == (0.72, 0.78)
        assert config.line.in_direction.value == "POSITIVE"
        assert config.roi.enabled
        assert (config.roi.x1, config.roi.y1, config.roi.x2, config.roi.y2) == (
            0.02,
            0.22,
            0.98,
            0.78,
        )


# --- 수동 정답 -----------------------------------------------------------


def test_occupancy_ground_truth_parses():
    rows = read_occupancy_ground_truth(EVAL / "a2_occupancy_ground_truth.csv")
    assert len(rows) == 30
    assert all(row.video_id == VIDEO_ID for row in rows)
    assert all(row.annotator for row in rows)


def test_occupancy_ground_truth_is_one_sample_per_second():
    """1초 간격 표본이라고 문서에 적었다. 파일도 그래야 한다."""
    rows = read_occupancy_ground_truth(EVAL / "a2_occupancy_ground_truth.csv")
    assert [row.frame_index for row in rows] == [t * 10 for t in range(30)]
    assert [row.timestamp_seconds for row in rows] == [float(t) for t in range(30)]


def test_ground_truth_stays_inside_the_annotated_window():
    """정답이 없는 구간까지 채점하면 그 숫자는 아무것도 재지 않은 것이다."""
    for path, reader in (
        (EVAL / "a2_occupancy_ground_truth.csv", read_occupancy_ground_truth),
        (EVAL / "a2_crossing_ground_truth.csv", read_crossing_ground_truth),
    ):
        for row in reader(path):
            assert row.timestamp_seconds <= WINDOW_END_SECONDS, path.name


def test_crossing_ground_truth_has_both_directions():
    rows = read_crossing_ground_truth(EVAL / "a2_crossing_ground_truth.csv")
    directions = {row.direction for row in rows}
    assert directions == {Direction.IN, Direction.OUT}
    assert sum(1 for r in rows if r.direction is Direction.IN) == 7
    assert sum(1 for r in rows if r.direction is Direction.OUT) == 4


def test_uncertain_crossing_is_marked_as_uncertain():
    """확신이 낮은 사건을 확신하는 것처럼 두면 오차가 거짓이 된다."""
    rows = read_crossing_ground_truth(EVAL / "a2_crossing_ground_truth.csv")
    assert any(row.notes and "불확실" in row.notes for row in rows)


def test_ground_truth_carries_no_identifying_column():
    for name in ("a2_occupancy_ground_truth.csv", "a2_crossing_ground_truth.csv"):
        header = (EVAL / name).read_text(encoding="utf-8").splitlines()[0]
        for token in ("person_id", "track_id", "name", "student"):
            assert token not in header


def test_annotation_rules_are_documented():
    """규칙 문서가 없으면 그 정답은 재현할 수 없다."""
    text = (EVAL / "a2_annotation_rules.md").read_text(encoding="utf-8")
    for required in ("BOTTOM_CENTER", "0.72", "IN", "OUT", "원본 영상"):
        assert required in text


# --- 실험 결과 파일 ------------------------------------------------------


def _metric_files() -> list[Path]:
    return sorted(EXPERIMENT.glob("metrics_*.json"))


def test_experiment_metrics_exist():
    assert _metric_files(), "A2 결과 파일이 없다"


def test_experiment_metrics_have_no_absolute_path():
    """전체 경로에는 사용자 이름과 현장 이름이 섞이곤 한다."""
    for path in [*_metric_files(), *EXPERIMENT.glob("*.json")]:
        text = path.read_text(encoding="utf-8")
        assert "C:\\\\" not in text and "C:\\" not in text, path.name
        assert "Users" not in text, path.name


def test_experiment_metrics_have_the_expected_shape():
    for path in _metric_files():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["windowEndSeconds"] == WINDOW_END_SECONDS
        assert payload["samples"] == 30
        occupancy = payload["occupancy"]
        for key in ("mae", "rmse", "bias"):
            assert isinstance(occupancy[key], (int, float))
        assert occupancy["mae"] >= 0.0
        assert occupancy["rmse"] >= occupancy["mae"] - 1e-9


def test_detection_only_result_reports_no_entry_count():
    """추적을 껐으면 입장 인원은 0이 아니라 없음이어야 한다."""
    payload = json.loads(
        (EXPERIMENT / "metrics_detection.json").read_text(encoding="utf-8")
    )
    assert payload["entry"] is None


def test_tracking_result_reports_an_entry_count():
    payload = json.loads(
        (EXPERIMENT / "metrics_tracking.json").read_text(encoding="utf-8")
    )
    entry = payload["entry"]
    assert entry["actualIn"] == 7
    assert entry["absoluteErrorIn"] == abs(entry["predictedIn"] - entry["actualIn"])


# --- 채점 정렬 -----------------------------------------------------------


def test_scoring_lines_up_ground_truth_with_the_same_frames():
    """프레임을 어긋나게 맞추면 지표가 조용히 틀린다."""
    rows = read_occupancy_ground_truth(EVAL / "a2_occupancy_ground_truth.csv")
    actual = [float(r.actual_count) for r in rows]
    perfect = score_occupancy(actual, actual)
    assert perfect.mae == 0.0
    assert perfect.bias == 0.0

    shifted = actual[1:] + actual[:1]
    assert score_occupancy(actual, shifted).mae > 0.0


def test_reliability_stratification_reports_insufficient_samples():
    """표본이 없는 계층을 0.0으로 적으면 완벽한 성능처럼 보인다."""
    for path in _metric_files():
        strat = json.loads(path.read_text(encoding="utf-8"))["reliabilityStratification"]
        for value in strat.values():
            if value["samples"] == 0:
                assert value["meanAbsError"] is None
