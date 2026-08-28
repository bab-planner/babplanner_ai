"""Stage A3 — 시간적 확인 baseline, 사건 대조, dev/holdout 분리.

무거운 GPU 추론은 여기서 돌리지 않는다. 지키는 것은 로직과, 실험을 나중에
다시 읽을 때 거짓이 되지 않게 하는 조건들이다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cctv_counting.confirmation import ConfirmationError, TemporalConfirmationCounter
from cctv_counting.config import CctvConfig, CountingMethod
from cctv_counting.groundtruth import read_occupancy_ground_truth
from cctv_counting.metrics import MetricError, TimedEvent, match_events

from conftest import person

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
EXP = ROOT / "experiments" / "a3_ablation"

DEV_END = 29.0
HOLDOUT_START = 30.0


# --- dev / holdout 분리 ---------------------------------------------------


def test_dev_and_holdout_do_not_overlap():
    """겹치면 holdout이 아니다. 가장 먼저 깨지면 안 되는 것."""
    dev = read_occupancy_ground_truth(EVAL / "a2_occupancy_ground_truth.csv")
    holdout = read_occupancy_ground_truth(EVAL / "a3_holdout_occupancy_ground_truth.csv")

    dev_frames = {row.frame_index for row in dev}
    holdout_frames = {row.frame_index for row in holdout}
    assert dev_frames & holdout_frames == set()

    assert max(row.timestamp_seconds for row in dev) <= DEV_END
    assert min(row.timestamp_seconds for row in holdout) >= HOLDOUT_START


def test_holdout_covers_the_rest_of_the_video():
    holdout = read_occupancy_ground_truth(EVAL / "a3_holdout_occupancy_ground_truth.csv")
    assert len(holdout) == 50
    assert [r.timestamp_seconds for r in holdout] == [float(t) for t in range(30, 80)]


def test_holdout_ground_truth_matches_its_sealed_checksum():
    """정답을 예측 본 뒤에 고치면 holdout이 tuning set이 된다.

    지문이 어긋나면 그 사실을 보고서에 적어야 한다. 조용히 넘어가면 안 된다.
    """
    lock = json.loads((EVAL / "a3_holdout_lock.json").read_text(encoding="utf-8"))
    assert lock["sealedBeforeSeeingHoldoutPredictions"] is True
    for name, expected in lock["sha256"].items():
        actual = hashlib.sha256((EVAL / name).read_bytes()).hexdigest()
        assert actual == expected, f"{name} 이 봉인 이후 변경되었다"


# --- 시간적 확인 baseline -------------------------------------------------


def _box(x: float = 100.0, y: float = 400.0):
    return person(x, y - 160, x + 60, y)


def test_confirmation_n1_is_exactly_detection_only():
    """N=1은 확인을 하지 않는 것과 같다. 대조군이 성립하는지 본다."""
    counter = TemporalConfirmationCounter(min_observations=1)
    for _ in range(3):
        assert len(counter.update([_box(), _box(400)])) == 2


def test_confirmation_n3_waits_for_three_observations():
    counter = TemporalConfirmationCounter(min_observations=3)
    assert counter.update([_box()]) == []
    assert counter.update([_box(102)]) == []
    assert len(counter.update([_box(104)])) == 1


def test_confirmation_streak_resets_when_the_box_disappears():
    """한 프레임이라도 끊기면 처음부터 다시 센다."""
    counter = TemporalConfirmationCounter(min_observations=3)
    counter.update([_box()])
    counter.update([_box(102)])
    counter.update([])                      # 끊김
    assert counter.update([_box(104)]) == []
    assert counter.update([_box(106)]) == []
    assert len(counter.update([_box(108)])) == 1


def test_confirmation_streak_resets_when_the_box_jumps_away():
    """겹치지 않으면 이어진 관측이 아니다."""
    counter = TemporalConfirmationCounter(min_observations=2)
    counter.update([_box(100)])
    assert counter.update([_box(600)]) == []


def test_confirmation_keeps_no_long_term_identity():
    """이것은 추적이 아니다. 번호도 궤적도 남기지 않는다.

    직전 한 프레임만 들고 있고, 돌려주는 것도 Detection 그대로다.
    """
    counter = TemporalConfirmationCounter(min_observations=2)
    counter.update([_box(), _box(400)])
    confirmed = counter.update([_box(102), _box(402)])
    assert counter.tracked_state_count == 2, "직전 프레임만 들고 있어야 한다"
    for item in confirmed:
        assert not hasattr(item, "track_id")


def test_confirmation_reset_clears_state():
    counter = TemporalConfirmationCounter(min_observations=2)
    counter.update([_box()])
    counter.reset()
    assert counter.tracked_state_count == 0
    assert counter.update([_box(102)]) == []


def test_confirmation_rejects_bad_settings():
    with pytest.raises(ConfirmationError):
        TemporalConfirmationCounter(min_observations=0)
    with pytest.raises(ConfirmationError):
        TemporalConfirmationCounter(iou_threshold=1.4)


def test_counting_method_is_readable_from_the_config():
    """실험 기록에 "무엇으로 셌는가"가 한 값으로 남아야 한다."""
    off = {"line": {"enabled": False}}
    a = CctvConfig.model_validate(
        {"tracker": {"enabled": False}, "confirmation": {"min_observations": 1}, **off}
    )
    b = CctvConfig.model_validate(
        {"tracker": {"enabled": False}, "confirmation": {"min_observations": 3}, **off}
    )
    c = CctvConfig.model_validate({"tracker": {"enabled": True}})
    assert a.counting_method is CountingMethod.DETECTION_ONLY
    assert b.counting_method is CountingMethod.TEMPORAL_CONFIRMATION
    assert c.counting_method is CountingMethod.TRACKING
    assert not a.counts_entries and not b.counts_entries and c.counts_entries


# --- 사건 대조 -------------------------------------------------------------


def test_event_matching_respects_direction():
    """방향이 다르면 시각이 같아도 짝이 아니다."""
    result = match_events(
        [TimedEvent(10.0, "IN")], [TimedEvent(10.0, "OUT")], tolerance_seconds=1.5
    )
    assert result.matched == []
    assert len(result.false_positives) == 1
    assert len(result.missed) == 1


def test_event_matching_respects_tolerance():
    inside = match_events(
        [TimedEvent(11.4, "IN")], [TimedEvent(10.0, "IN")], tolerance_seconds=1.5
    )
    outside = match_events(
        [TimedEvent(11.6, "IN")], [TimedEvent(10.0, "IN")], tolerance_seconds=1.5
    )
    assert len(inside.matched) == 1
    assert len(outside.matched) == 0


def test_event_matching_pairs_each_event_once():
    """예측 두 개가 정답 하나에 동시에 붙으면 안 된다."""
    result = match_events(
        [TimedEvent(10.0, "IN"), TimedEvent(10.2, "IN")],
        [TimedEvent(10.1, "IN")],
        tolerance_seconds=1.5,
    )
    assert len(result.matched) == 1
    assert len(result.false_positives) == 1
    assert result.missed == []


def test_unmatched_predicted_event_is_a_false_positive():
    result = match_events(
        [TimedEvent(50.0, "IN")], [TimedEvent(10.0, "IN")], tolerance_seconds=1.5
    )
    assert [e.timestamp_seconds for e in result.false_positives] == [50.0]


def test_unmatched_ground_truth_event_is_a_miss():
    result = match_events([], [TimedEvent(10.0, "IN")], tolerance_seconds=1.5)
    assert len(result.missed) == 1
    assert result.recall == 0.0


def test_precision_is_undefined_when_nothing_was_predicted():
    """예측이 0건인데 정밀도 1.0을 주면 아무것도 안 낸 모델이 완벽해 보인다."""
    result = match_events([], [TimedEvent(10.0, "IN")], tolerance_seconds=1.5)
    assert result.precision is None
    assert result.f1 is None


def test_recall_is_undefined_when_there_is_no_ground_truth():
    result = match_events([TimedEvent(10.0, "IN")], [], tolerance_seconds=1.5)
    assert result.recall is None
    assert result.precision == 0.0
    assert result.f1 is None


def test_perfect_match_scores_one():
    events = [TimedEvent(10.0, "IN"), TimedEvent(20.0, "OUT")]
    result = match_events(events, events, tolerance_seconds=1.5)
    assert result.precision == 1.0 and result.recall == 1.0 and result.f1 == 1.0


def test_negative_tolerance_is_rejected():
    with pytest.raises(MetricError):
        match_events([], [], tolerance_seconds=-1.0)


def test_event_matching_is_deterministic():
    predicted = [TimedEvent(10.0, "IN"), TimedEvent(10.5, "IN"), TimedEvent(30.0, "OUT")]
    actual = [TimedEvent(10.2, "IN"), TimedEvent(29.9, "OUT")]
    first = match_events(predicted, actual, 1.5)
    second = match_events(predicted, actual, 1.5)
    assert first == second


# --- 실험 기록 -------------------------------------------------------------


def test_selection_rule_was_fixed_before_the_holdout():
    record = json.loads((EXP / "selection.json").read_text(encoding="utf-8"))
    assert record["ruleFixedBeforeHoldout"] is True
    assert record["rule"], "규칙이 문서로 남아 있어야 한다"
    assert record["selectedForOccupancy"] in {r["config"] for r in _dev_rows()}
    assert record["selectedForEntry"] in {r["config"] for r in _dev_rows()}


def _dev_rows() -> list[dict]:
    import csv

    with (EXP / "dev_metrics.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_selection_follows_the_recorded_rule():
    """기록된 규칙을 dev 표에 다시 적용하면 같은 답이 나와야 한다."""
    record = json.loads((EXP / "selection.json").read_text(encoding="utf-8"))
    rows = [
        {"config": r["config"], "mae": float(r["mae"]), "bias": float(r["bias"])}
        for r in _dev_rows()
    ]
    best = min(rows, key=lambda r: (round(r["mae"], 6), round(abs(r["bias"]), 6)))
    ties = [
        r["config"]
        for r in rows
        if round(r["mae"], 6) == round(best["mae"], 6)
        and round(abs(r["bias"]), 6) == round(abs(best["bias"]), 6)
    ]
    assert record["selectedForOccupancy"] in ties


def test_holdout_metrics_declare_their_window():
    for path in EXP.glob("holdout_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "windowStartSeconds" in payload:
            assert payload["windowStartSeconds"] == HOLDOUT_START
            assert payload["windowEndSeconds"] == 79.0


def test_event_analysis_records_that_tolerance_was_fixed_first():
    payload = json.loads(
        (EXP / "holdout_events_minhits1.json").read_text(encoding="utf-8")
    )
    assert payload["toleranceFixedBeforeSeeingResults"] is True
    assert payload["toleranceSeconds"] == pytest.approx(1.5)


def test_small_event_samples_are_marked_exploratory():
    """표본 7건짜리 F1을 성능처럼 읽으면 안 된다."""
    payload = json.loads(
        (EXP / "holdout_events_minhits1.json").read_text(encoding="utf-8")
    )
    for direction in payload["directions"].values():
        if direction["actual"] < 10:
            assert direction["exploratoryOnlySmallSample"] is True


def test_experiment_records_have_no_absolute_path():
    for path in EXP.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "C:\\" not in text and "Users" not in text, path.name


def test_cuda_metadata_is_json_serializable():
    """장치 정보가 결과 파일에 그대로 실려야 재현 조건을 알 수 있다."""
    payload = json.loads((EXP / "runtime.json").read_text(encoding="utf-8"))
    for key in ("device", "torch", "torchvision", "cudaBuild", "gpuName"):
        assert key in payload, key
    assert json.loads(json.dumps(payload)) == payload
    assert payload["device"] in {"cpu", "cuda"}


def test_stationary_false_positive_study_is_analysis_not_a_filter():
    """정지 사물 분석은 기록일 뿐, 파이프라인에 필터를 넣지 않았다.

    "오래 정지하면 사람이 아니다" 같은 규칙은 실제로 서 있는 사람을 지운다.
    실제로 이 영상에도 여러 초 동안 서 있는 사람이 있었다.
    """
    payload = json.loads((EXP / "h1_stationary_fp.json").read_text(encoding="utf-8"))
    assert "필터를 넣지 않았다" in payload["note"]

    import cctv_counting.confirmation as confirmation
    import cctv_counting.tracking as tracking

    for module in (tracking, confirmation):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for banned in ("stationary", "is_static", "static_filter"):
            assert banned not in source, module.__name__
