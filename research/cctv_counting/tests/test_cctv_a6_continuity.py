"""Stage A6 — 궤적 연속성 지표, 원인 분류, 그리고 실험 기록의 조건.

지표 쪽에서 지키는 것은 "정의가 결과를 보고 움직이지 않는다"이다. 기록
쪽에서 지키는 것은 "FINAL이 정말 한 번도 손대지 않은 집합이었다"이다.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from cctv_counting.attribution import (
    COVERAGE_FLOOR,
    Attribution,
    FailureCause,
    attribute,
    tally,
)
from cctv_counting.benchmark import GroundTruthBox
from cctv_counting.continuity import (
    ContinuityError,
    TrajectoryTrace,
    associate,
    crossing_window,
    fragments_for,
    gaps_for,
    id_switches_for,
    summarize,
)
from cctv_counting.geometry import BoundingBox

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "a6_continuity"
A5EXP = ROOT / "experiments" / "a5_small_person"
FRAME_WIDTH, FRAME_HEIGHT = 384, 288
FPS = 25.0


def box(x: float, y: float = 100.0, w: float = 40.0, h: float = 120.0) -> BoundingBox:
    return BoundingBox(x, y, x + w, y + h)


def walk(
    trace: TrajectoryTrace,
    frames: range,
    *,
    gt_id: int = 1,
    track_id: int | None = 1,
    detected: bool = True,
    step: float = 4.0,
    start: float = 100.0,
) -> None:
    """한 사람이 곧게 걷는 구간을 기록에 넣는다."""
    for frame in frames:
        truth = [(gt_id, box(start + frame * step))]
        tracked = [(track_id, box(start + frame * step))] if track_id else []
        detections = [box(start + frame * step)] if detected else []
        trace.observe(frame, truth, tracked, detections)


# --- 정답 대응 -------------------------------------------------------------


def test_association_is_one_to_one_and_global():
    """한 예측이 두 정답에 붙으면 연속성이 실제보다 좋아 보인다."""
    truth = [(1, box(0.0)), (2, box(30.0))]
    predicted = [(9, box(15.0))]
    mapping = associate(truth, predicted, iou_threshold=0.1)
    assert len(mapping) == 1
    assert set(mapping.values()) == {9}


def test_association_requires_the_iou_threshold():
    assert associate([(1, box(0.0))], [(9, box(300.0))]) == {}


def test_association_rejects_a_bad_threshold():
    with pytest.raises(ContinuityError):
        associate([(1, box(0.0))], [(9, box(0.0))], iou_threshold=1.5)


# --- fragmentation / id switch --------------------------------------------


def test_a_perfectly_tracked_person_is_one_fragment():
    trace = TrajectoryTrace()
    walk(trace, range(0, 20))
    assert fragments_for(trace, 1) == 1
    assert id_switches_for(trace, 1) == 0


def test_a_broken_track_counts_more_fragments():
    trace = TrajectoryTrace()
    walk(trace, range(0, 10), track_id=1)
    walk(trace, range(10, 14), track_id=None, detected=False)
    walk(trace, range(14, 24), track_id=2)
    assert fragments_for(trace, 1) == 2
    assert id_switches_for(trace, 1) == 1


def test_an_id_change_without_a_gap_still_counts_as_a_fragment():
    trace = TrajectoryTrace()
    walk(trace, range(0, 10), track_id=1)
    walk(trace, range(10, 20), track_id=2)
    assert fragments_for(trace, 1) == 2
    assert id_switches_for(trace, 1) == 1


def test_a_never_tracked_person_has_no_fragments():
    trace = TrajectoryTrace()
    walk(trace, range(0, 10), track_id=None, detected=False)
    assert fragments_for(trace, 1) == 0


# --- gap recovery ----------------------------------------------------------


def test_a_gap_closed_with_the_same_id_is_recovered():
    trace = TrajectoryTrace()
    walk(trace, range(0, 10), track_id=1)
    walk(trace, range(10, 14), track_id=None, detected=False)
    walk(trace, range(14, 24), track_id=1)
    gaps = gaps_for(trace, 1, FPS)
    assert len(gaps) == 1
    assert gaps[0].recovered
    # 덮이지 않은 프레임은 10~13의 4개이고, 끊긴 시간은 마지막으로 덮인
    # 프레임 9에서 다시 덮인 프레임 14까지의 5프레임분이다.
    assert gaps[0].frames == 4
    assert gaps[0].seconds == pytest.approx(5 / FPS)


def test_a_gap_closed_with_a_new_id_is_not_recovered():
    trace = TrajectoryTrace()
    walk(trace, range(0, 10), track_id=1)
    walk(trace, range(10, 14), track_id=None, detected=False)
    walk(trace, range(14, 24), track_id=2)
    assert [gap.recovered for gap in gaps_for(trace, 1, FPS)] == [False]


def test_leading_and_trailing_absence_is_not_a_gap():
    """아직 안 나타났거나 이미 나간 것은 끊긴 것이 아니다."""
    trace = TrajectoryTrace()
    walk(trace, range(0, 5), track_id=None, detected=False)
    walk(trace, range(5, 15), track_id=1)
    walk(trace, range(15, 20), track_id=None, detected=False)
    assert gaps_for(trace, 1, FPS) == []


def test_summary_reports_rates_not_zeros_when_undefined():
    trace = TrajectoryTrace()
    walk(trace, range(0, 10), track_id=1)
    summary = summarize(trace, FPS)
    assert summary.gap_recovery_rate is None  # 끊김이 없으면 정의되지 않는다
    assert summary.fragments_per_gt_track == pytest.approx(1.0)


def test_gaps_reject_a_bad_fps():
    trace = TrajectoryTrace()
    walk(trace, range(0, 4))
    with pytest.raises(ContinuityError):
        gaps_for(trace, 1, 0.0)


# --- 통과 창 연속성 --------------------------------------------------------


def gt_frames_for(frames: range, *, gt_id: int = 1, step: float = 4.0) -> dict:
    return {
        frame: [
            GroundTruthBox(
                track_id=gt_id, bbox=box(100.0 + frame * step), role="walker"
            )
        ]
        for frame in frames
    }


def test_crossing_window_measures_only_the_window():
    trace = TrajectoryTrace()
    walk(trace, range(0, 100))
    window = crossing_window(
        trace,
        gt_frames_for(range(0, 100)),
        gt_track_id=1,
        frame_index=50,
        window_seconds=1.0,
        fps=FPS,
        frame_width=FRAME_WIDTH,
        line_x_normalized=0.5,
        min_confirm_distance=0.03,
    )
    assert window.window_frames == 51  # ±25 프레임
    assert window.detection_coverage == pytest.approx(1.0)
    assert window.track_coverage == pytest.approx(1.0)
    assert window.dominant_track_share == pytest.approx(1.0)
    assert window.distinct_track_ids == 1


def test_crossing_window_sees_a_broken_track_as_low_continuity():
    trace = TrajectoryTrace()
    walk(trace, range(0, 45), track_id=1)
    walk(trace, range(45, 55), track_id=None, detected=False)
    walk(trace, range(55, 100), track_id=2)
    window = crossing_window(
        trace,
        gt_frames_for(range(0, 100)),
        gt_track_id=1,
        frame_index=50,
        window_seconds=1.0,
        fps=FPS,
        frame_width=FRAME_WIDTH,
        line_x_normalized=0.5,
        min_confirm_distance=0.03,
    )
    assert window.distinct_track_ids == 2
    assert window.dominant_track_share < 0.5
    assert window.track_coverage < 1.0


def test_crossing_window_rejects_a_bad_window():
    trace = TrajectoryTrace()
    walk(trace, range(0, 10))
    with pytest.raises(ContinuityError):
        crossing_window(
            trace,
            gt_frames_for(range(0, 10)),
            gt_track_id=1,
            frame_index=5,
            window_seconds=0.0,
            fps=FPS,
            frame_width=FRAME_WIDTH,
            line_x_normalized=0.5,
            min_confirm_distance=0.03,
        )


# --- 원인 분류 -------------------------------------------------------------


def window_stub(**kwargs):
    from cctv_counting.continuity import CrossingWindow

    defaults = dict(
        gt_track_id=1,
        frame_index=50,
        window_frames=51,
        detection_coverage=0.9,
        track_coverage=0.9,
        dominant_track_share=0.9,
        distinct_track_ids=1,
        spans_both_sides=False,
    )
    defaults.update(kwargs)
    return CrossingWindow(**defaults)


def test_no_detection_is_a_detection_gap():
    result = attribute(window_stub(detection_coverage=0.1, track_coverage=0.1), [], max_age_seconds=1.2)
    assert result.cause is FailureCause.DETECTION_GAP


def test_boxes_without_tracks_is_an_association_break():
    result = attribute(
        window_stub(detection_coverage=0.9, track_coverage=0.2), [], max_age_seconds=1.2
    )
    assert result.cause is FailureCause.ASSOCIATION_BREAK


def test_a_long_gap_with_a_new_id_is_track_expiry():
    from cctv_counting.continuity import Gap

    gap = Gap(1, 40, 90, 49, 2.0, before_id=1, after_id=2)
    result = attribute(
        window_stub(distinct_track_ids=2), [gap], max_age_seconds=1.2
    )
    assert result.cause is FailureCause.TRACK_EXPIRY


def test_a_short_gap_with_a_new_id_is_an_id_switch():
    from cctv_counting.continuity import Gap

    gap = Gap(1, 48, 52, 3, 0.16, before_id=1, after_id=2)
    result = attribute(
        window_stub(distinct_track_ids=2), [gap], max_age_seconds=1.2
    )
    assert result.cause is FailureCause.ID_SWITCH


def test_one_unbroken_track_across_the_line_blames_the_line_rule():
    result = attribute(
        window_stub(distinct_track_ids=1, spans_both_sides=True),
        [],
        max_age_seconds=1.2,
    )
    assert result.cause is FailureCause.LINE_GEOMETRY


def test_coverage_floor_is_a_single_fixed_constant():
    """규칙을 사건마다 다르게 적용하지 않는다."""
    assert COVERAGE_FLOOR == pytest.approx(0.5)


def test_tally_reports_every_cause_even_at_zero():
    counts = tally([attribute(window_stub(detection_coverage=0.0), [], max_age_seconds=1.0)])
    assert set(counts) == {cause.value for cause in FailureCause}
    assert counts["DETECTION_GAP"] == 1
    assert counts["ID_SWITCH"] == 0


def test_attribution_records_its_own_evidence():
    result = attribute(window_stub(detection_coverage=0.2), [], max_age_seconds=1.0)
    assert isinstance(result, Attribution)
    payload = result.as_dict()
    assert payload["cause"] == "DETECTION_GAP"
    assert payload["detectionCoverage"] == pytest.approx(0.2)


# --- 실험 기록의 조건 ------------------------------------------------------


def load(name: str) -> dict:
    path = EXP / name
    if not path.exists():
        pytest.skip(f"{path.name}이 아직 없습니다.")
    return json.loads(path.read_text(encoding="utf-8"))


def test_final_split_does_not_overlap_development_or_analysis():
    """FINAL이 정말 처음 보는 시퀀스인가. 이것이 깨지면 숫자가 의미를 잃는다."""
    freeze = load("final_test_freeze.json")
    final = set(freeze["finalSequences"])

    development = {row["sequence"] for row in load("development.json")["rows"]}
    analysis = {row["sequence"] for row in load("analysis_shop.json")["rows"]}
    lobby = {row["sequence"] for row in load("analysis_lobby.json")["rows"]}
    a5_final = {
        row["sequence"]
        for row in json.loads(
            (A5EXP / "final_sequences.json").read_text(encoding="utf-8")
        )["sequences"]
    }

    assert len(final) == 12
    assert final & development == set()
    assert final & analysis == set()
    assert final & lobby == set()
    assert final & a5_final == set()


def test_final_sequences_are_a_different_camera_not_a_second_view_of_seen_events():
    """1cor는 1front와 time synchronised다. 같은 사건을 다시 채점하면 안 된다."""
    freeze = load("final_test_freeze.json")
    for sequence in freeze["finalSequences"]:
        assert sequence.endswith("2cor"), sequence


def test_analysis_roles_are_recorded_as_consumed():
    freeze = load("final_test_freeze.json")
    roles = freeze["dataRoles"]
    assert "A5 FINAL" in roles["ANALYSIS_LOBBY"]
    assert "원인 분석" in roles["ANALYSIS_LOBBY"]
    assert "A4 TEST" in roles["ANALYSIS_SHOP"]


def test_the_freeze_records_a_selection_rule_with_guardrails():
    freeze = load("final_test_freeze.json")
    rule = freeze["selectionRule"]
    assert "guardrail" in rule["occupancy"]
    assert "guardrail" in rule["entry"]
    assert rule["entry"]["primary"].startswith("DEV")
    assert freeze["baseline"] == "P0_baseline"
    assert freeze["selected"] in freeze["policies"]


def test_the_freeze_hash_matches_the_recorded_digest():
    """동결 파일이 FINAL 이후에 바뀌지 않았다는 것을 지문으로 확인한다.

    A5와 같은 규약이다. 지문은 **지문을 뺀 기록**에 대해 계산하고 기록 안에
    넣는다. 옆 파일로 두면 .gitignore가 걸러 커밋되지 않는다.
    """
    record = load("final_test_freeze.json")
    expected = record.pop("sha256OfRecordWithoutHash")
    assert len(expected) == 64
    payload = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == expected


def test_the_detector_was_frozen_and_not_fine_tuned():
    freeze = load("final_test_freeze.json")
    assert freeze["detector"]["fineTuned"] is False
    assert freeze["detector"]["tag"] == "E0_mobilenet_R0"


def test_re_identification_stays_forbidden_in_the_freeze():
    freeze = load("final_test_freeze.json")
    joined = " ".join(freeze["forbidden"])
    assert "re-identification" in joined
    assert "cross-session identity" in joined


def test_event_records_carry_the_evidence_for_their_verdict():
    """사건 기록에 근거가 없으면 나중에 그 판정을 확인할 수 없다."""
    payload = load("development.json")
    required = {
        "frameIndex",
        "direction",
        "gtTrackId",
        "matched",
        "windowFrames",
        "detectionCoverage",
        "trackCoverage",
        "dominantTrackShare",
        "distinctTrackIds",
        "spansBothSides",
    }
    seen_missed = False
    for row in payload["rows"]:
        for entry in row["entryRows"]:
            for event in entry["events"]:
                assert required <= set(event)
                if not event["matched"]:
                    seen_missed = True
                    assert event["cause"] in {c.value for c in FailureCause}
    assert seen_missed


def test_event_records_carry_no_personal_data():
    """기록에 남는 것은 임시 번호와 프레임 번호뿐이다."""
    payload = load("development.json")
    text = json.dumps(payload, ensure_ascii=False).lower()
    for word in ("name", "face", "embedding", "descriptor", "person_id", "identity"):
        assert word not in text


def test_the_line_rule_is_the_same_for_every_sequence():
    """시퀀스마다 선을 옮기면 좋은 숫자는 나오지만 아무것도 재지 않은 것이 된다."""
    freeze = load("final_test_freeze.json")
    geometry = freeze["geometry"]
    assert geometry["lineXNormalized"] == pytest.approx(0.5)
    assert "A4와 완전히 같다" in geometry["note"]


def test_no_raw_dataset_or_weights_are_tracked():
    """corridor 영상과 annotation은 재배포하지 않는다."""
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT.parents[1],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    for path in tracked:
        lowered = path.lower()
        assert not lowered.endswith((".mpg", ".avi", ".mp4", ".pt", ".pth", ".onnx"))
        assert "datasets/raw/" not in lowered
