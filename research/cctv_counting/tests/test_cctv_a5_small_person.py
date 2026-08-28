"""Stage A5 — 사람 크기별 검출, 해상도 profile, fresh test 분리.

무거운 모델 추론은 여기서 돌리지 않는다. 지키는 것은 정의와 분할, 그리고
실험 기록이 나중에 거짓이 되지 않게 하는 조건들이다.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from cctv_counting.geometry import BoundingBox
from cctv_counting.sizeeval import (
    MATCH_IOU,
    MEDIUM_MAX,
    SMALL_MAX,
    MatchSummary,
    SizeBin,
    SizeEvalError,
    accumulate,
    match_frame,
    normalized_area,
    normalized_height,
    size_bin,
)

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "a5_small_person"
A4EXP = ROOT / "experiments" / "a4_cross_video"


def box(x: float, y: float, w: float, h: float) -> BoundingBox:
    return BoundingBox(x, y, x + w, y + h)


# --- 크기 정의 -------------------------------------------------------------


def test_normalized_height_is_relative_to_image_height():
    assert normalized_height(box(0, 0, 20, 72), 288) == pytest.approx(0.25)
    assert normalized_height(box(0, 0, 20, 28.8), 288) == pytest.approx(0.10)


def test_normalized_area_is_relative_to_image_area():
    assert normalized_area(box(0, 0, 38.4, 28.8), 384, 288) == pytest.approx(0.01)


def test_size_helpers_reject_bad_image_size():
    with pytest.raises(SizeEvalError):
        normalized_height(box(0, 0, 10, 10), 0)
    with pytest.raises(SizeEvalError):
        normalized_area(box(0, 0, 10, 10), 0, 288)


def test_size_bin_boundaries_are_fixed_and_upper_inclusive():
    """경계값은 위쪽 구간에 넣는다. 결과를 보고 옮기지 않는다."""
    assert SMALL_MAX == pytest.approx(0.10)
    assert MEDIUM_MAX == pytest.approx(0.20)
    assert size_bin(0.0999) is SizeBin.SMALL
    assert size_bin(0.10) is SizeBin.MEDIUM
    assert size_bin(0.1999) is SizeBin.MEDIUM
    assert size_bin(0.20) is SizeBin.LARGE
    assert size_bin(0.95) is SizeBin.LARGE


def test_size_bin_rejects_negative():
    with pytest.raises(SizeEvalError):
        size_bin(-0.01)


# --- 매칭 -----------------------------------------------------------------


def test_matching_is_one_to_one():
    """한 예측이 두 정답에 붙으면 recall과 precision이 동시에 부풀려진다."""
    truth = [box(0, 0, 40, 100), box(4, 4, 40, 100)]
    predicted = [box(2, 2, 40, 100)]
    matches, unmatched_p, unmatched_t = match_frame(predicted, truth)
    assert len(matches) == 1
    assert unmatched_p == []
    assert len(unmatched_t) == 1


def test_duplicate_predictions_do_not_both_match():
    truth = [box(0, 0, 40, 100)]
    predicted = [box(0, 0, 40, 100), box(1, 1, 40, 100)]
    matches, unmatched_p, _ = match_frame(predicted, truth)
    assert len(matches) == 1
    assert len(unmatched_p) == 1, "두 번째 예측은 오탐이어야 한다"


def test_matching_requires_the_iou_threshold():
    truth = [box(0, 0, 40, 100)]
    far = [box(200, 200, 40, 100)]
    matches, unmatched_p, unmatched_t = match_frame(far, truth)
    assert matches == []
    assert len(unmatched_p) == 1 and len(unmatched_t) == 1


def test_matching_is_deterministic():
    truth = [box(0, 0, 40, 100), box(100, 0, 40, 100)]
    predicted = [box(1, 1, 40, 100), box(101, 1, 40, 100)]
    assert match_frame(predicted, truth) == match_frame(predicted, truth)


def test_match_iou_threshold_is_fixed():
    assert MATCH_IOU == pytest.approx(0.5)


def test_matching_rejects_bad_threshold():
    with pytest.raises(SizeEvalError):
        match_frame([], [], iou_threshold=0.0)


# --- 집계 -----------------------------------------------------------------


def test_recall_by_size_is_accumulated_per_bin():
    summary = MatchSummary()
    small = box(0, 0, 8, 20)      # h/H = 20/288 = 0.069 -> SMALL
    large = box(100, 0, 30, 80)   # 80/288 = 0.278 -> LARGE
    accumulate(summary, [large], [small, large], image_width=384, image_height=288)
    assert summary.by_size[SizeBin.SMALL].total == 1
    assert summary.by_size[SizeBin.SMALL].matched == 0
    assert summary.by_size[SizeBin.LARGE].matched == 1
    assert summary.recall == pytest.approx(0.5)


def test_empty_size_bin_reports_none_not_zero():
    """표본 0인 구간에 recall 0.0을 주면 "전부 놓쳤다"로 읽힌다."""
    summary = MatchSummary()
    accumulate(summary, [], [], image_width=384, image_height=288)
    for b in SizeBin:
        assert summary.by_size[b].recall is None
    assert summary.recall is None
    assert summary.precision is None


def test_false_positives_per_frame():
    summary = MatchSummary()
    for _ in range(4):
        accumulate(summary, [box(300, 200, 20, 50)], [], image_width=384, image_height=288)
    assert summary.frames == 4
    assert summary.false_positives == 4
    assert summary.false_positives_per_frame == pytest.approx(1.0)
    assert summary.precision == pytest.approx(0.0)


def test_missed_and_matched_heights_are_recorded():
    """놓친 상자의 크기 분포를 봐야 H4를 판정할 수 있다."""
    summary = MatchSummary()
    small = box(0, 0, 8, 20)
    accumulate(summary, [], [small], image_width=384, image_height=288)
    assert summary.missed_heights and summary.matched_heights == []
    assert summary.missed_heights[0] == pytest.approx(20 / 288)


# --- detector transform ----------------------------------------------------


def test_min_size_without_max_size_is_refused():
    """min_size만 올리면 긴 변이 max_size에 걸려 의도한 해상도가 안 나온다.

    A5에서 실제로 확인했다. min_size=1200 / max_size=1333이면 짧은 변이
    1200이 아니라 999가 된다.
    """
    from cctv_counting.detectors.torchvision_detector import (
        DetectorLoadError,
        TorchvisionPersonDetector,
    )

    with pytest.raises((DetectorLoadError, RuntimeError)) as error:
        TorchvisionPersonDetector(device="cpu", min_size=1200)
    assert "max_size" in str(error.value)


def test_resolution_profiles_are_recorded_in_the_freeze():
    freeze = json.loads((EXP / "final_test_freeze.json").read_text(encoding="utf-8"))
    dev = json.loads((EXP / "development.json").read_text(encoding="utf-8"))
    shapes = {r["candidate"]: tuple(r["transformedShape"]) for r in dev["rows"]}
    # 기본 profile은 이미 확대되어 들어간다. 이것을 모르면 외부 resize가
    # 효과 있다고 착각한다.
    assert shapes["E0_mobilenet_R0"] == (800, 1066)
    assert shapes["E1_mobilenet_R1"] == (1200, 1600)
    assert shapes["E2_mobilenet_R2"] == (1600, 2133)
    assert freeze["frozenConfig"]["threshold"] == pytest.approx(0.7)


def test_candidate_metadata_is_complete():
    dev = json.loads((EXP / "development.json").read_text(encoding="utf-8"))
    for row in dev["rows"]:
        for key in ("model", "minSize", "maxSize", "transformedShape",
                    "detectFps", "peakVramMb", "threshold"):
            assert key in row, key
        assert row["minSize"] <= row["maxSize"]


def test_at_most_two_detector_families_were_used():
    """모델을 무작정 탐색하지 않는다."""
    dev = json.loads((EXP / "development.json").read_text(encoding="utf-8"))
    assert len({r["model"] for r in dev["rows"]}) <= 2


# --- 분할 -----------------------------------------------------------------


def _final_sequences() -> set[str]:
    listing = json.loads((EXP / "final_sequences.json").read_text(encoding="utf-8"))
    return {row["sequence"] for row in listing["sequences"]}


def test_final_test_sequences_do_not_overlap_development_or_analysis():
    """A4에서 이미 본 시퀀스를 final test라고 부르면 안 된다."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from a4_run import SEQUENCES

    seen = {s for s, _, _ in SEQUENCES}
    assert _final_sequences() & seen == set()
    assert len(_final_sequences()) >= 3


def test_final_test_comes_from_a_different_site():
    listing = json.loads((EXP / "final_sequences.json").read_text(encoding="utf-8"))
    assert "CAVIARDATA1" in listing["source"]
    assert listing["license"] == "Creative Commons BY-SA"
    assert listing["acknowledgement"]


def test_a4_test_set_is_reclassified_as_analysis():
    """결과를 이미 본 집합을 "손대지 않은 test"라고 부르지 않는다."""
    text = (ROOT.parents[1] / "reports" / "cctv_a4_cross_video.md").read_text(encoding="utf-8")
    assert "ANALYSIS SET" in text
    assert "손대지 않은 test set" in text


def test_a4_equality_wording_was_corrected():
    """구조적 항등을 관찰 증거처럼 적어 두면 안 된다."""
    text = (ROOT.parents[1] / "reports" / "cctv_a4_cross_video.md").read_text(encoding="utf-8")
    assert "구조적" in text and "항등" in text
    assert "time_since_update" in text


# --- 동결과 선택 -----------------------------------------------------------


def test_final_freeze_was_written_before_the_final_run():
    freeze = json.loads((EXP / "final_test_freeze.json").read_text(encoding="utf-8"))
    assert freeze["frozenBeforeFinalTest"] is True
    assert freeze["selectionRules"]
    assert "sha256OfRecordWithoutHash" in freeze
    assert len(freeze["sha256OfRecordWithoutHash"]) == 64
    assert freeze["noFineTuning"] is True


def test_selection_followed_the_recorded_rule():
    """기록된 규칙을 DEV 표에 다시 적용하면 같은 답이 나와야 한다."""
    freeze = json.loads((EXP / "final_test_freeze.json").read_text(encoding="utf-8"))
    agg = json.loads((EXP / "development_aggregate.json").read_text(encoding="utf-8"))
    margin = freeze["maeMarginPersons"]
    best = min(v["macroMae"] for v in agg.values())
    tied = {k for k, v in agg.items() if v["macroMae"] - best <= margin}
    assert set(freeze["maeTiedWithinMargin"]) == tied
    expected = sorted(
        tied, key=lambda k: (-(agg[k]["small"] or 0.0), -agg[k]["fps"])
    )[0]
    assert freeze["selectedCandidate"] == expected


def test_final_test_did_not_select_the_candidate():
    """FINAL 결과로 후보를 바꾸면 FINAL이 dev가 된다."""
    freeze = json.loads((EXP / "final_test_freeze.json").read_text(encoding="utf-8"))
    assert freeze["selectedCandidate"] in freeze["candidatesConsidered"]
    contrast = freeze["alsoEvaluatedOnFinalForContrast"]
    assert contrast != freeze["selectedCandidate"]
    assert "선택에 쓰지 않았다" in freeze["contrastNote"]


def test_margin_was_justified_before_results():
    freeze = json.loads((EXP / "final_test_freeze.json").read_text(encoding="utf-8"))
    assert freeze["maeMarginPersons"] == pytest.approx(0.10)
    assert freeze["marginNote"]


# --- 결과 기록 -------------------------------------------------------------


def test_experiment_records_have_no_absolute_path():
    for path in EXP.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "C:\\" not in text and "Users" not in text, path.name


def test_vram_metadata_is_json_safe():
    """GPU 메모리 수치가 직렬화되어 재현 조건에 남아야 한다."""
    dev = json.loads((EXP / "development.json").read_text(encoding="utf-8"))
    for row in dev["rows"]:
        peak = row["peakVramMb"]
        assert peak is None or isinstance(peak, (int, float))
        assert json.loads(json.dumps(peak)) == peak


def test_no_raw_dataset_or_weights_are_tracked():
    gitignore = (ROOT.parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "datasets/raw/" in gitignore
    assert "*.pth" in gitignore


def test_charts_exist_and_are_plain_svg():
    """원본 프레임이나 사람 이미지를 내보내지 않는다. 집계 수치만."""
    assets = ROOT.parents[1] / "reports" / "assets"
    charts = sorted(assets.glob("*.svg"))
    assert charts, "그림이 없다"
    for chart in charts:
        text = chart.read_text(encoding="utf-8")
        assert text.lstrip().startswith("<svg")
        assert "<image" not in text, f"{chart.name}: 래스터 이미지가 들어 있다"
        assert "base64" not in text


# --- 통과 사건 크기 분석 (H4) ---------------------------------------------


def _crossing_module():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import a5_crossing_size

    return a5_crossing_size


def test_crossing_events_match_one_to_one_within_tolerance():
    """한 예측이 두 정답 사건을 동시에 맞혔다고 세면 recall이 부풀려진다."""
    m = _crossing_module()

    class Event:
        def __init__(self, frame, direction):
            self.frame_index = frame
            self.direction = type("D", (), {"value": direction})()

    truth = [Event(100, "IN"), Event(105, "IN")]  # 4초, 4.2초
    predicted = [(4.1, "IN", 0.2)]
    hit, used = m.match_by_time(predicted, truth, 1.5)
    assert len(hit) == 1 and len(used) == 1


def test_crossing_match_respects_direction():
    m = _crossing_module()

    class Event:
        def __init__(self, frame, direction):
            self.frame_index = frame
            self.direction = type("D", (), {"value": direction})()

    hit, _ = m.match_by_time([(4.0, "OUT", 0.2)], [Event(100, "IN")], 1.5)
    assert hit == set(), "방향이 다르면 같은 사건이 아니다"


def test_crossing_match_respects_tolerance():
    m = _crossing_module()

    class Event:
        def __init__(self, frame, direction):
            self.frame_index = frame
            self.direction = type("D", (), {"value": direction})()

    hit, _ = m.match_by_time([(9.0, "IN", 0.2)], [Event(100, "IN")], 1.5)
    assert hit == set()


def test_crossing_ground_truth_records_person_size():
    """H4를 판정하려면 사건마다 그 사람의 크기를 알아야 한다."""
    from cctv_counting.benchmark import GroundTruthBox, crossing_ground_truth

    frames = {
        i: [GroundTruthBox(1, BoundingBox(x, 100, x + 20, 160), "walker")]
        for i, x in enumerate(range(300, 40, -20))
    }
    events = crossing_ground_truth(
        frames, frame_width=384, frame_height=288,
        line_x_normalized=0.5, min_confirm_distance=0.03,
    )
    assert len(events) == 1
    assert events[0].direction.value == "IN"
    assert events[0].normalized_height == pytest.approx(60 / 288)


def test_h4_chain_records_all_three_steps():
    """세 단계 중 하나라도 빠지면 H4를 지지한다고 말할 수 없다."""
    path = EXP / "crossing_size.json"
    if not path.exists():
        pytest.skip("crossing_size.json 없음")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["baseline"] != payload["contrast"]
    for seq in payload["sequences"]:
        chain = seq["chain"]
        assert chain is not None
        for row in chain["rows"]:
            # 1단계: 사건 크기, 2단계: 검출 회복, 3단계: 사건 회복
            assert "normalizedHeight" in row
            assert "baseTrackRecall" in row and "contrastTrackRecall" in row
            assert isinstance(row["eventRecovered"], bool)


def test_recovery_rate_is_bounded_by_what_was_missed():
    """회복 수가 놓친 수를 넘으면 계산이 틀린 것이다."""
    path = EXP / "crossing_size.json"
    if not path.exists():
        pytest.skip("crossing_size.json 없음")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for seq in payload["sequences"]:
        chain = seq["chain"]
        assert 0 <= chain["eventsRecoveredByContrast"] <= chain["missedByBaseline"]
        assert len(chain["rows"]) == chain["missedByBaseline"]


# --- 구조적 오탐 (H1) ------------------------------------------------------


def test_structural_fp_split_is_exhaustive():
    """구조적 + 산발적 = 전체. 어느 쪽에도 안 들어간 오탐이 있으면 안 된다."""
    for name in ("structural_fp_dev.json", "structural_fp_final.json"):
        path = EXP / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for tag, r in payload["candidates"].items():
            assert (
                r["persistentFalsePositives"] + r["transientFalsePositives"]
                == r["falsePositives"]
            ), f"{name}/{tag}"


def test_structural_fp_grid_was_fixed_in_advance():
    path = EXP / "structural_fp_dev.json"
    if not path.exists():
        pytest.skip("structural_fp_dev.json 없음")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["grid"] == [16, 12]
    assert payload["persistentRatio"] == pytest.approx(0.20)


# --- 높이 히스토그램 -------------------------------------------------------


def test_histogram_median_matches_plain_median():
    from cctv_counting.sizeeval import height_histogram, histogram_median

    for values in ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4], [0.05] * 7 + [0.9]):
        expected = statistics.median(values)
        assert histogram_median(height_histogram(values)) == pytest.approx(
            expected, abs=0.001
        )


def test_histogram_of_empty_is_none_not_zero():
    from cctv_counting.sizeeval import (
        histogram_fraction_below,
        histogram_median,
        histogram_min,
    )

    assert histogram_median({}) is None
    assert histogram_min({}) is None
    assert histogram_fraction_below({}, 0.10) is None


def test_histogram_preserves_count():
    """접는 과정에서 상자가 사라지면 recall 분모가 틀어진다."""
    from cctv_counting.sizeeval import height_histogram

    values = [0.0512, 0.0514, 0.1319, 0.2, 0.2]
    assert sum(height_histogram(values).values()) == len(values)


def test_result_files_store_histograms_not_raw_lists():
    """상자별 높이를 그대로 두면 실험 기록이 수백 KB가 된다."""
    for name in ("development.json", "final_test.json"):
        path = EXP / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            assert "missedHeights" not in row["detection"]
            assert isinstance(row["detection"]["missedHeightHistogram"], dict)
        assert path.stat().st_size < 600_000, f"{name}이 너무 크다"
