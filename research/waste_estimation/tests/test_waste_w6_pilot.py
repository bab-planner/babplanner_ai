"""W6 — 통제된 연출 파일럿의 규약.

여기서 지키는 것은 성능이 아니라 **경계**다.

    연출된 상태를 자연 잔반이라고 부르지 않는다
    사람 정보를 받을 자리를 두지 않는다
    정답은 저울 두 값에서만 나온다
    표본이 모자라면 평가하지 않는다
    학습은 LeFood DEVELOPMENT만
    파일럿을 보고 아무것도 고르지 않는다
    원본 사진은 저장소에 들어가지 않는다
    fixture를 갈아치우지 않는다

이 단계에는 성능 숫자가 없다. 데이터가 0건이기 때문이다. 그래서 여기 있는
테스트는 **데이터가 생겼을 때 무엇이 지켜져야 하는가**를 고정한다.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from waste_estimation.pilot import (
    ALLOWED_SOURCE,
    BIN_EDGES,
    HIGH_WASTE_THRESHOLD,
    MIN_FOOD_GROUPS,
    MIN_HIGH_WASTE,
    MIN_VALID_SAMPLES,
    REQUIRED_COLUMNS,
    WASTE_RATE_TOLERANCE,
    PilotError,
    PilotRejection,
    PilotSample,
    bin_counts,
    evaluate_gate,
    load_manifest,
    repeatability,
    summarize_repeatability,
)
from waste_estimation.schemas import SampleSource, WasteSchemaError

requires_numpy = pytest.mark.skipif(
    importlib.util.find_spec("numpy") is None,
    reason="numpy가 필요합니다 (.venv-waste에서 돕니다)",
)

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports" / "waste_w6_controlled_domain_pilot.md"
RUNBOOK = ROOT / "docs" / "waste_controlled_pilot_runbook.md"
TEMPLATE = (
    ROOT / "docs" / "templates" / "waste_controlled_pilot_manifest_template.csv"
)
SCRIPTS = ROOT / "research" / "waste_estimation" / "scripts"

HEADER = ",".join(
    (
        "sample_id",
        "source",
        "food_group",
        "capture_session_id",
        "image_file",
        "served_edible_mass_g",
        "leftover_edible_mass_g",
        "waste_rate",
        "repeat_index",
    )
)


def text_of(path: Path) -> str:
    """마크다운 인용부호를 떼고 공백을 정규화한다.

    인용 블록에서 줄바꿈된 문장은 각 줄 앞에 `>`가 붙는다. 그대로 이으면
    문장 가운데 `>`가 남아 부분 문자열 검사가 어긋난다.
    """
    if not path.exists():
        pytest.skip(f"{path.name}이 아직 없습니다.")
    lines = [
        line.lstrip("> ").rstrip()
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    return " ".join(" ".join(lines).split())


def write_manifest(tmp_path: Path, rows: list[str], header: str = HEADER) -> Path:
    path = tmp_path / "manifest.csv"
    path.write_text("\n".join([header] + rows) + "\n", encoding="utf-8")
    return path


def row(
    sample_id: str,
    served: float,
    leftover: float,
    group: str = "rice",
    source: str = "controlled_staged_pilot",
    image: str | None = None,
    waste_rate: str = "",
    repeat: int = 0,
) -> str:
    image = image if image is not None else f"{sample_id}.jpg"
    return (
        f"{sample_id},{source},{group},w6_s1,{image},{served},{leftover},"
        f"{waste_rate},{repeat}"
    )


def many(count: int, tmp_path: Path, groups: int = 3, high: int = 6) -> Path:
    rows = []
    for index in range(count):
        group = f"food{index % groups}"
        leftover = 260 if index < high else 20
        rows.append(row(f"pilot_w6_{index:04d}", 300, leftover, group=group))
    return write_manifest(tmp_path, rows)


# --- 연출 표본이라는 사실 -------------------------------------------------------


def test_controlled_staged_is_a_separate_source_from_pilot():
    """`PILOT`과 한 이름으로 두면 언젠가 '실제 식습관'으로 읽힌다."""
    assert SampleSource.CONTROLLED_STAGED.value == "controlled_staged_pilot"
    assert SampleSource.CONTROLLED_STAGED is not SampleSource.PILOT
    assert ALLOWED_SOURCE is SampleSource.CONTROLLED_STAGED


def test_a_natural_pilot_row_is_rejected(tmp_path):
    """출처를 섞어 넣을 수 없다."""
    path = write_manifest(tmp_path, [row("p1", 300, 100, source="pilot")])
    samples, rejections = load_manifest(path)
    assert samples == []
    assert rejections[0]["reason"] == PilotRejection.WRONG_SOURCE.value


def test_report_never_calls_the_samples_natural_meals():
    text = text_of(REPORT)
    assert "실제로 먹고 남긴 사진" not in text.replace(
        '"실제로 먹고 남긴 사진"이라고 쓰지 않는다', ""
    )
    assert "연출된 잔반 상태" in text
    assert "사람 대상 실험이 아니다" in text


def test_runbook_forbids_human_subject_expansion():
    text = text_of(RUNBOOK)
    assert "참가자를 모으지 않는다" in text
    assert "얼굴을 찍지 않는다" in text
    for word in ("학번", "이름", "연락처"):
        assert word in text


def test_report_lists_what_the_pilot_cannot_answer():
    text = text_of(REPORT)
    assert "답할 수 없다" in text
    assert "학생들이 실제로 얼마나 남기는가" in text


# --- 개인정보 ------------------------------------------------------------------


def test_a_personal_column_rejects_the_whole_file(tmp_path):
    """행 하나가 아니라 파일 전체를 거절한다. 열이 있으면 이미 수집한 것이다."""
    header = HEADER + ",student_name"
    path = write_manifest(tmp_path, [row("p1", 300, 100) + ",홍길동"], header=header)
    with pytest.raises(PilotError, match="개인 식별"):
        load_manifest(path)


def test_pilot_sample_has_no_place_for_a_person(tmp_path):
    fields = set(PilotSample.__dataclass_fields__)
    for banned in ("name", "phone", "student", "face", "participant"):
        assert not any(banned in field for field in fields), banned


def test_absolute_image_paths_are_rejected(tmp_path):
    """절대 경로는 개인 폴더 구조를 그대로 노출한다."""
    for bad in ("/home/me/x.jpg", "C:\\Users\\me\\x.jpg"):
        path = write_manifest(tmp_path, [row("p1", 300, 100, image=bad)])
        samples, rejections = load_manifest(path)
        assert samples == []
        assert rejections[0]["reason"] == PilotRejection.ABSOLUTE_PATH.value


def test_template_contains_no_personal_column_and_no_absolute_path():
    with TEMPLATE.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = list(reader)
    for banned in ("name", "phone", "student", "email", "address", "birth"):
        assert not any(banned in column.lower() for column in columns), banned
    for entry in rows:
        assert not Path(entry["image_file"]).is_absolute()
        assert ":" not in entry["image_file"][:3]


# --- 정답 ----------------------------------------------------------------------


def test_ground_truth_comes_from_the_two_weights():
    sample = PilotSample("p1", "rice", "s1", "p1.jpg", 300.0, 75.0)
    assert sample.waste_rate == pytest.approx(0.25)


def test_served_must_be_positive(tmp_path):
    path = write_manifest(tmp_path, [row("p1", 0, 0)])
    samples, rejections = load_manifest(path)
    assert samples == []
    assert rejections[0]["reason"] == PilotRejection.NON_POSITIVE_SERVED.value


def test_leftover_cannot_exceed_served(tmp_path):
    """받은 것보다 많이 남길 수 없다. 식기 무게가 섞였을 때 여기 걸린다."""
    path = write_manifest(tmp_path, [row("p1", 300, 712)])
    samples, rejections = load_manifest(path)
    assert samples == []
    assert rejections[0]["reason"] == PilotRejection.LEFTOVER_EXCEEDS_SERVED.value


def test_negative_leftover_is_rejected(tmp_path):
    path = write_manifest(tmp_path, [row("p1", 300, -5)])
    samples, rejections = load_manifest(path)
    assert rejections[0]["reason"] == PilotRejection.NEGATIVE_LEFTOVER.value


def test_a_mistyped_waste_rate_is_caught(tmp_path):
    """옮겨 적기 실수를 잡는 장치다."""
    path = write_manifest(tmp_path, [row("p1", 300, 75, waste_rate="0.75")])
    samples, rejections = load_manifest(path)
    assert samples == []
    assert rejections[0]["reason"] == PilotRejection.WASTE_RATE_MISMATCH.value


def test_a_rounded_waste_rate_still_passes(tmp_path):
    path = write_manifest(tmp_path, [row("p1", 300, 75, waste_rate="0.2500")])
    samples, rejections = load_manifest(path)
    assert len(samples) == 1 and rejections == []
    assert WASTE_RATE_TOLERANCE > 0


def test_missing_food_group_is_refused(tmp_path):
    path = write_manifest(tmp_path, [row("p1", 300, 100, group="")])
    with pytest.raises(WasteSchemaError, match="food_group"):
        load_manifest(path)


def test_required_columns_must_all_be_present(tmp_path):
    header = ",".join(name for name in REQUIRED_COLUMNS if name != "served_edible_mass_g")
    path = tmp_path / "manifest.csv"
    path.write_text(header + "\n", encoding="utf-8")
    with pytest.raises(PilotError, match="필수 열"):
        load_manifest(path)


# --- 구간과 고잔반 ---------------------------------------------------------------


def test_bins_are_the_ones_earlier_stages_fixed():
    assert BIN_EDGES == (0.2, 0.4, 0.6, 0.8)
    assert HIGH_WASTE_THRESHOLD == 0.8


def test_bin_counts_use_the_left_closed_convention():
    """W5와 같은 규약. 마지막 칸이 고잔반 집합과 정확히 같다."""
    samples = [
        PilotSample(f"p{i}", "rice", "s1", f"p{i}.jpg", 100.0, leftover)
        for i, leftover in enumerate((0, 20, 45, 65, 80, 100))
    ]
    assert bin_counts(samples) == [1, 1, 1, 1, 2, ][:5]


def test_high_waste_counting_matches_the_threshold():
    samples = [
        PilotSample(f"p{i}", "rice", "s1", f"p{i}.jpg", 100.0, leftover)
        for i, leftover in enumerate((79, 80, 81))
    ]
    high = [s for s in samples if s.waste_rate >= HIGH_WASTE_THRESHOLD]
    assert len(high) == 2


# --- 중복 -----------------------------------------------------------------------


def test_the_same_sample_id_twice_is_a_duplicate(tmp_path):
    path = write_manifest(tmp_path, [row("p1", 300, 100), row("p1", 300, 120)])
    samples, rejections = load_manifest(path)
    assert len(samples) == 1
    assert rejections[0]["reason"] == PilotRejection.DUPLICATE_SAMPLE_ID.value


def test_repeat_rows_share_a_sample_id_on_purpose(tmp_path):
    """반복 계량 절차가 중복으로 거절되면 §5가 통째로 막힌다."""
    path = write_manifest(
        tmp_path,
        [
            row("p1", 300, 100, repeat=0),
            row("p1", 300, 102, repeat=1),
            row("p1", 300, 99, repeat=2),
        ],
    )
    samples, rejections = load_manifest(path)
    assert len(samples) == 3
    assert rejections == []
    assert [s.repeat_index for s in samples] == [0, 1, 2]


def test_a_duplicate_image_is_rejected(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"same-bytes")
    (images / "b.jpg").write_bytes(b"same-bytes")
    path = write_manifest(
        tmp_path, [row("p1", 300, 100, image="a.jpg"), row("p2", 300, 120, image="b.jpg")]
    )
    samples, rejections = load_manifest(path, images)
    assert len(samples) == 1
    assert rejections[0]["reason"] == PilotRejection.DUPLICATE_IMAGE.value


def test_repeat_rows_may_reuse_the_same_image(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "p1.jpg").write_bytes(b"one-plate")
    path = write_manifest(
        tmp_path, [row("p1", 300, 100, repeat=0), row("p1", 300, 103, repeat=1)]
    )
    samples, rejections = load_manifest(path, images)
    assert len(samples) == 2 and rejections == []


def test_a_missing_image_is_rejected(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    path = write_manifest(tmp_path, [row("p1", 300, 100)])
    samples, rejections = load_manifest(path, images)
    assert samples == []
    assert rejections[0]["reason"] == PilotRejection.MISSING_IMAGE.value


# --- 수용 관문 -------------------------------------------------------------------


def test_gate_thresholds_are_the_documented_ones():
    assert (MIN_VALID_SAMPLES, MIN_FOOD_GROUPS, MIN_HIGH_WASTE) == (30, 3, 6)
    text = text_of(REPORT)
    assert "≥ 30" in text and "≥ 3" in text and "≥ 6" in text


def test_a_full_pilot_passes_the_gate(tmp_path):
    path = many(34, tmp_path)
    samples, rejections = load_manifest(path)
    gate = evaluate_gate(samples, rejections)
    assert gate["passed"] is True
    assert gate["status"] == "PILOT_READY"


def test_too_few_samples_blocks_evaluation(tmp_path):
    path = many(12, tmp_path)
    samples, rejections = load_manifest(path)
    gate = evaluate_gate(samples, rejections)
    assert gate["passed"] is False
    assert gate["status"] == "PILOT_INSUFFICIENT"
    assert gate["checks"]["validSamples"]["passed"] is False


def test_too_few_food_groups_blocks_evaluation(tmp_path):
    path = many(34, tmp_path, groups=2)
    samples, rejections = load_manifest(path)
    gate = evaluate_gate(samples, rejections)
    assert gate["passed"] is False
    assert gate["checks"]["foodGroups"]["passed"] is False


def test_too_few_high_waste_samples_blocks_evaluation(tmp_path):
    path = many(34, tmp_path, high=2)
    samples, rejections = load_manifest(path)
    gate = evaluate_gate(samples, rejections)
    assert gate["passed"] is False
    assert gate["checks"]["highWasteSamples"]["passed"] is False


def test_any_invalid_row_blocks_evaluation(tmp_path):
    path = many(34, tmp_path)
    rows = path.read_text(encoding="utf-8").rstrip().split("\n")
    rows.append(row("pilot_w6_9999", 300, 999))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    samples, rejections = load_manifest(path)
    gate = evaluate_gate(samples, rejections)
    assert gate["passed"] is False
    assert gate["checks"]["invalidAfterExclusion"]["passed"] is False


def test_the_gate_explains_why_it_exists():
    from waste_estimation.pilot import evaluate_gate as function

    assert "성능으로 인용된다" in (function.__doc__ or "")


def test_the_audit_cli_reports_zero_when_there_is_no_data():
    """데이터가 없다는 것을 0으로 말한다. 추정하지 않는다."""
    source = (SCRIPTS / "w6_pilot_audit.py").read_text(encoding="utf-8")
    assert "BLOCKED_PENDING_CONTROLLED_DATA" in source
    assert '"validSamples": 0' in source


def test_the_evaluate_cli_stops_when_the_gate_fails():
    source = (SCRIPTS / "w6_evaluate.py").read_text(encoding="utf-8")
    assert "if not gate[\"passed\"]:" in source
    assert "모델을 평가하지 않습니다" in source


# --- 반복 계량 -------------------------------------------------------------------


def test_repeatability_reports_grams_and_waste_rate():
    """5 g의 흔들림은 300 g에서 0.017이지만 60 g에서는 0.083이다."""
    big = repeatability([100.0, 105.0, 95.0], 300.0)
    small = repeatability([100.0, 105.0, 95.0], 60.0)
    assert big["stdG"] == pytest.approx(small["stdG"])
    assert small["stdWasteRate"] > big["stdWasteRate"]
    assert big["rangeG"] == pytest.approx(10.0)


def test_repeatability_needs_at_least_two_measurements():
    with pytest.raises(PilotError):
        repeatability([100.0], 300.0)


def test_repeatability_refuses_a_non_positive_denominator():
    with pytest.raises(PilotError):
        repeatability([100.0, 101.0], 0.0)


def test_measurement_floor_is_summarised_for_reading_next_to_mae():
    summary = summarize_repeatability(
        [repeatability([100.0, 102.0, 98.0], 300.0), repeatability([50.0, 51.0], 200.0)]
    )
    assert summary["samplesRepeated"] == 2
    assert "측정 잡음과 구분되지 않는다" in summary["interpretation"]


def test_report_requires_the_floor_next_to_the_model_error():
    text = text_of(REPORT)
    assert "측정 반복성" in text
    assert "숨김" in text


# --- 동결된 평가 -----------------------------------------------------------------


def test_training_uses_lefood_development_only():
    source = (SCRIPTS / "w6_evaluate.py").read_text(encoding="utf-8")
    assert "LeFood DEVELOPMENT" in source
    assert "파일럿 라벨은 학습에" in source
    assert '"trainingSource": "LeFood DEVELOPMENT only"' in source


def test_pilot_labels_are_never_fitted():
    source = (SCRIPTS / "w6_evaluate.py").read_text(encoding="utf-8")
    assert '"pilotUsedForFitting": False' in source
    assert '"pilotUsedForSelection": False' in source
    assert '"pilotUsedForThresholds": False' in source


def test_global_mean_comes_from_lefood_not_from_the_pilot():
    """파일럿 평균을 쓰면 baseline이 정답 분포를 미리 본 것이 된다."""
    source = (SCRIPTS / "w6_evaluate.py").read_text(encoding="utf-8")
    assert "dev_targets.mean()" in source
    assert "파일럿 평균을 쓰면 누수다" in source
    assert '"globalMeanSource": "LeFood DEVELOPMENT mean"' in source


def test_the_frozen_structure_is_read_not_chosen():
    source = (SCRIPTS / "w6_evaluate.py").read_text(encoding="utf-8")
    assert 'selected = development["selected"]' in source
    assert "여기서 바꾸지 않는다" in source
    assert "argmin" not in source


def test_the_comparators_include_a_global_mean_and_the_two_models():
    source = (SCRIPTS / "w6_evaluate.py").read_text(encoding="utf-8")
    for name in ("B0_GLOBAL_MEAN", "S0_STANDARD_RIDGE"):
        assert name in source


def test_reproducibility_hashes_are_recorded():
    source = (SCRIPTS / "w6_evaluate.py").read_text(encoding="utf-8")
    for key in (
        "w5PolicyHash",
        "w5SelectedConfigHash",
        "lefoodSplitManifestHash",
        "pilotManifestHash",
    ):
        assert key in source, key


def test_the_domain_gap_is_labelled_descriptive_only():
    source = (SCRIPTS / "w6_evaluate.py").read_text(encoding="utf-8")
    assert "같은 모집단이 아니므로 동등 비교가 아니다" in source
    assert "통계적 동등 비교라고 말하지 않는다" in text_of(REPORT)


def test_selective_prediction_is_not_revived_as_a_safety_mechanism():
    text = text_of(REPORT)
    assert "안전장치로 되살리지 않고" in text


# --- 보고와 상태 ------------------------------------------------------------------


def test_report_status_is_blocked_while_there_is_no_data():
    text = text_of(REPORT)
    assert "BLOCKED_PENDING_CONTROLLED_DATA" in text
    assert "이 문서에는 성능 숫자가 없다" in text


def test_report_records_all_three_hypotheses_as_pending():
    text = text_of(REPORT)
    for name in ("H-W6-A", "H-W6-B", "H-W6-C"):
        assert name in text
    assert "판정 보류" in text


def test_report_keeps_the_minimum_safety_condition_for_the_demo():
    text = text_of(REPORT)
    assert "확정값처럼 보여 주지 않는다" in text


def test_report_admits_the_fixed_capture_makes_results_optimistic():
    text = text_of(REPORT)
    assert "실사용보다 낙관적으로 나올 것이다" in text


def test_no_synthetic_images_were_generated():
    text = text_of(REPORT)
    assert "합성 이미지를 만들지 않았다" in text
    images = ROOT / "research" / "waste_estimation" / "data" / "private"
    assert not images.exists() or not list(images.rglob("*.jpg"))


def test_runbook_is_distinct_from_the_human_participant_one():
    text = text_of(RUNBOOK)
    assert "waste_pilot_collection_runbook.md" in text
    assert "다른 문서다" in text


def test_template_rows_are_marked_as_examples():
    content = TEMPLATE.read_text(encoding="utf-8")
    assert "예시 행입니다" in content
    assert "controlled_staged_pilot" in content


# --- production 경계 ---------------------------------------------------------------


def test_ai_service_has_no_runtime_waste_analysis():
    """최종 AI 범위는 CCTV 혼잡도와 식수 인원 예측 둘뿐이다.

    잔반 분석은 AI 서비스의 운영 경로가 아니다(잔반 인증은 Backend의 관리자 승인
    기능). 연구 결과가 운영 경로로 들어오지 않았음을 이 경계로 확인한다.
    """
    assert not (ROOT / "src" / "babplanner_ai" / "waste").exists()
    main = (ROOT / "src" / "babplanner_ai" / "main.py").read_text(encoding="utf-8")
    assert "waste" not in main


def test_no_pilot_code_is_wired_into_the_service():
    service = ROOT / "src" / "babplanner_ai"
    hits = [
        path
        for path in service.rglob("*.py")
        if "waste_estimation.pilot" in path.read_text(encoding="utf-8")
    ]
    assert hits == []


def test_report_keeps_the_production_boundary():
    text = text_of(REPORT)
    assert "WASTE_MODEL_MODE=fixture" in text
    assert "FastAPI wiring 없음" in text


def test_private_pilot_data_is_ignored_by_git():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "research/waste_estimation/data/" in ignore
    assert "data/private/" in ignore


def test_no_pilot_image_is_tracked():
    base = ROOT / "research" / "waste_estimation"
    tracked = [
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png")
        for path in base.rglob(pattern)
        if "data" not in path.parts
    ]
    assert tracked == []


def test_no_weights_are_tracked():
    base = ROOT / "research" / "waste_estimation" / "experiments"
    heavy = [
        path
        for pattern in ("*.pt", "*.pth", "*.npy", "*.npz")
        for path in base.rglob(pattern)
    ]
    assert heavy == []
