"""W1 — 문제 정의가 문서와 코드에서 어긋나지 않는지.

여기서 검사하는 것은 모델이 아니라 **경계**다. 서비스 계약, 정답의 단위,
세 결과(APPROVED / REJECTED / FAILED)의 뜻, 공개 데이터의 허용 범위.

이런 것들은 시간이 지나면 조용히 어긋난다. 한쪽 문서만 고치고 다른 쪽을
잊는 식으로. 그때 어긋난 채로 학습이 시작되면 무엇을 배운 모델인지 알 수
없게 된다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from waste_estimation.public_datasets import (
    CANDIDATES,
    DatasetRole,
    LicenseStatus,
    PublicDataset,
    regression_ground_truth_candidates,
    usable_candidates,
)
from waste_estimation.schemas import LABEL_VERSION

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports" / "waste_w1_problem_definition.md"
AUDIT = ROOT / "research" / "waste_estimation" / "dataset_audit.md"
DATASETS = ROOT / "research" / "waste_estimation" / "datasets" / "README.md"


# --- 서비스 경계 -----------------------------------------------------------


def test_ai_service_has_no_runtime_waste_analysis():
    """최종 AI 범위는 CCTV 혼잡도와 식수 인원 예측 둘뿐이다.

    잔반 분석은 AI 서비스의 운영 경로가 아니다(잔반 인증은 Backend의 관리자 승인
    기능). 연구 결과가 운영 경로로 들어오지 않았음을 이 경계로 확인한다.
    """
    assert not (ROOT / "src" / "babplanner_ai" / "waste").exists()
    main = (ROOT / "src" / "babplanner_ai" / "main.py").read_text(encoding="utf-8")
    assert "waste" not in main


# --- 정답 정의가 문서와 같은가 ---------------------------------------------


def test_label_version_matches_the_dataset_spec():
    assert LABEL_VERSION == "waste-mass-ratio-v1"
    spec = (ROOT / "docs" / "waste_dataset_spec.md").read_text(encoding="utf-8")
    assert LABEL_VERSION in spec


def test_ground_truth_formula_is_quoted_from_the_protocol():
    """연구 문서가 정의를 새로 만들지 않고 프로토콜을 인용한다."""
    protocol = (ROOT / "docs" / "waste_ground_truth_protocol.md").read_text(
        encoding="utf-8"
    )
    formula = "waste_rate = leftover_edible_mass_g / served_edible_mass_g"
    assert formula in protocol
    assert formula in REPORT.read_text(encoding="utf-8")
    assert formula in (
        ROOT / "research" / "waste_estimation" / "src" / "waste_estimation"
        / "schemas.py"
    ).read_text(encoding="utf-8")


# --- 실제 데이터 상태 -------------------------------------------------------


def test_audit_document_records_zero_actual_samples():
    """0을 감추지 않는다. 이 단계에서 가장 중요한 결과다."""
    text = " ".join(AUDIT.read_text(encoding="utf-8").split())
    assert "학습에 쓸 수 있는 실제 표본은 0건" in text


def test_repository_has_no_collected_waste_images():
    """저장소에 있는 이미지는 합성 fixture뿐이어야 한다."""
    data_dir = ROOT / "data"
    stray = [
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    assert stray == [], f"data/ 에 이미지가 있습니다: {stray}"


def test_synthetic_fixture_is_not_called_training_data():
    # 줄바꿈에 걸려도 문장은 같다. 공백을 접어서 본다.
    audit = " ".join(AUDIT.read_text(encoding="utf-8").split())
    assert "학습 데이터가 아니다" in audit


# --- 공개 데이터 -----------------------------------------------------------


def test_unverified_dataset_cannot_be_given_a_role():
    """조건을 못 읽은 데이터에 역할을 주면 코드가 거부한다."""
    with pytest.raises(ValueError):
        PublicDataset(
            name="made-up",
            official_source="https://example.invalid",
            license_name=None,
            license_status=LicenseStatus.UNVERIFIED,
            images=10,
            has_mass_ground_truth=False,
            has_before_after=False,
            has_segmentation=False,
            roles=(DatasetRole.BENCHMARK,),
            domain_gap="",
        )


def test_verified_dataset_must_name_its_license():
    with pytest.raises(ValueError):
        PublicDataset(
            name="made-up",
            official_source="https://example.invalid",
            license_name=None,
            license_status=LicenseStatus.VERIFIED,
            images=10,
            has_mass_ground_truth=False,
            has_before_after=False,
            has_segmentation=False,
            roles=(DatasetRole.REPRESENTATION,),
            domain_gap="",
        )


def test_every_candidate_records_its_official_source():
    for dataset in CANDIDATES:
        assert dataset.official_source.startswith("http"), dataset.name
        assert dataset.domain_gap.strip(), dataset.name


def test_unverified_candidates_are_not_usable():
    for dataset in CANDIDATES:
        if dataset.license_status is LicenseStatus.UNVERIFIED:
            assert dataset.usable is False, dataset.name
            assert dataset.usable_as_regression_ground_truth is False, dataset.name


def test_segmentation_only_dataset_is_never_a_regression_ground_truth():
    """마스크 면적 비율을 질량비로 바꾸지 않는다."""
    for dataset in CANDIDATES:
        if dataset.has_segmentation and not dataset.has_mass_ground_truth:
            assert dataset.usable_as_regression_ground_truth is False, dataset.name


def test_at_least_four_candidates_were_reviewed():
    assert len(CANDIDATES) >= 4


def test_usable_candidates_are_the_ones_with_read_licenses():
    names = {dataset.name for dataset in usable_candidates()}
    assert names == {"Nutrition5k", "LeFood-set"}
    assert {d.name for d in regression_ground_truth_candidates()} <= names


def test_datasets_document_warns_against_pixel_ratio_conversion():
    text = " ".join(DATASETS.read_text(encoding="utf-8").split())
    assert "남은 픽셀 비율" in text
    assert "leftover_edible_mass_g / served_edible_mass_g" in text


# --- 보고서가 필요한 것을 담고 있는가 ---------------------------------------


def test_report_states_the_information_limitation():
    text = " ".join(REPORT.read_text(encoding="utf-8").split())
    # 사진 한 장으로 정확한 중량 측정이 된다고 말하지 않는다.
    assert "원리적으로 정확한 중량 측정이 아니다" in text
    assert "추정한다" in text


def test_report_registers_all_four_research_questions():
    text = REPORT.read_text(encoding="utf-8")
    for name in ("RQ-W1", "RQ-W2", "RQ-W3", "RQ-W4"):
        assert name in text


def test_report_compares_three_formulations():
    text = REPORT.read_text(encoding="utf-8")
    for name in ("T1. Coarse Classification", "T2. Ordinal", "T3. Continuous"):
        assert name in text


def test_report_separates_failed_from_rejected():
    text = " ".join(REPORT.read_text(encoding="utf-8").split())
    assert "FAILED" in text
    assert "모델 판정이 아니다" in text
