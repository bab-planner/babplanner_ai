"""정답 라벨의 조건.

여기서 지키는 것은 하나다. 측정이 이상하면 조용히 고쳐 정상처럼 만들지
않는다.
"""

from __future__ import annotations

import pytest

from babplanner_ai.dataset.schema import (
    CURRENT_LABEL_VERSION,
    AnalysisLabel,
    DatasetSample,
    GtExclusionReason,
    RejectionReason,
    ServedMassSource,
)


def approved(**overrides: object) -> DatasetSample:
    values: dict[str, object] = {
        "sample_id": "s1",
        "image_path": "images/s1.png",
        "menu_id": 3,
        "capture_session_id": "S01",
        "participant_group_id": "P01",
        "served_edible_mass_g": 500.0,
        "leftover_edible_mass_g": 100.0,
        "waste_rate": 0.2,
        "served_mass_source": ServedMassSource.MEASURED,
        "analysis_label": AnalysisLabel.APPROVED,
        "label_version": CURRENT_LABEL_VERSION,
    }
    values.update(overrides)
    return DatasetSample(**values)  # type: ignore[arg-type]


def rejected(**overrides: object) -> DatasetSample:
    values: dict[str, object] = {
        "sample_id": "s2",
        "image_path": "images/s2.png",
        "menu_id": 3,
        "capture_session_id": "S01",
        "analysis_label": AnalysisLabel.REJECTED,
        "rejection_reason": RejectionReason.SEVERE_BLUR,
        "label_version": CURRENT_LABEL_VERSION,
    }
    values.update(overrides)
    return DatasetSample(**values)  # type: ignore[arg-type]


def test_measured_sample_is_valid() -> None:
    sample = approved()

    assert sample.waste_rate == 0.2
    assert sample.is_primary_benchmark


def test_rate_zero_means_nothing_left() -> None:
    sample = approved(leftover_edible_mass_g=0.0, waste_rate=0.0)

    assert sample.waste_rate == 0.0


def test_rate_one_means_everything_left() -> None:
    sample = approved(leftover_edible_mass_g=500.0, waste_rate=1.0)

    assert sample.waste_rate == 1.0


def test_served_mass_must_be_positive() -> None:
    # 아무것도 받지 않은 끼니의 잔반률은 정의되지 않는다.
    with pytest.raises(Exception):
        approved(served_edible_mass_g=0.0, leftover_edible_mass_g=0.0, waste_rate=0.0)


def test_negative_leftover_is_rejected() -> None:
    # 0으로 깎아서 정상 라벨로 만들지 않는다. 측정이 틀렸다는 신호다.
    with pytest.raises(Exception):
        approved(leftover_edible_mass_g=-5.0, waste_rate=0.0)


def test_leftover_above_served_is_rejected() -> None:
    with pytest.raises(Exception):
        approved(leftover_edible_mass_g=600.0, waste_rate=1.0)


def test_rate_must_match_the_measured_ratio() -> None:
    # 잔반률은 두 무게에서 나오는 값이지 따로 적는 값이 아니다.
    with pytest.raises(Exception):
        approved(waste_rate=0.5)


def test_rate_tolerates_only_serialization_error() -> None:
    sample = approved(waste_rate=0.2 + 1e-9)

    assert sample.waste_rate == pytest.approx(0.2)


def test_rate_above_one_is_not_clipped() -> None:
    with pytest.raises(Exception):
        approved(leftover_edible_mass_g=515.0, waste_rate=1.03)


def test_approved_requires_a_rate() -> None:
    with pytest.raises(Exception):
        approved(waste_rate=None)


def test_approved_requires_a_mass_source() -> None:
    with pytest.raises(Exception):
        approved(served_mass_source=None)


def test_approved_cannot_carry_a_rejection_reason() -> None:
    with pytest.raises(Exception):
        approved(rejection_reason=RejectionReason.SEVERE_BLUR)


def test_rejected_sample_needs_no_rate() -> None:
    sample = rejected()

    assert sample.waste_rate is None
    assert not sample.is_primary_benchmark


def test_rejected_sample_must_not_carry_a_rate() -> None:
    # 0으로 채우면 "다 먹었다"는 뜻이 되어 정답이 오염된다.
    with pytest.raises(Exception):
        rejected(waste_rate=0.0)


def test_rejected_sample_requires_a_reason() -> None:
    with pytest.raises(Exception):
        rejected(rejection_reason=None)


def test_label_version_is_required() -> None:
    with pytest.raises(Exception):
        approved(label_version="")


def test_unknown_mass_source_is_rejected() -> None:
    with pytest.raises(Exception):
        approved(served_mass_source="GUESSED")


def test_visual_estimate_is_not_primary_benchmark() -> None:
    # 사람이 눈으로 매긴 값은 기록할 수는 있어도 정답 집합에는 못 들어간다.
    sample = approved(served_mass_source=ServedMassSource.VISUAL_ESTIMATE)

    assert sample.analysis_label is AnalysisLabel.APPROVED
    assert not sample.is_primary_benchmark


def test_standard_portion_is_not_primary_benchmark() -> None:
    sample = approved(served_mass_source=ServedMassSource.STANDARD_PORTION)

    assert not sample.is_primary_benchmark


def test_schema_has_no_confidence_target() -> None:
    # confidence는 사람이 붙이는 정답이 아니라 모델이 내는 값이다.
    fields = set(DatasetSample.model_fields)

    assert "confidence" not in fields
    assert "confidence_target" not in fields


def test_unknown_columns_are_rejected() -> None:
    with pytest.raises(Exception):
        approved(reward_points=100)


def test_group_key_prefers_participant() -> None:
    participant, session = approved().group_key_candidates

    assert participant == "P01"
    assert session == "S01"


# --- 정답 제외 --------------------------------------------------------------
#
# 뼈가 든 메뉴는 가식부만 따로 잴 수 없어 질량비가 그 식판의 잔반률을 뜻하지
# 않는다. 버리지는 않되 정답으로는 쓰지 않는다.


def test_no_exclusion_by_default() -> None:
    sample = approved()

    assert sample.gt_exclusion_reason is None
    assert not sample.is_gt_excluded
    assert sample.is_primary_benchmark
    assert sample.is_supervised_usable


def test_excluded_sample_leaves_the_benchmark_and_training() -> None:
    sample = approved(gt_exclusion_reason=GtExclusionReason.BONE_IN)

    assert sample.is_gt_excluded
    assert not sample.is_primary_benchmark
    assert not sample.is_supervised_usable
    # 값은 그대로 남는다. 빼는 것이지 지우는 것이 아니다.
    assert sample.waste_rate is not None


def test_other_needs_a_note() -> None:
    # OTHER만 남으면 나중에 왜 뺐는지 알 방법이 없다.
    with pytest.raises(Exception, match="이유를 함께"):
        approved(gt_exclusion_reason=GtExclusionReason.OTHER)


def test_other_with_a_note_is_fine() -> None:
    sample = approved(
        gt_exclusion_reason=GtExclusionReason.OTHER,
        gt_exclusion_note="식후에 국그릇이 하나 늘었다",
    )

    assert sample.is_gt_excluded


def test_bone_in_does_not_need_a_note() -> None:
    # 사유 이름만으로 뜻이 선다. 설명을 강요하면 아무 말이나 적게 된다.
    assert approved(gt_exclusion_reason=GtExclusionReason.BONE_IN).is_gt_excluded


def test_note_without_a_reason_is_rejected() -> None:
    # 설명만 남으면 이 표본이 빠지는지 아닌지를 읽는 사람마다 다르게 본다.
    with pytest.raises(Exception, match="제외 사유가 없습니다"):
        approved(gt_exclusion_note="뼈가 있었던 것 같다")


def test_rejected_sample_can_also_carry_an_exclusion() -> None:
    # 뼈 메뉴에서 흐린 사진이 나왔다면 두 사실이 다 남아야 한다.
    sample = DatasetSample(
        sample_id="s9",
        image_path="images/s9.png",
        menu_id=7,
        capture_session_id="S01",
        analysis_label=AnalysisLabel.REJECTED,
        rejection_reason=RejectionReason.SEVERE_BLUR,
        gt_exclusion_reason=GtExclusionReason.BONE_IN,
        label_version=CURRENT_LABEL_VERSION,
    )

    assert sample.is_gt_excluded
    assert not sample.is_supervised_usable
