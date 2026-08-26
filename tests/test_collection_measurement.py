"""저울 숫자를 정답으로 바꾸는 계산."""

from __future__ import annotations

import pytest

from babplanner_ai.collection.measurement import compute_mass, repeat_stats
from babplanner_ai.collection.models import (
    MeasurementError,
    RawMeasurementRecord,
    TareProfile,
)
from babplanner_ai.dataset.schema import (
    AnalysisLabel,
    RejectionReason,
    ServedMassSource,
)

NOW = "2026-08-27T04:00:00Z"


def tare(*weights: float) -> TareProfile:
    return TareProfile(
        tare_profile_id="tp_000000000001",
        description="4칸 식판",
        tare_measurements_g=list(weights) or [400.0],
        measured_at=NOW,
    )


def record(**overrides: object) -> RawMeasurementRecord:
    values: dict[str, object] = {
        "sample_id": "s1",
        "collection_session_id": "cs_000000000001",
        "capture_session_id": "cap_000000000001",
        "menu_id": 3,
        "tare_profile_id": "tp_000000000001",
        "pre_meal_total_weight_g": 900.0,
        "post_meal_total_weight_g": 500.0,
        "served_mass_source": ServedMassSource.MEASURED,
        "analysis_label": AnalysisLabel.APPROVED,
        "created_at": NOW,
    }
    values.update(overrides)
    return RawMeasurementRecord(**values)  # type: ignore[arg-type]


def test_measured_path_subtracts_the_tare() -> None:
    mass = compute_mass(record(), tare(400.0))

    assert mass.served_edible_mass_g == 500.0
    assert mass.leftover_edible_mass_g == 100.0
    assert mass.waste_rate == pytest.approx(0.2)


def test_nothing_left_is_zero() -> None:
    mass = compute_mass(record(post_meal_total_weight_g=400.0), tare(400.0))

    assert mass.leftover_edible_mass_g == 0.0
    assert mass.waste_rate == 0.0


def test_everything_left_is_one() -> None:
    mass = compute_mass(record(post_meal_total_weight_g=900.0), tare(400.0))

    assert mass.waste_rate == 1.0


def test_pre_weight_at_or_below_tare_is_refused() -> None:
    # 받은 음식이 없으면 잔반률을 말할 수 없다.
    with pytest.raises(MeasurementError, match="식기 무게보다 크지 않"):
        compute_mass(record(pre_meal_total_weight_g=400.0), tare(400.0))


def test_post_weight_below_tare_is_refused() -> None:
    with pytest.raises(MeasurementError, match="식기 무게보다 작"):
        compute_mass(record(post_meal_total_weight_g=350.0), tare(400.0))


def test_leftover_above_served_is_refused() -> None:
    # 받은 것보다 많이 남길 수 없다.
    with pytest.raises(MeasurementError, match="잔반량이 제공량보다 많"):
        compute_mass(
            record(pre_meal_total_weight_g=500.0, post_meal_total_weight_g=900.0),
            tare(400.0),
        )


def test_out_of_range_is_not_clipped() -> None:
    # 1.04를 1로 깎으면 그 표본은 정상처럼 보이지만 안에는 잘못된 측정이 남는다.
    with pytest.raises(MeasurementError):
        compute_mass(
            record(pre_meal_total_weight_g=500.0, post_meal_total_weight_g=504.0),
            tare(400.0),
        )


def test_fractional_grams_are_kept() -> None:
    # 0.1g 저울의 정밀도를 반올림으로 버리지 않는다.
    mass = compute_mass(
        record(pre_meal_total_weight_g=900.4, post_meal_total_weight_g=500.2),
        tare(400.1),
    )

    assert mass.served_edible_mass_g == pytest.approx(500.3)
    assert mass.leftover_edible_mass_g == pytest.approx(100.1)


def test_missing_tare_is_refused() -> None:
    with pytest.raises(MeasurementError, match="tare 프로파일이 필요"):
        compute_mass(record(), None)


def test_missing_weight_is_refused() -> None:
    with pytest.raises(MeasurementError, match="배식 후 전체 중량"):
        compute_mass(record(pre_meal_total_weight_g=None), tare(400.0))


def test_standard_portion_uses_the_declared_amount() -> None:
    mass = compute_mass(
        record(
            served_mass_source=ServedMassSource.STANDARD_PORTION,
            standard_served_edible_mass_g=500.0,
            pre_meal_total_weight_g=None,
            post_meal_total_weight_g=450.0,
        ),
        tare(400.0),
    )

    assert mass.served_edible_mass_g == 500.0
    assert mass.leftover_edible_mass_g == 50.0
    assert mass.waste_rate == pytest.approx(0.1)


def test_standard_portion_needs_the_declared_amount() -> None:
    with pytest.raises(MeasurementError, match="표준 배식량"):
        compute_mass(
            record(
                served_mass_source=ServedMassSource.STANDARD_PORTION,
                standard_served_edible_mass_g=None,
            ),
            tare(400.0),
        )


def test_visual_estimate_cannot_be_recorded_as_truth() -> None:
    # 눈대중은 정답이 아니다. 수집 도구의 정답 경로로 들어오지 못하게 한다.
    with pytest.raises(Exception, match="눈대중"):
        record(served_mass_source=ServedMassSource.VISUAL_ESTIMATE)


def test_rejected_record_gets_no_rate() -> None:
    rejected = record(
        analysis_label=AnalysisLabel.REJECTED,
        rejection_reason=RejectionReason.SEVERE_BLUR,
        served_mass_source=None,
    )

    with pytest.raises(MeasurementError, match="거절된 기록"):
        compute_mass(rejected, tare(400.0))


def test_rejected_record_needs_a_reason() -> None:
    with pytest.raises(Exception, match="사유"):
        record(analysis_label=AnalysisLabel.REJECTED, served_mass_source=None)


def test_repeat_stats_describe_the_spread() -> None:
    stats = repeat_stats([880.0, 880.5, 879.5])

    assert stats is not None
    assert stats.count == 3
    assert stats.spread == pytest.approx(1.0)
    assert stats.median == pytest.approx(880.0)


def test_repeat_stats_of_nothing_is_none() -> None:
    assert repeat_stats([]) is None
