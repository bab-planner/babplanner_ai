"""W1 — 정답 정의, 지표, 분할.

모델이 없어도 검증할 수 있는 것들이다. 오히려 모델보다 먼저 고정되어야
하는 것들이라 여기 있다.
"""

from __future__ import annotations

import pytest

from waste_estimation.audit import audit_images, audit_samples, summarize_rates
from waste_estimation.metrics import (
    MetricError,
    bias,
    class_index_mae,
    mean_absolute_error,
    root_mean_squared_error,
    to_class_index,
)
from waste_estimation.schemas import (
    SampleSource,
    WasteSample,
    WasteSchemaError,
    validate_waste_rate,
    waste_rate_from_mass,
)
from waste_estimation.selective import (
    coverage_at,
    coverage_risk_curve,
    risk_at_coverage,
    selective_risk,
)
from waste_estimation.splits import (
    SplitError,
    group_holdout_split,
    group_kfold,
    leaked_groups,
)


def sample(sample_id: str, group: str, rate: float | None = 0.2) -> WasteSample:
    return WasteSample(
        sample_id=sample_id,
        source=SampleSource.SYNTHETIC,
        group_id=group,
        waste_rate=rate,
    )


# --- 정답 정의 -------------------------------------------------------------


def test_waste_rate_is_the_mass_ratio_from_the_protocol():
    """500g 받아 100g 남기면 0.20. 프로토콜 문서의 예시 그대로다."""
    assert waste_rate_from_mass(500.0, 100.0) == pytest.approx(0.20)


def test_valid_waste_rate_passes():
    for value in (0.0, 0.5, 1.0):
        assert validate_waste_rate(value) == pytest.approx(value)


def test_negative_waste_rate_is_refused():
    with pytest.raises(WasteSchemaError):
        validate_waste_rate(-0.01)


def test_waste_rate_above_one_is_refused():
    """1.2를 1.0으로 잘라서 통과시키지 않는다. 그것은 측정 오류다."""
    with pytest.raises(WasteSchemaError):
        validate_waste_rate(1.2)


def test_served_mass_must_be_positive():
    with pytest.raises(WasteSchemaError):
        waste_rate_from_mass(0.0, 10.0)
    with pytest.raises(WasteSchemaError):
        waste_rate_from_mass(-100.0, 10.0)


def test_leftover_mass_cannot_be_negative():
    with pytest.raises(WasteSchemaError):
        waste_rate_from_mass(500.0, -1.0)


def test_leftover_cannot_exceed_served():
    """받은 것보다 많이 남길 수 없다."""
    with pytest.raises(WasteSchemaError):
        waste_rate_from_mass(300.0, 400.0)


def test_sample_requires_a_group_id():
    """분할 단위가 없으면 누수를 막을 수 없다."""
    with pytest.raises(WasteSchemaError):
        WasteSample(
            sample_id="a", source=SampleSource.PILOT, group_id="  ", waste_rate=0.1
        )


def test_sample_schema_has_no_personal_fields():
    """이름·학번·전화번호가 들어올 자리가 없다는 것을 필드로 확인한다."""
    from waste_estimation.schemas import FORBIDDEN_FIELD_WORDS

    fields = vars(sample("a", "g")).keys()
    for field in fields:
        for word in FORBIDDEN_FIELD_WORDS:
            assert word not in field.lower(), field


# --- 회귀 지표 -------------------------------------------------------------


def test_mae_comes_from_the_service_implementation():
    """지표를 두 번 구현하지 않는다. 갈라지는 날이 온다."""
    from babplanner_ai.dataset import metrics as service_metrics

    assert mean_absolute_error is service_metrics.mean_absolute_error
    assert root_mean_squared_error is service_metrics.root_mean_squared_error


def test_mae_and_rmse():
    actual = [0.10, 0.20, 0.30]
    predicted = [0.20, 0.20, 0.10]
    assert mean_absolute_error(actual, predicted) == pytest.approx(0.10)
    assert root_mean_squared_error(actual, predicted) == pytest.approx(
        ((0.10**2 + 0.0 + 0.20**2) / 3) ** 0.5
    )


def test_bias_keeps_the_sign_that_mae_throws_away():
    """늘 적게 예측하는 것과 늘 많게 예측하는 것은 MAE가 같아도 다른 문제다."""
    actual = [0.20, 0.20]
    under = [0.10, 0.10]
    over = [0.30, 0.30]
    assert mean_absolute_error(actual, under) == pytest.approx(
        mean_absolute_error(actual, over)
    )
    assert bias(actual, under) == pytest.approx(-0.10)
    assert bias(actual, over) == pytest.approx(0.10)


def test_bias_refuses_empty_input():
    with pytest.raises(MetricError):
        bias([], [])


# --- 순서형 지표 -----------------------------------------------------------


def test_class_index_needs_explicit_edges():
    """구간 경계에 기본값을 두지 않는다. 정해지지 않은 정책이 굳는다."""
    import inspect

    signature = inspect.signature(to_class_index)
    assert signature.parameters["edges"].default is inspect.Parameter.empty


def test_class_index_boundary_is_inclusive_downward():
    edges = [0.2, 0.5]
    assert to_class_index(0.0, edges) == 0
    assert to_class_index(0.2, edges) == 0
    assert to_class_index(0.21, edges) == 1
    assert to_class_index(0.5, edges) == 1
    assert to_class_index(0.51, edges) == 2


def test_class_index_refuses_unsorted_edges():
    with pytest.raises(MetricError):
        to_class_index(0.3, [0.5, 0.2])


def test_class_index_mae_counts_distance_not_just_correctness():
    """세 칸 틀린 것과 한 칸 틀린 것을 같게 세지 않는다."""
    assert class_index_mae([0, 0], [1, 1]) == pytest.approx(1.0)
    assert class_index_mae([0, 0], [3, 3]) == pytest.approx(3.0)
    assert class_index_mae([1, 2], [1, 2]) == pytest.approx(0.0)


def test_class_index_mae_refuses_non_integer_classes():
    with pytest.raises(MetricError):
        class_index_mae([0.5], [1])


# --- 선택적 예측 -----------------------------------------------------------


def test_coverage_is_the_approved_fraction():
    assert coverage_at([0.9, 0.8, 0.2, 0.1], 0.5) == pytest.approx(0.5)
    assert coverage_at([0.9, 0.8], 0.0) == pytest.approx(1.0)


def test_selective_risk_only_scores_approved_samples():
    actual = [0.10, 0.10, 0.10]
    predicted = [0.10, 0.10, 0.90]  # 마지막이 크게 틀렸다
    confidence = [0.9, 0.9, 0.1]

    assert selective_risk(actual, predicted, confidence, 0.0) == pytest.approx(
        0.8 / 3
    )
    # 틀린 것을 거절하면 남은 것의 오차가 0이다.
    assert selective_risk(actual, predicted, confidence, 0.5) == pytest.approx(0.0)


def test_zero_coverage_returns_none_not_zero():
    """아무것도 승인하지 않은 모델을 완벽하다고 말하지 않는다."""
    result = selective_risk([0.1], [0.9], [0.2], threshold=0.9)
    assert result is None


def test_coverage_risk_curve_is_ordered_and_uses_observed_thresholds():
    actual = [0.10, 0.10, 0.10, 0.10]
    predicted = [0.10, 0.12, 0.40, 0.90]
    confidence = [0.9, 0.8, 0.4, 0.1]

    points = coverage_risk_curve(actual, predicted, confidence)
    coverages = [point.coverage for point in points]
    assert coverages == sorted(coverages, reverse=True)
    assert points[0].coverage == pytest.approx(1.0)
    # 버릴수록 남은 것의 오차가 줄어든다.
    assert points[0].risk > points[-1].risk


def test_risk_at_coverage_returns_none_when_target_unreachable():
    actual = [0.1, 0.1]
    predicted = [0.1, 0.5]
    confidence = [0.9, 0.9]
    points = coverage_risk_curve(actual, predicted, confidence)
    # 모든 점의 coverage가 1.0이므로 그보다 큰 목표는 없다.
    assert risk_at_coverage(points, 1.0) is not None
    with pytest.raises(MetricError):
        risk_at_coverage(points, 0.0)


def test_selective_metrics_refuse_mismatched_lengths():
    with pytest.raises(MetricError):
        selective_risk([0.1, 0.2], [0.1], [0.9], threshold=0.5)


# --- 분할 -----------------------------------------------------------------


def test_group_split_never_leaks_a_group():
    samples = [sample(f"s{i}", f"g{i // 3}") for i in range(30)]
    train, holdout = group_holdout_split(samples, holdout_ratio=0.3, seed=7)

    assert train and holdout
    assert leaked_groups(train, holdout) == set()


def test_group_split_is_deterministic_for_a_seed():
    samples = [sample(f"s{i}", f"g{i // 2}") for i in range(20)]
    first = group_holdout_split(samples, seed=1)[1]
    second = group_holdout_split(samples, seed=1)[1]
    assert [s.sample_id for s in first] == [s.sample_id for s in second]


def test_group_split_refuses_a_single_group():
    """그룹이 하나면 누수 없이 나눌 수 없다. 조용히 나누지 않는다."""
    samples = [sample(f"s{i}", "only-one") for i in range(10)]
    with pytest.raises(SplitError):
        group_holdout_split(samples)


def test_group_kfold_covers_every_sample_exactly_once():
    samples = [sample(f"s{i}", f"g{i // 2}") for i in range(20)]
    folds = group_kfold(samples, folds=5, seed=3)

    assert len(folds) == 5
    seen: list[str] = []
    for train, test in folds:
        assert leaked_groups(train, test) == set()
        seen += [s.sample_id for s in test]
    assert sorted(seen) == sorted(s.sample_id for s in samples)


# --- 데이터 점검 -----------------------------------------------------------


def test_audit_counts_labels_and_duplicates():
    samples = [
        sample("a", "g1", 0.1),
        sample("b", "g1", None),
        sample("a", "g2", 0.4),  # 중복 sample_id
    ]
    report = audit_samples(samples)

    assert report.total == 3
    assert report.labeled == 2
    assert report.unlabeled == 1
    assert report.duplicate_sample_ids == ["a"]
    assert report.groups == 2


def test_audit_says_not_usable_when_there_are_no_labels():
    """정답이 하나도 없으면 무엇을 해도 학습이 아니다."""
    report = audit_samples([sample("a", "g1", None)])
    assert report.labeled == 0
    assert report.usable_for_training is False


def test_distribution_of_an_empty_set_is_none_not_zero():
    """표본이 없는데 평균 0을 내면 '잔반률 0%'로 읽힌다."""
    summary = summarize_rates([])
    assert summary.count == 0
    assert summary.mean is None
    assert summary.median is None


def test_distribution_summarises_actual_values():
    summary = summarize_rates([0.0, 0.2, 0.4], bin_width=0.2)
    assert summary.count == 3
    assert summary.minimum == pytest.approx(0.0)
    assert summary.maximum == pytest.approx(0.4)
    assert summary.median == pytest.approx(0.2)


def test_image_audit_reports_a_missing_directory_as_zero(tmp_path):
    report = audit_images(tmp_path / "nope")
    assert report.exists is False
    assert report.image_count == 0


def test_image_audit_finds_exact_duplicates(tmp_path):
    """같은 사진이 학습과 평가에 함께 들어가는 가장 흔한 경우를 잡는다."""
    (tmp_path / "a.png").write_bytes(b"same-bytes")
    (tmp_path / "b.png").write_bytes(b"same-bytes")
    (tmp_path / "c.png").write_bytes(b"different")

    report = audit_images(tmp_path)
    assert report.image_count == 3
    assert report.duplicate_groups == [["a.png", "b.png"]]
    assert report.unique_images == 2


def test_image_audit_ignores_non_image_files(tmp_path):
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "a.png").write_bytes(b"img")
    assert audit_images(tmp_path).image_count == 1
