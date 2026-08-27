"""평가 지표."""

from __future__ import annotations

import math

import pytest

from cctv_counting.metrics import (
    ConditionScores,
    MetricError,
    entry_count_error,
    entry_count_mae,
    occupancy_bias,
    occupancy_mae,
    occupancy_rmse,
    render_condition_table,
    score_by_condition,
    score_occupancy,
)


def test_perfect_prediction_has_zero_error():
    assert occupancy_mae([3, 5, 2], [3, 5, 2]) == 0.0
    assert occupancy_rmse([3, 5, 2], [3, 5, 2]) == 0.0


def test_occupancy_mae_is_in_units_of_people():
    """MAE 2.0은 '평균 두 명 틀린다'로 그대로 읽혀야 한다."""
    assert occupancy_mae([10, 10], [8, 14]) == pytest.approx(3.0)


def test_occupancy_rmse_punishes_large_errors():
    mild = occupancy_rmse([10, 10], [8, 12])
    harsh = occupancy_rmse([10, 10], [10, 14])
    assert harsh > mild


def test_rmse_matches_the_definition():
    assert occupancy_rmse([0, 0], [3, 4]) == pytest.approx(math.sqrt(12.5))


def test_bias_separates_over_and_under_counting():
    """MAE가 같아도 계속 적게 세는 것과 많이 세는 것은 다른 문제다."""
    assert occupancy_bias([10, 10], [12, 12]) == pytest.approx(2.0)
    assert occupancy_bias([10, 10], [8, 8]) == pytest.approx(-2.0)
    assert occupancy_mae([10, 10], [12, 12]) == occupancy_mae([10, 10], [8, 8])


def test_score_occupancy_bundles_the_three():
    scores = score_occupancy([4, 4], [5, 3])
    assert scores.count == 2
    assert scores.mae == pytest.approx(1.0)
    assert scores.bias == pytest.approx(0.0)
    assert "MAE" in scores.render()


def test_empty_input_is_rejected_not_scored_zero():
    """0건에 대한 MAE 0.0은 완벽한 성능처럼 보이지만 아무것도 재지 않은 것이다."""
    with pytest.raises(MetricError):
        occupancy_mae([], [])


def test_length_mismatch_is_rejected():
    with pytest.raises(MetricError):
        occupancy_mae([1, 2, 3], [1, 2])


def test_nan_is_rejected():
    with pytest.raises(MetricError):
        occupancy_mae([1.0], [float("nan")])


def test_infinity_is_rejected():
    with pytest.raises(MetricError):
        occupancy_rmse([1.0], [float("inf")])


def test_negative_counts_are_rejected():
    with pytest.raises(MetricError):
        occupancy_mae([-1], [1])


def test_zero_actual_count_is_fine_for_mae():
    """빈 식당 구간에서도 MAE는 정의된다. MAPE를 안 쓰는 이유가 이것이다."""
    assert occupancy_mae([0, 0, 4], [1, 0, 4]) == pytest.approx(1 / 3)


def test_entry_count_error_is_absolute():
    assert entry_count_error(actual=120, predicted=113) == 7
    assert entry_count_error(actual=113, predicted=120) == 7


def test_entry_count_error_rejects_negative():
    with pytest.raises(MetricError):
        entry_count_error(actual=-1, predicted=3)


def test_entry_count_error_rejects_non_integer():
    with pytest.raises(MetricError):
        entry_count_error(actual=1.5, predicted=3)  # type: ignore[arg-type]


def test_entry_count_mae_across_videos():
    assert entry_count_mae([100, 200], [90, 210]) == pytest.approx(10.0)


def test_entry_count_mae_rejects_empty():
    with pytest.raises(MetricError):
        entry_count_mae([], [])


def test_condition_scores_group_by_tag():
    rows = score_by_condition(
        [
            ("daylight", 10.0, 10.0),
            ("daylight", 8.0, 9.0),
            ("low-light", 12.0, 6.0),
        ]
    )
    assert [row.condition for row in rows] == ["daylight", "low-light"]
    assert rows[0].samples == 2
    assert rows[1].occupancy.mae == pytest.approx(6.0)


def test_condition_scores_reject_a_blank_tag():
    """조건을 모르는 표본에 조건 이름을 붙이면 조건별 표 전체가 거짓이 된다."""
    with pytest.raises(MetricError):
        score_by_condition([("", 1.0, 1.0)])


def test_condition_table_without_annotations_says_so():
    table = render_condition_table([])
    assert "조건 annotation 없음" in table
    assert "N/A" in table


def test_condition_table_renders_rows():
    table = render_condition_table(
        [
            ConditionScores(
                condition="crowded",
                samples=4,
                occupancy=score_occupancy([10, 10], [8, 8]),
                entry_error=None,
            )
        ]
    )
    assert "| crowded | 4 | 2.000 | N/A |" in table
