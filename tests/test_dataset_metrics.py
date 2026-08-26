"""평가 지표."""

from __future__ import annotations

import inspect
import math

import pytest

from babplanner_ai.dataset import metrics
from babplanner_ai.dataset.metrics import (
    MetricError,
    mean_absolute_error,
    median_absolute_error,
    root_mean_squared_error,
    score,
)


def test_mae_is_the_mean_absolute_error() -> None:
    actual = [0.2, 0.5, 0.8]
    predicted = [0.3, 0.5, 0.6]

    assert mean_absolute_error(actual, predicted) == pytest.approx(0.1)


def test_mae_is_zero_for_a_perfect_match() -> None:
    assert mean_absolute_error([0.2, 0.4], [0.2, 0.4]) == 0.0


def test_rmse_punishes_large_errors_more() -> None:
    actual = [0.0, 0.0]
    predicted = [0.0, 0.4]

    assert root_mean_squared_error(actual, predicted) == pytest.approx(
        math.sqrt((0.0 + 0.16) / 2)
    )
    assert root_mean_squared_error(actual, predicted) > mean_absolute_error(
        actual, predicted
    )


def test_median_absolute_error_odd_count() -> None:
    assert median_absolute_error([0.0, 0.0, 0.0], [0.1, 0.5, 0.2]) == pytest.approx(0.2)


def test_median_absolute_error_even_count() -> None:
    assert median_absolute_error([0.0, 0.0], [0.2, 0.4]) == pytest.approx(0.3)


def test_median_is_steadier_than_the_mean() -> None:
    actual = [0.0] * 5
    predicted = [0.01, 0.01, 0.01, 0.01, 0.9]

    assert median_absolute_error(actual, predicted) < mean_absolute_error(
        actual, predicted
    )


def test_empty_input_is_refused() -> None:
    # 빈 입력에 0을 돌려주면 완벽한 성능처럼 보인다.
    with pytest.raises(MetricError, match="비어 있습니다"):
        mean_absolute_error([], [])


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(MetricError, match="개수가 다릅니다"):
        mean_absolute_error([0.1, 0.2], [0.1])


@pytest.mark.parametrize("bad", [-0.01, 1.01, 5.0])
def test_values_outside_the_range_are_refused(bad: float) -> None:
    with pytest.raises(MetricError, match="0~1"):
        mean_absolute_error([0.5], [bad])


def test_out_of_range_truth_is_refused() -> None:
    with pytest.raises(MetricError, match="0~1"):
        mean_absolute_error([1.5], [0.5])


def test_nan_is_refused() -> None:
    with pytest.raises(MetricError, match="숫자가 아닙니다"):
        mean_absolute_error([float("nan")], [0.5])


def test_score_reports_all_three() -> None:
    scores = score([0.2, 0.5, 0.8], [0.3, 0.5, 0.6])

    assert scores.count == 3
    assert scores.mae == pytest.approx(0.1)
    assert scores.rmse >= scores.mae
    assert "MAE" in scores.render()


def test_render_explains_the_scale() -> None:
    # MAE 0.08 이 "8퍼센트포인트"로 읽히는 것이 이 지표를 고른 이유다.
    text = score([0.0], [0.08]).render()

    assert "8.00%p" in text


def test_no_threshold_classification_metrics() -> None:
    # 운영 경계가 아직 없다. 지금 만들면 정해지지도 않은 정책이 지표로 굳는다.
    source = inspect.getsource(metrics)

    for forbidden in ("accuracy", "precision", "recall", "f1", "threshold", "eligible"):
        assert forbidden not in source.lower()
