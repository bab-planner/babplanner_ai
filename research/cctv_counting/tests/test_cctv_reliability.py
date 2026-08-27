"""신뢰 신호."""

from __future__ import annotations

import pytest

from cctv_counting.reliability import (
    ReliabilityError,
    ReliabilityFlag,
    ReliabilityPolicy,
    evaluate_reliability,
)


def test_confident_detections_are_ok():
    signal = evaluate_reliability([0.9, 0.85], 2, ReliabilityPolicy())
    assert signal.flag is ReliabilityFlag.OK
    assert not signal.is_low_reliability
    assert signal.mean_detection_confidence == pytest.approx(0.875)
    assert signal.min_detection_confidence == pytest.approx(0.85)


def test_weak_detections_are_flagged():
    signal = evaluate_reliability(
        [0.3, 0.35], 2, ReliabilityPolicy(min_mean_confidence=0.5)
    )
    assert signal.flag is ReliabilityFlag.LOW_CONFIDENCE
    assert signal.is_low_reliability


def test_empty_frame_has_no_confidence_not_zero():
    """0.0은 '신뢰도가 0이었다'로 읽힌다. '잴 것이 없었다'와 다르다."""
    signal = evaluate_reliability([], 0, ReliabilityPolicy())
    assert signal.mean_detection_confidence is None
    assert signal.min_detection_confidence is None
    assert signal.flag is ReliabilityFlag.NO_DETECTION


def test_empty_frame_is_treated_as_low_reliability():
    """빈 식당과 전부 놓친 화면을 여기서는 구분할 수 없다."""
    signal = evaluate_reliability([], 0, ReliabilityPolicy())
    assert signal.is_low_reliability


def test_crowding_is_flagged_when_the_policy_asks():
    signal = evaluate_reliability(
        [0.95] * 40, 40, ReliabilityPolicy(crowded_track_count=30)
    )
    assert signal.flag is ReliabilityFlag.CROWDED


def test_crowding_wins_over_confidence():
    """상자가 겹쳐 과소 추정되는 상황이 낮은 신뢰도보다 먼저 알려져야 한다."""
    signal = evaluate_reliability(
        [0.2] * 40,
        40,
        ReliabilityPolicy(min_mean_confidence=0.5, crowded_track_count=30),
    )
    assert signal.flag is ReliabilityFlag.CROWDED


def test_crowding_is_off_by_default():
    signal = evaluate_reliability([0.95] * 100, 100, ReliabilityPolicy())
    assert signal.flag is ReliabilityFlag.OK


def test_high_confidence_does_not_mean_correct_count():
    """confidence는 개수의 정확도가 아니다. 문서가 아니라 값으로 남겨 둔다.

    두 명이 겹쳐 상자 하나로 잡힌 상황을 흉내 낸다. 모델은 확신하고,
    개수는 절반이다. 신호는 OK다.
    """
    signal = evaluate_reliability([0.99], 1, ReliabilityPolicy())
    assert signal.flag is ReliabilityFlag.OK
    assert signal.active_track_count == 1


def test_rejects_confidence_out_of_range():
    with pytest.raises(ReliabilityError):
        evaluate_reliability([1.2], 1, ReliabilityPolicy())


def test_rejects_nan_confidence():
    with pytest.raises(ReliabilityError):
        evaluate_reliability([float("nan")], 1, ReliabilityPolicy())


def test_rejects_negative_track_count():
    with pytest.raises(ReliabilityError):
        evaluate_reliability([0.9], -1, ReliabilityPolicy())


def test_policy_rejects_bad_threshold():
    with pytest.raises(ReliabilityError):
        ReliabilityPolicy(min_mean_confidence=1.5)


def test_policy_rejects_zero_crowd_threshold():
    with pytest.raises(ReliabilityError):
        ReliabilityPolicy(crowded_track_count=0)
