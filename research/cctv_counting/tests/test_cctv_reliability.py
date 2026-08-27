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


def test_reliability_has_no_crowding_state():
    """혼잡도는 신뢰도가 아니다. 백엔드의 CROWDED 등급과도 섞이면 안 된다."""
    assert {flag.value for flag in ReliabilityFlag} == {
        "OK",
        "LOW_CONFIDENCE",
        "NO_DETECTION",
    }


def test_a_full_room_is_not_automatically_unreliable():
    """사람이 많다고 관측을 자동으로 못 믿을 것으로 만들지 않는다.

    몇 명부터 붐비는지는 급식소 좌석 수에 달렸고 그 임계값은 아직 없다.
    """
    signal = evaluate_reliability([0.95] * 100, 100, ReliabilityPolicy())
    assert signal.flag is ReliabilityFlag.OK


def test_policy_does_not_accept_a_crowding_threshold():
    """설정에서도 이 축을 되살릴 수 없어야 한다."""
    with pytest.raises(TypeError):
        ReliabilityPolicy(crowded_track_count=30)  # type: ignore[call-arg]


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


def test_no_detection_does_not_claim_the_room_was_empty():
    """이 값은 "식당이 비었다"도 "모델이 실패했다"도 아니다.

    detector가 유효한 사람 상자를 하나도 내지 않았다는 사실만 말한다. 실제로
    아무도 없었는지, 있었는데 전부 놓쳤는지는 런타임에서 구분할 수 없다.
    """
    empty_room = evaluate_reliability([], 0, ReliabilityPolicy())
    everyone_missed = evaluate_reliability([], 0, ReliabilityPolicy())
    assert empty_room.flag is everyone_missed.flag is ReliabilityFlag.NO_DETECTION
    assert empty_room == everyone_missed


def test_high_confidence_cannot_detect_missed_people():
    """누락은 이 신호로 절대 잡히지 않는다.

    실제로 10명이 있는데 5명만 잡혔고 그 5개가 전부 0.95라면, 평균 신뢰도는
    높고 flag는 OK이며 계수는 절반이다. 못 잡은 것에는 confidence 자체가
    없기 때문이다. 이 성질을 값으로 고정해 둔다.
    """
    all_ten_found = evaluate_reliability([0.95] * 10, 10, ReliabilityPolicy())
    only_five_found = evaluate_reliability([0.95] * 5, 5, ReliabilityPolicy())
    assert all_ten_found.flag is only_five_found.flag is ReliabilityFlag.OK
    assert (
        all_ten_found.mean_detection_confidence
        == only_five_found.mean_detection_confidence
    )


def test_reliability_is_a_heuristic_not_a_calibrated_probability():
    """flag는 오차 확률이 아니다. 임계값 하나로 갈리는 heuristic이다."""
    policy = ReliabilityPolicy(min_mean_confidence=0.5)
    just_below = evaluate_reliability([0.499], 1, policy)
    just_above = evaluate_reliability([0.501], 1, policy)
    assert just_below.flag is ReliabilityFlag.LOW_CONFIDENCE
    assert just_above.flag is ReliabilityFlag.OK
