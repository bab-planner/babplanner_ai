"""선 통과 판정과 중복 계수 방지.

선: (0.1, 0.6) -> (0.9, 0.6) 수평선.
부호: y > 0.6 이면 POSITIVE(화면 아래), y < 0.6 이면 NEGATIVE(화면 위).
in_direction = NEGATIVE 이면 아래에서 위로 넘어가는 것이 입장이다.
"""

from __future__ import annotations

import pytest

from cctv_counting.crossing import (
    CrossingError,
    CrossingEvent,
    InDirection,
    LineCrossingCounter,
    count_line_band_detections,
)
from cctv_counting.geometry import Line, NormalizedPoint

LINE = Line(NormalizedPoint(0.1, 0.6), NormalizedPoint(0.9, 0.6))


def _counter(**overrides) -> LineCrossingCounter:
    defaults = dict(
        line=LINE,
        in_direction=InDirection.NEGATIVE,
        band=0.01,
        min_confirm_distance=0.03,
        cooldown_seconds=10.0,
    )
    defaults.update(overrides)
    return LineCrossingCounter(**defaults)


def _walk(
    counter: LineCrossingCounter,
    ys: list[float],
    track_id: int = 1,
    fps: float = 1.0,
) -> list:
    """프레임 번호와 시각을 함께 넣는다. 기본은 1fps(프레임당 1초)라
    프레임 수와 초가 같아 읽기 쉽다."""
    events = []
    for frame, y in enumerate(ys):
        events.extend(
            counter.update(frame, frame / fps, {track_id: NormalizedPoint(0.5, y)})
        )
    return events


def test_walking_from_below_to_above_is_an_entry():
    counter = _counter()
    events = _walk(counter, [0.80, 0.70, 0.62, 0.55, 0.45])
    assert [event.event for event in events] == [CrossingEvent.ENTRY]
    assert counter.entries == 1
    assert counter.exits == 0


def test_walking_from_above_to_below_is_an_exit():
    counter = _counter()
    events = _walk(counter, [0.40, 0.50, 0.58, 0.65, 0.80])
    assert [event.event for event in events] == [CrossingEvent.EXIT]
    assert counter.exits == 1
    assert counter.entries == 0


def test_flipping_in_direction_flips_the_meaning():
    """카메라를 반대편에 달면 IN과 OUT이 뒤집힌다. 코드가 아니라 설정이 정한다."""
    counter = _counter(in_direction=InDirection.POSITIVE)
    events = _walk(counter, [0.80, 0.70, 0.62, 0.55, 0.45])
    assert [event.event for event in events] == [CrossingEvent.EXIT]


def test_walking_up_to_the_line_without_crossing_counts_nothing():
    counter = _counter()
    events = _walk(counter, [0.85, 0.80, 0.75, 0.70, 0.68])
    assert events == []
    assert counter.entries == counter.exits == 0


def test_standing_exactly_on_the_line_counts_nothing():
    counter = _counter()
    events = _walk(counter, [0.60, 0.60, 0.60, 0.60])
    assert events == []


def test_touching_the_line_and_going_back_counts_nothing():
    """선을 밟았다가 되돌아가는 것은 통과가 아니다."""
    counter = _counter()
    events = _walk(counter, [0.80, 0.70, 0.60, 0.70, 0.80])
    assert events == []


def test_jitter_on_the_line_does_not_produce_repeated_events():
    """같은 사람이 선 위에서 떨린다고 IN, IN, IN이 되면 이용 인원이 폭발한다."""
    counter = _counter()
    ys = [0.80, 0.70] + [0.601, 0.599, 0.602, 0.598, 0.6005, 0.5995] * 4 + [0.50]
    events = _walk(counter, ys)
    assert len(events) == 1
    assert events[0].event is CrossingEvent.ENTRY


def test_shallow_oscillation_past_the_band_is_still_ignored():
    """band는 벗어나도 확정 거리에 못 미치면 확정하지 않는다."""
    counter = _counter(band=0.005, min_confirm_distance=0.05)
    events = _walk(counter, [0.62, 0.58, 0.62, 0.58, 0.62, 0.58])
    assert events == []


def test_cooldown_suppresses_an_immediate_bounce_back():
    counter = _counter(cooldown_seconds=20.0)
    events = _walk(counter, [0.80, 0.50, 0.80, 0.50])
    assert len(events) == 1
    assert events[0].event is CrossingEvent.ENTRY


def test_genuine_re_entry_after_cooldown_counts_twice():
    """실제로 나갔다 다시 들어온 사람은 두 번 온 것이다."""
    counter = _counter(cooldown_seconds=3.0)
    ys = [0.80] * 3 + [0.45] * 5 + [0.80] * 5 + [0.45] * 5
    events = _walk(counter, ys)
    kinds = [event.event for event in events]
    assert kinds == [CrossingEvent.ENTRY, CrossingEvent.EXIT, CrossingEvent.ENTRY]
    assert counter.entries == 2
    assert counter.exits == 1


def test_person_already_inside_at_video_start_is_not_an_entry():
    """영상이 시작될 때 이미 안에 있던 사람을 입장으로 세면 안 된다."""
    counter = _counter()
    events = _walk(counter, [0.40, 0.35, 0.30, 0.25])
    assert events == []
    assert counter.entries == 0


def test_two_tracks_are_counted_independently():
    counter = _counter()
    for frame, (a, b) in enumerate(
        zip([0.80, 0.70, 0.55, 0.45], [0.40, 0.50, 0.68, 0.80])
    ):
        counter.update(
            frame,
            float(frame),
            {1: NormalizedPoint(0.3, a), 2: NormalizedPoint(0.7, b)},
        )
    assert counter.entries == 1
    assert counter.exits == 1


def test_track_state_survives_a_missing_frame():
    """잠깐 가려졌다 다시 보인 사람은 이어서 판단해야 한다."""
    counter = _counter()
    counter.update(0, 0.0, {1: NormalizedPoint(0.5, 0.80)})
    counter.update(1, 1.0, {})
    counter.update(2, 2.0, {})
    events = counter.update(3, 3.0, {1: NormalizedPoint(0.5, 0.45)})
    assert [event.event for event in events] == [CrossingEvent.ENTRY]


def test_forget_removes_state_of_dead_tracks():
    """끝난 track의 상태를 들고 있으면 번호가 재사용될 때 엉뚱하게 시작한다."""
    counter = _counter()
    counter.update(
        0, 0.0, {1: NormalizedPoint(0.5, 0.80), 2: NormalizedPoint(0.5, 0.20)}
    )
    assert counter.tracked_state_count == 2
    counter.forget({1})
    assert counter.tracked_state_count == 1


def test_reset_clears_totals_and_state():
    counter = _counter()
    _walk(counter, [0.80, 0.45])
    counter.reset()
    assert counter.entries == 0
    assert counter.exits == 0
    assert counter.tracked_state_count == 0


def test_events_carry_the_frame_they_happened_on():
    counter = _counter()
    events = _walk(counter, [0.80, 0.70, 0.45])
    assert events[0].frame_index == 2


def test_min_confirm_distance_must_not_be_smaller_than_band():
    with pytest.raises(CrossingError):
        _counter(band=0.05, min_confirm_distance=0.01)


def test_negative_cooldown_is_rejected():
    with pytest.raises(CrossingError):
        _counter(cooldown_seconds=-1.0)


def test_non_finite_cooldown_is_rejected():
    with pytest.raises(CrossingError):
        _counter(cooldown_seconds=float("nan"))


def test_non_finite_timestamp_is_rejected():
    counter = _counter()
    with pytest.raises(CrossingError):
        counter.update(0, float("inf"), {1: NormalizedPoint(0.5, 0.8)})


def test_naive_line_band_proxy_counts_every_frame_near_the_line():
    """추적 없이 세면 한 사람이 지나가는 동안 여러 번 세인다. 그게 요점이다."""
    points = [NormalizedPoint(0.5, 0.6005), NormalizedPoint(0.5, 0.9)]
    assert count_line_band_detections(LINE, points, 0.01) == 1
    assert count_line_band_detections(LINE, [], 0.01) == 0
