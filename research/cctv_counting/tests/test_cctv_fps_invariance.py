"""같은 궤적은 몇 fps로 샘플링하든 같은 수가 나와야 한다.

이 파일이 지키는 것은 하나다. **계수 정책이 시간에 걸려 있지 프레임 수에
걸려 있지 않다는 것.**

cooldown을 프레임 수로 두면 같은 설정 파일이 영상마다 다른 정책이 된다.
15프레임은 30fps에서 0.5초지만 10fps에서는 1.5초, 60fps에서는 0.25초다.
카메라를 바꾸거나 frame_stride를 조정하는 것만으로 입장 인원이 달라지는데,
그 변화는 계수 규칙이 바뀐 탓이지 사람이 다르게 다닌 탓이 아니다. 그런
결과는 비교할 수 없다.

여기서는 **물리적으로 동일한 궤적**을 여러 fps로 샘플링해 같은 답이 나오는지
본다. 사람은 같은 속도로 같은 경로를 걷는다. 달라지는 것은 그것을 몇 번
들여다보느냐뿐이다.
"""

from __future__ import annotations

import pytest

from cctv_counting.crossing import (
    CrossingEvent,
    InDirection,
    LineCrossingCounter,
)
from cctv_counting.geometry import Line, NormalizedPoint

LINE = Line(NormalizedPoint(0.0, 0.6), NormalizedPoint(1.0, 0.6))

#: 세 가지 샘플링 속도. 실제 CCTV가 이 범위 안에 있다.
SAMPLING_RATES = (10.0, 30.0, 60.0)


def _counter(cooldown_seconds: float = 0.5) -> LineCrossingCounter:
    return LineCrossingCounter(
        line=LINE,
        in_direction=InDirection.NEGATIVE,
        band=0.01,
        min_confirm_distance=0.03,
        cooldown_seconds=cooldown_seconds,
    )


def _sample(trajectory, duration_seconds: float, fps: float):
    """연속 궤적 y(t)를 fps로 샘플링한다.

    trajectory는 시각(초)을 받아 정규화된 y를 돌려주는 함수다. 같은 함수를
    다른 fps로 훑으므로, 들어가는 물리적 움직임은 정확히 같다.
    """
    frame_count = int(round(duration_seconds * fps))
    for index in range(frame_count + 1):
        timestamp = index / fps
        yield index, timestamp, trajectory(timestamp)


def _run(trajectory, duration_seconds: float, fps: float, cooldown_seconds=0.5):
    counter = _counter(cooldown_seconds)
    events = []
    for index, timestamp, y in _sample(trajectory, duration_seconds, fps):
        events.extend(
            counter.update(index, timestamp, {1: NormalizedPoint(0.5, y)})
        )
    return counter, events


def _walk_in(t: float) -> float:
    """4초에 걸쳐 아래(0.9)에서 위(0.2)로 곧게 걸어 들어간다."""
    return max(0.2, 0.9 - (0.7 / 4.0) * t)


def _jitter_on_the_line(t: float) -> float:
    """선까지 걸어와 2초간 선 위에서 떨다가 마저 들어간다.

    떨림 폭은 ±0.02다. 불감대(0.01)는 벗어나지만 확정 거리(0.03)에는 못
    미치므로, 이 떨림을 막는 것은 이력(hysteresis) 층이다. 떨림은 시간의
    함수라 물리적으로는 fps와 무관하지만, fps가 높을수록 더 여러 번
    관측된다. 프레임 수를 세는 정책이라면 여기서 값이 갈린다.
    """
    if t < 1.0:
        return 0.9 - 0.29 * t          # 0.90 -> 0.61
    if t < 3.0:
        return 0.600 + 0.02 * (1 if int(t * 7) % 2 else -1)
    return max(0.2, 0.60 - 0.4 * (t - 3.0))


def _oscillating_past_the_confirm_distance(t: float) -> float:
    """0.4초마다 확정 거리 밖까지 왔다 갔다 한다.

    이력으로는 막히지 않는다. 진폭이 확정 거리(0.03)를 넘기 때문이다. 여기서
    중복을 막는 것은 재계수 금지 구간(cooldown)이고, 그 정책이 시간 단위이므로
    사건 수는 관측 빈도가 아니라 흐른 시간으로 정해진다.

    0.4초 주기는 10 / 30 / 60 fps 모두에서 정확히 샘플링된다.
    """
    return 0.55 if int(t / 0.4) % 2 == 0 else 0.65


def _in_out_in(t: float) -> float:
    """들어갔다(2초) 나왔다(2초) 다시 들어간다(2초). 실제 재방문 세 구간."""
    if t < 2.0:
        return 0.9 - 0.35 * t          # 0.90 -> 0.20
    if t < 4.0:
        return 0.2 + 0.35 * (t - 2.0)  # 0.20 -> 0.90
    return max(0.2, 0.9 - 0.35 * (t - 4.0))


@pytest.mark.parametrize("fps", SAMPLING_RATES)
def test_one_entry_at_every_sampling_rate(fps):
    """10, 30, 60 fps 어디서든 한 번 들어간 것은 한 번이다."""
    counter, events = _run(_walk_in, 4.0, fps)
    assert counter.entries == 1
    assert counter.exits == 0
    assert [event.event for event in events] == [CrossingEvent.ENTRY]


def test_entry_count_is_identical_across_sampling_rates():
    """세 결과를 나란히 놓고 같은지 본다."""
    totals = {
        fps: _run(_walk_in, 4.0, fps)[0].entries for fps in SAMPLING_RATES
    }
    assert len(set(totals.values())) == 1, totals


@pytest.mark.parametrize("fps", SAMPLING_RATES)
def test_jitter_does_not_duplicate_at_any_sampling_rate(fps):
    """fps가 높을수록 떨림을 더 여러 번 본다. 그래도 한 번이어야 한다."""
    counter, events = _run(_jitter_on_the_line, 5.0, fps)
    assert counter.entries == 1, [e.event for e in events]
    assert counter.exits == 0


def test_jitter_result_does_not_depend_on_sampling_rate():
    results = {
        fps: _run(_jitter_on_the_line, 5.0, fps)[0].entries
        for fps in SAMPLING_RATES
    }
    assert len(set(results.values())) == 1, results


@pytest.mark.parametrize("fps", SAMPLING_RATES)
def test_in_out_in_is_two_visits_at_any_sampling_rate(fps):
    """실제 재방문은 fps와 무관하게 두 번이다."""
    counter, events = _run(_in_out_in, 6.0, fps, cooldown_seconds=0.5)
    assert [event.event for event in events] == [
        CrossingEvent.ENTRY,
        CrossingEvent.EXIT,
        CrossingEvent.ENTRY,
    ]
    assert counter.entries == 2
    assert counter.exits == 1


def test_in_out_in_totals_match_across_sampling_rates():
    results = {
        fps: (
            lambda c: (c.entries, c.exits)
        )(_run(_in_out_in, 6.0, fps, cooldown_seconds=0.5)[0])
        for fps in SAMPLING_RATES
    }
    assert len(set(results.values())) == 1, results


def test_cooldown_is_measured_in_seconds_not_frames():
    """같은 fps에서 cooldown만 늘리면 실제 재방문이 억제된다.

    cooldown이 시간 단위임을 보이는 대조군이다. 6초 왕복을 10초 cooldown으로
    보면 두 번째 입장이 눌린다.
    """
    short, _ = _run(_in_out_in, 6.0, 30.0, cooldown_seconds=0.5)
    long, _ = _run(_in_out_in, 6.0, 30.0, cooldown_seconds=10.0)
    assert short.entries == 2
    assert long.entries == 1


def test_the_same_cooldown_means_the_same_duration_at_every_rate():
    """cooldown 10초는 10fps에서도 60fps에서도 10초다."""
    suppressed = {
        fps: _run(_in_out_in, 6.0, fps, cooldown_seconds=10.0)[0].entries
        for fps in SAMPLING_RATES
    }
    assert set(suppressed.values()) == {1}, suppressed


def test_frame_stride_does_not_change_the_count():
    """매 프레임을 보지 않고 건너뛰어도 결과가 같아야 한다.

    frame_stride는 속도를 위한 설정이지 계수 정책이 아니다. 이것이 값을
    바꾸면 빠르게 돌린 실험과 느리게 돌린 실험을 비교할 수 없다.
    """
    every_frame, _ = _run(_walk_in, 4.0, 60.0)

    counter = _counter()
    for index, timestamp, y in _sample(_walk_in, 4.0, 60.0):
        if index % 3:  # 3프레임에 한 장만 처리한다
            continue
        counter.update(index, timestamp, {1: NormalizedPoint(0.5, y)})

    assert counter.entries == every_frame.entries == 1


def test_events_report_the_time_they_happened():
    """사건의 시각이 fps에 관계없이 실제 통과 시점 근처여야 한다."""
    for fps in SAMPLING_RATES:
        _, events = _run(_walk_in, 4.0, fps)
        # 0.9 -> 0.2 를 4초에 가므로 확정 거리(0.03)를 넘는 것은 대략 1.9초.
        assert events[0].timestamp_seconds == pytest.approx(1.9, abs=0.2)


@pytest.mark.parametrize("fps", SAMPLING_RATES)
def test_rapid_oscillation_is_capped_by_elapsed_time_not_frame_count(fps):
    """확정 거리 밖까지 흔들리는 경우, 사건 수는 시간이 정한다.

    0.4초마다 방향이 바뀌고 cooldown이 1초이면, 몇 fps로 보든 사건은 같은
    시각에 같은 수만큼 난다. 프레임 수를 세는 정책이라면 60fps에서 10fps의
    여섯 배를 보게 되고, 그 순간 두 실험은 비교할 수 없게 된다.
    """
    counter, events = _run(
        _oscillating_past_the_confirm_distance, 4.0, fps, cooldown_seconds=1.0
    )
    assert (counter.entries, counter.exits) == (2, 2), [
        (round(e.timestamp_seconds, 3), e.event.value) for e in events
    ]


def test_rapid_oscillation_gives_the_same_totals_at_every_rate():
    results = {
        fps: (
            lambda c: (c.entries, c.exits)
        )(_run(_oscillating_past_the_confirm_distance, 4.0, fps, cooldown_seconds=1.0)[0])
        for fps in SAMPLING_RATES
    }
    assert len(set(results.values())) == 1, results


def test_rapid_oscillation_event_times_agree_within_one_sample():
    """사건의 종류와 순서는 같고, 시각은 한 샘플 간격 안에서 일치한다.

    시각까지 정확히 같기를 요구하면 안 된다. 사건은 실제로 관측한 프레임에서만
    날 수 있으므로, 시간 해상도는 샘플링 간격보다 좋아질 수 없다. cooldown이
    풀리는 순간이 두 프레임 사이에 떨어지면, 10fps는 최대 0.1초 늦게, 60fps는
    최대 0.017초 늦게 그것을 알아챈다.

    이것은 프레임 의존성이 아니라 관측의 한계다. 구분해서 검사한다. **개수와
    순서는 완전히 같아야 하고, 시각은 가장 성긴 샘플링 간격 안이어야 한다.**
    """
    timings = {}
    for fps in SAMPLING_RATES:
        _, events = _run(
            _oscillating_past_the_confirm_distance, 4.0, fps, cooldown_seconds=1.0
        )
        timings[fps] = events

    kinds = {
        fps: tuple(event.event.value for event in events)
        for fps, events in timings.items()
    }
    assert len(set(kinds.values())) == 1, kinds

    coarsest_interval = 1.0 / min(SAMPLING_RATES)
    reference = timings[max(SAMPLING_RATES)]
    for fps, events in timings.items():
        for observed, expected in zip(events, reference):
            assert observed.timestamp_seconds == pytest.approx(
                expected.timestamp_seconds, abs=coarsest_interval
            ), (fps, observed, expected)


def test_observation_counts_really_do_differ_across_rates():
    """앞의 테스트들이 공허하지 않다는 확인.

    같은 4초를 10fps와 60fps로 보면 관측 수는 여섯 배 차이가 난다. 그런데도
    계수 결과가 같다는 것이 이 파일이 주장하는 바다. 관측 수마저 같았다면
    아무것도 증명하지 못한 것이다.
    """
    sampled = {
        fps: len(list(_sample(_walk_in, 4.0, fps))) for fps in SAMPLING_RATES
    }
    assert sampled[60.0] > sampled[30.0] > sampled[10.0]
