"""짧은 시간 창으로 묶어 보기.

프레임 하나의 occupancy는 생각보다 많이 흔들린다. 사람이 잠깐 가려지고,
detector가 한 프레임 놓치고, track이 min_hits를 채우는 데 몇 프레임 걸린다.
"지금 혼잡한가"라는 질문에 대해서는 1~5초 창의 통계가 프레임 하나보다
안정적일 가능성이 높다.

**그렇다고 창 통계를 기본 결과로 두지는 않는다.** 평활한 곡선은 언제나 더
그럴듯해 보인다. 원본 관측을 감추고 평활값만 보여 주면, 실제로 얼마나
흔들리는지가 보고서에서 사라진다. 그래서 기본 출력은 프레임별 원본이고,
여기 있는 것은 그 위에 얹는 부가 계산이다.

창 통계가 정말 프레임 단위보다 나은지는 정답 데이터로 재 봐야 안다. 아직
재지 않았다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cctv_counting.observations import FrameObservation


class WindowError(ValueError):
    """창 설정이 성립하지 않는다."""


@dataclass(frozen=True)
class WindowSummary:
    """시간 창 하나의 요약."""

    start_seconds: float
    end_seconds: float
    frame_count: int
    mean_occupancy: float
    median_occupancy: float
    max_occupancy: int
    #: 이 창 안에서 늘어난 누적 입장 수. 추적이 없으면 None.
    entries_in_window: int | None
    #: 낮은 신뢰 프레임의 비율.
    low_reliability_ratio: float


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def aggregate_windows(
    observations: list[FrameObservation],
    window_seconds: float,
) -> list[WindowSummary]:
    """관측을 고정 길이 창으로 묶는다.

    창은 첫 관측의 시각이 아니라 0초를 기준으로 자른다. 같은 영상을 어디서
    끊어 읽어도 창 경계가 같아야 비교할 수 있다.

    관측이 없는 창은 결과에 넣지 않는다. 빈 창에 occupancy 0을 채우면 영상이
    끊긴 구간이 "아무도 없던 구간"으로 둔갑한다.
    """
    if math.isnan(window_seconds) or math.isinf(window_seconds) or window_seconds <= 0:
        raise WindowError(f"window_seconds는 0보다 커야 합니다: {window_seconds}")
    if not observations:
        return []

    buckets: dict[int, list[FrameObservation]] = {}
    for observation in observations:
        index = int(observation.timestamp_seconds // window_seconds)
        buckets.setdefault(index, []).append(observation)

    summaries: list[WindowSummary] = []

    # 직전 창이 끝난 시점의 누적 입장 수. 창 안에서 늘어난 양은 이것과의
    # 차이다. 첫 창은 직전이 없으므로 그 창 첫 관측의 누적값을 기준으로
    # 삼는다. 영상 중간부터 읽은 경우 시작 시점에 이미 쌓여 있던 인원을
    # 첫 창의 몫으로 돌리지 않으려는 것이다.
    previous_total: int | None = None

    for index in sorted(buckets):
        group = sorted(buckets[index], key=lambda item: item.frame_index)
        counts = [item.occupancy_count for item in group]

        entries: int | None = None
        last_entries = group[-1].entries_total
        if last_entries is not None:
            baseline = previous_total
            if baseline is None:
                baseline = group[0].entries_total or 0
            entries = max(0, last_entries - baseline)
            previous_total = last_entries

        low = sum(1 for item in group if item.reliability_flag.value != "OK")
        summaries.append(
            WindowSummary(
                start_seconds=index * window_seconds,
                end_seconds=(index + 1) * window_seconds,
                frame_count=len(group),
                mean_occupancy=sum(counts) / len(counts),
                median_occupancy=_median(counts),
                max_occupancy=max(counts),
                entries_in_window=entries,
                low_reliability_ratio=low / len(group),
            )
        )
    return summaries
