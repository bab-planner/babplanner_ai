"""놓친 통과 사건의 원인 분류.

**왜 필요한가.** A5는 "통과 누락의 절반 이상은 검출 문제가 아니다"까지
좁혔다. 그 다음 질문은 "그러면 무엇인가"이고, 그것을 답하려면 놓친 사건
하나하나에 원인을 붙일 수 있어야 한다. 원인이 다르면 고칠 곳도 다르다.

    DETECTION_GAP       detector가 그 사람을 못 봤다        -> detector 문제
    ASSOCIATION_BREAK   상자는 있는데 track에 못 붙였다     -> 연결 문제
    TRACK_EXPIRY        끊긴 시간이 max_age를 넘었다        -> 소멸 정책 문제
    ID_SWITCH           번호가 바뀌었다                     -> 연결/식별 문제
    LINE_GEOMETRY       끊김 없이 이어졌는데도 사건이 없다  -> 선 판정 규칙 문제
    OTHER               위 어디에도 해당하지 않는다

**규칙은 결과를 보기 전에 고정한다.** 아래 순서대로 처음 걸리는 것 하나를
붙인다. 사후에 순서를 바꾸면 어떤 이야기든 만들 수 있다.

**자동 분류를 그대로 결론으로 쓰지 않는다.** 이것은 41건 x 여러 조건을
사람이 전부 볼 수 없어서 두는 1차 분류다. 대표 사건은 프레임 단위 추적
기록을 직접 읽어 확인하고, 그 확인 결과를 보고서에 적는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cctv_counting.continuity import CrossingWindow, Gap


class FailureCause(str, Enum):
    DETECTION_GAP = "DETECTION_GAP"
    ASSOCIATION_BREAK = "ASSOCIATION_BREAK"
    TRACK_EXPIRY = "TRACK_EXPIRY"
    ID_SWITCH = "ID_SWITCH"
    LINE_GEOMETRY = "LINE_GEOMETRY"
    OTHER = "OTHER"


#: 이 아래로 덮이면 "사실상 없었다"로 본다. 결과를 보기 전에 고정했다.
#: 절반이라는 값에 특별한 근거는 없다. 다만 사후에 바꾸지 않는다.
COVERAGE_FLOOR = 0.5


@dataclass(frozen=True)
class Attribution:
    """놓친 사건 하나에 붙는 원인과 그 근거."""

    cause: FailureCause
    detection_coverage: float | None
    track_coverage: float | None
    dominant_track_share: float | None
    distinct_track_ids: int
    longest_gap_seconds: float
    spans_both_sides: bool

    def as_dict(self) -> dict:
        return {
            "cause": self.cause.value,
            "detectionCoverage": self.detection_coverage,
            "trackCoverage": self.track_coverage,
            "dominantTrackShare": self.dominant_track_share,
            "distinctTrackIds": self.distinct_track_ids,
            "longestGapSeconds": self.longest_gap_seconds,
            "spansBothSides": self.spans_both_sides,
        }


def attribute(
    window: CrossingWindow,
    gaps_in_window: list[Gap],
    *,
    max_age_seconds: float,
) -> Attribution:
    """놓친 통과 사건 하나의 원인을 고른다.

    gaps_in_window는 그 사람의 끊김 중 이 사건 창과 겹치는 것들이다.
    max_age_seconds는 그 실행에 쓴 tracker의 소멸 기준이다. 끊긴 시간이
    그보다 길면 tracker가 규칙상 지울 수밖에 없었다는 뜻이므로, 연결
    실패가 아니라 소멸 정책의 문제로 분류한다.
    """
    longest = max((gap.seconds for gap in gaps_in_window), default=0.0)
    switched = any(not gap.recovered for gap in gaps_in_window)

    if window.window_frames == 0:
        cause = FailureCause.OTHER
    elif (window.detection_coverage or 0.0) < COVERAGE_FLOOR:
        cause = FailureCause.DETECTION_GAP
    elif (window.track_coverage or 0.0) < COVERAGE_FLOOR:
        cause = FailureCause.ASSOCIATION_BREAK
    elif longest > max_age_seconds and switched:
        cause = FailureCause.TRACK_EXPIRY
    elif window.distinct_track_ids >= 2:
        cause = FailureCause.ID_SWITCH
    elif window.distinct_track_ids == 1 and window.spans_both_sides:
        cause = FailureCause.LINE_GEOMETRY
    else:
        cause = FailureCause.OTHER

    return Attribution(
        cause=cause,
        detection_coverage=window.detection_coverage,
        track_coverage=window.track_coverage,
        dominant_track_share=window.dominant_track_share,
        distinct_track_ids=window.distinct_track_ids,
        longest_gap_seconds=longest,
        spans_both_sides=window.spans_both_sides,
    )


def tally(attributions: list[Attribution]) -> dict[str, int]:
    """원인별 건수. 보고서 표에 그대로 들어간다."""
    counts = {cause.value: 0 for cause in FailureCause}
    for item in attributions:
        counts[item.cause.value] += 1
    return counts
