"""가상 선 통과 판정.

입구에 선을 하나 긋고, 그 선을 IN 방향으로 넘은 사람을 센다. 이 값이 한
끼 실제 이용 인원의 대리 지표다.

**어느 쪽이 IN인가는 코드가 정하지 않는다.** 카메라를 반대편에 달면 IN과
OUT이 뒤집힌다. 그래서 방향은 config에서 온다. 선을 start -> end 로 그으면
그 진행 방향 기준으로 왼쪽이 POSITIVE, 오른쪽이 NEGATIVE다(화면 좌표계는
y축이 아래로 향하므로 눈으로 볼 때는 반대로 보인다). config의 in_direction은
"어느 쪽으로 **들어가는** 것을 입장으로 볼 것인가"를 말한다::

    line.start = [0.10, 0.60]
    line.end   = [0.90, 0.60]
    in_direction = NEGATIVE   # 화면 위쪽(y가 작아지는 쪽)으로 넘어가면 입장

**중복 계수를 막는 세 겹.**

선 위에서 사람이 조금만 떨려도 IN, OUT, IN이 연달아 찍힌다. 실제로는 한 번
들어간 것이다. 반대로 진짜 다시 들어온 사람은 두 번 세야 한다. 이 둘을
가르는 것이 여기의 핵심이고, 세 가지를 쌓아 둔다.

1. **불감대(band).** 선에서 band보다 가까우면 ON_LINE으로 보고 확정된
   쪽을 바꾸지 않는다. 선 바로 위의 미세한 떨림을 통째로 무시한다.
2. **이력(hysteresis).** band를 벗어났다고 바로 확정하지 않는다.
   min_confirm_distance 이상 떨어져야 그쪽에 있다고 확정한다. band보다
   크게 잡아, 넘어갔다 되돌아오는 얕은 진동을 걸러낸다.
3. **재계수 금지 구간(cooldown).** 한 번 센 track은 cooldown_frames 동안
   다시 세지 않는다. 그 시간이 지난 뒤의 IN -> OUT -> IN은 실제 재방문으로
   보고 두 번 센다.

track이 사라지면 그 상태도 함께 지운다. 상태를 영상 밖으로 들고 나가지
않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from cctv_counting.geometry import Line, NormalizedPoint, Side


class CrossingError(ValueError):
    """통과 판정 설정이 성립하지 않는다."""


class InDirection(str, Enum):
    """선의 어느 쪽으로 넘어가는 것을 입장으로 볼 것인가."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class CrossingEvent(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"


@dataclass(frozen=True)
class Crossing:
    """통과 사건 하나.

    track_id는 영상 세션 안에서만 의미가 있는 임시 번호다. 이 사건을 밖으로
    내보낼 때는 번호가 아니라 집계된 개수만 나간다.
    """

    track_id: int
    frame_index: int
    event: CrossingEvent


@dataclass
class _TrackCrossingState:
    """track 하나의 통과 상태."""

    #: 마지막으로 확정된 쪽. 아직 확정된 적 없으면 None.
    confirmed_side: Side | None = None

    #: 마지막으로 사건을 낸 프레임. 없으면 None.
    last_event_frame: int | None = None

    #: 마지막으로 관측된 프레임. 오래된 상태를 치우는 데 쓴다.
    last_seen_frame: int = -1


@dataclass
class LineCrossingCounter:
    """선 통과를 세는 상태 기계.

    band:                 이 거리 안은 ON_LINE. 정규화 단위다.
    min_confirm_distance: 이 거리 이상 떨어져야 그쪽에 있다고 확정한다.
    cooldown_frames:      사건을 낸 뒤 이만큼은 같은 track을 다시 세지 않는다.
    """

    line: Line
    in_direction: InDirection = InDirection.NEGATIVE
    band: float = 0.01
    min_confirm_distance: float = 0.03
    cooldown_frames: int = 15

    entries: int = field(default=0, init=False)
    exits: int = field(default=0, init=False)
    _states: dict[int, _TrackCrossingState] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.band < 0.0:
            raise CrossingError(f"band는 0 이상이어야 합니다: {self.band}")
        if self.min_confirm_distance < self.band:
            raise CrossingError(
                "min_confirm_distance는 band 이상이어야 합니다: "
                f"{self.min_confirm_distance} < {self.band}"
            )
        if self.cooldown_frames < 0:
            raise CrossingError(
                f"cooldown_frames는 0 이상이어야 합니다: {self.cooldown_frames}"
            )

    def reset(self) -> None:
        """영상이 바뀌면 부른다."""
        self.entries = 0
        self.exits = 0
        self._states.clear()

    @property
    def tracked_state_count(self) -> int:
        """상태를 들고 있는 track 수. 상태가 새는지 보는 용도다."""
        return len(self._states)

    def _event_for(self, previous: Side, current: Side) -> CrossingEvent:
        """어느 쪽에서 어느 쪽으로 갔는가를 입장/퇴장으로 번역한다."""
        entering_side = (
            Side.POSITIVE
            if self.in_direction is InDirection.POSITIVE
            else Side.NEGATIVE
        )
        if current is entering_side:
            return CrossingEvent.ENTRY
        return CrossingEvent.EXIT

    def update(
        self,
        frame_index: int,
        positions: dict[int, NormalizedPoint],
    ) -> list[Crossing]:
        """한 프레임의 track별 위치를 넣고, 이번에 생긴 사건들을 받는다.

        positions는 {track_id: 정규화된 대표점}이다. 이 프레임에 없는
        track은 그냥 빠져 있으면 된다. 잠깐 가려진 사람이 다시 나타나면
        확정된 쪽이 그대로 남아 있어서 이어서 판단한다.
        """
        events: list[Crossing] = []

        for track_id, point in sorted(positions.items()):
            state = self._states.get(track_id)
            if state is None:
                state = _TrackCrossingState()
                self._states[track_id] = state
            state.last_seen_frame = frame_index

            distance = self.line.signed_distance_normalized(point)
            side = self.line.side_of(point, band=self.band)

            if side is Side.ON_LINE:
                # 선 위다. 확정된 쪽을 건드리지 않는다.
                continue
            if abs(distance) < self.min_confirm_distance:
                # band는 벗어났지만 아직 확정하기에는 얕다.
                continue

            previous = state.confirmed_side
            if previous is None:
                # 처음 확정되는 쪽이다. 사건이 아니다. 영상 시작 시점에 이미
                # 안쪽에 있던 사람을 입장으로 세면 안 된다.
                state.confirmed_side = side
                continue
            if previous is side:
                continue

            state.confirmed_side = side

            if (
                state.last_event_frame is not None
                and frame_index - state.last_event_frame < self.cooldown_frames
            ):
                # 방금 셌다. 같은 왕복의 되돌아옴으로 본다.
                continue

            event = self._event_for(previous, side)
            state.last_event_frame = frame_index
            if event is CrossingEvent.ENTRY:
                self.entries += 1
            else:
                self.exits += 1
            events.append(Crossing(track_id, frame_index, event))

        return events

    def forget(self, track_ids: set[int]) -> None:
        """사라진 track의 상태를 지운다.

        추적이 끝난 번호를 계속 들고 있으면, 나중에 같은 번호가 다른
        사람에게 재사용될 때 엉뚱한 쪽에서 시작한다. 그리고 영상이 길어질수록
        메모리에 사람 궤적 정보가 쌓인다. 둘 다 원하지 않는다.
        """
        for track_id in list(self._states):
            if track_id not in track_ids:
                del self._states[track_id]


def count_line_band_detections(
    line: Line,
    points: list[NormalizedPoint],
    band: float,
) -> int:
    """선 근처에 있는 점의 개수.

    추적 없이 detection만으로 입장 인원을 세려는 순진한 대리 지표다. 같은
    사람이 선을 지나는 동안 매 프레임 다시 세이므로 실제보다 크게 나온다.
    쓰라고 두는 것이 아니라, 추적이 무엇을 해 주는지 숫자로 보여 주려고
    둔다. baseline A의 비교 대상이다.
    """
    if band < 0.0:
        raise CrossingError(f"band는 0 이상이어야 합니다: {band}")
    return sum(
        1 for point in points if line.side_of(point, band=band) is Side.ON_LINE
    )
