"""Multi-object tracking.

**왜 추적이 필요한가.** 프레임마다 사람 수를 세는 것만으로는 "지금 몇 명
있는가"까지밖에 못 간다. "몇 명이 들어왔는가"는 같은 사람을 프레임 사이에서
같은 사람으로 이어 붙여야만 셀 수 있다. 이 둘은 다른 질문이고, 추적은 두
번째 질문의 전제다.

**왜 ByteTrack 계열인가.**

- 외형 특징(re-id embedding)을 쓰지 않는다. 사람을 식별하지 않겠다는 이
  프로젝트의 제약과 정확히 맞는다. 상자의 위치만으로 잇는다.
- 낮은 신뢰도 상자를 버리지 않고 2단계로 다시 매칭한다. 급식소처럼 사람이
  겹치는 장면에서 가려진 사람의 신뢰도가 잠깐 떨어졌다고 track이 끊기면,
  같은 사람이 새 id를 받고 입장 인원이 부풀려진다.
- detector 출력만 있으면 되고 별도 학습이 필요 없다. baseline으로 적당하다.
- 실시간에 가깝게 돈다. 서비스에 붙일 것이므로 중요하다.

원 논문: Zhang et al., ByteTrack: Multi-Object Tracking by Associating Every
Detection Box, ECCV 2022. 공식 구현(ifzhang/ByteTrack)은 MIT다.

**원 구현과 다른 점.** 여기 있는 것은 공식 구현을 옮긴 것이 아니라 BYTE의
연결 규칙을 이 저장소 안에서 다시 쓴 것이다. 의도적으로 두 군데를 단순화했다.

1. Kalman filter 대신 등속 외삽을 쓴다. 직전 두 프레임의 중심 이동량을
   그대로 다음 프레임에 더한다.
2. Hungarian 최적 할당 대신 IoU 내림차순 greedy 매칭을 쓴다.

둘 다 numpy/scipy 없이 표준 라이브러리만으로 돌리기 위한 선택이고, core
로직을 무거운 의존성 없이 그대로 단위 테스트할 수 있게 하는 것이 목적이다.
그래서 여기서 나온 수치를 ByteTrack 논문 성능으로 읽으면 안 된다. 대가는
reports/cctv_baseline.md 의 한계 절에 적어 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from cctv_counting.detection import Detection
from cctv_counting.geometry import BoundingBox


class TrackingError(ValueError):
    """추적 설정이 성립하지 않는다."""


class TrackState(str, Enum):
    """track이 지금 어떤 상태인가."""

    #: 막 생겼고 아직 믿을 수 없다. occupancy에 넣지 않는다.
    TENTATIVE = "TENTATIVE"

    #: 연속으로 충분히 맞았다. 세는 대상이다.
    CONFIRMED = "CONFIRMED"

    #: 이번 프레임에서 못 찾았다. 아직 지우지는 않는다.
    LOST = "LOST"


@dataclass(frozen=True)
class TrackedPerson:
    """한 프레임에서 추적된 사람 하나.

    track_id는 **이 영상 한 세션 안에서만 의미가 있는 임시 번호**다. 다른
    영상, 다른 날, 다른 카메라의 같은 번호는 같은 사람이 아니다. 사람을
    가리키는 식별자가 아니라 프레임 사이를 잇는 포인터일 뿐이다.
    """

    track_id: int
    bbox: BoundingBox
    confidence: float
    state: TrackState

    @property
    def center_x(self) -> float:
        return self.bbox.center.x

    @property
    def center_y(self) -> float:
        return self.bbox.center.y


@dataclass
class _Track:
    """내부 상태. 밖으로 내보내지 않는다."""

    track_id: int
    bbox: BoundingBox
    confidence: float
    state: TrackState
    hits: int = 1
    time_since_update: int = 0
    velocity: tuple[float, float] = (0.0, 0.0)
    ever_confirmed: bool = False

    def predicted_bbox(self) -> BoundingBox:
        """등속 외삽. 사람이 계속 걷고 있다는 가정이다."""
        dx, dy = self.velocity
        if dx == 0.0 and dy == 0.0:
            return self.bbox
        return self.bbox.translated(dx, dy)

    def apply(self, detection: Detection) -> None:
        previous = self.bbox.center
        current = detection.bbox.center
        self.velocity = (current.x - previous.x, current.y - previous.y)
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self) -> None:
        self.time_since_update += 1
        self.state = TrackState.LOST


def greedy_match(
    tracks: list[_Track],
    detections: list[Detection],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """IoU가 큰 쌍부터 하나씩 확정한다.

    (track index, detection index) 쌍과, 짝을 못 찾은 양쪽 index를 돌려준다.
    동점일 때는 index가 작은 쪽이 이긴다. 같은 입력이면 항상 같은 결과가
    나와야 실험을 재현할 수 있다.
    """
    candidates: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(tracks):
        predicted = track.predicted_bbox()
        for detection_index, detection in enumerate(detections):
            iou = predicted.iou(detection.bbox)
            if iou >= iou_threshold and iou > 0.0:
                candidates.append((iou, track_index, detection_index))

    # IoU 내림차순, 같으면 index 오름차순.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, track_index, detection_index in candidates:
        if track_index in used_tracks or detection_index in used_detections:
            continue
        used_tracks.add(track_index)
        used_detections.add(detection_index)
        matches.append((track_index, detection_index))

    unmatched_tracks = [i for i in range(len(tracks)) if i not in used_tracks]
    unmatched_detections = [
        i for i in range(len(detections)) if i not in used_detections
    ]
    return matches, unmatched_tracks, unmatched_detections


@dataclass
class ByteAssociationTracker:
    """BYTE 방식 2단계 연결.

    high_threshold: 이 이상이면 믿을 만한 상자. 새 track은 여기서만 생긴다.
    low_threshold:  이 미만은 아예 버린다. 그 사이는 2단계에서만 쓴다.
    iou_threshold:  1단계 매칭에 필요한 최소 IoU.
    low_iou_threshold: 2단계 매칭 기준. 가려진 상자는 모양이 망가져 있어서
                       1단계보다 느슨하게 잡는다.
    max_age:        몇 프레임까지 못 찾아도 살려 둘 것인가. 짧으면 잠깐
                    가려진 사람이 새 id를 받아 입장 인원이 늘고, 길면 나간
                    사람이 남아 occupancy가 늘어난다.
    min_hits:       몇 번 맞아야 세기 시작할 것인가. 1회성 오탐이 occupancy에
                    바로 들어오는 것을 막는다.
    """

    high_threshold: float = 0.5
    low_threshold: float = 0.1
    iou_threshold: float = 0.3
    low_iou_threshold: float = 0.2
    max_age: int = 30
    min_hits: int = 3

    _tracks: list[_Track] = field(default_factory=list, init=False, repr=False)
    _next_id: int = field(default=1, init=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "high_threshold",
            "low_threshold",
            "iou_threshold",
            "low_iou_threshold",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise TrackingError(f"{name}은(는) 0~1이어야 합니다: {value}")
        if self.low_threshold > self.high_threshold:
            raise TrackingError(
                "low_threshold가 high_threshold보다 클 수 없습니다: "
                f"{self.low_threshold} > {self.high_threshold}"
            )
        if self.max_age < 1:
            raise TrackingError(f"max_age는 1 이상이어야 합니다: {self.max_age}")
        if self.min_hits < 1:
            raise TrackingError(f"min_hits는 1 이상이어야 합니다: {self.min_hits}")

    @property
    def next_track_id(self) -> int:
        """다음에 나갈 번호. 테스트와 로깅용이다."""
        return self._next_id

    def reset(self) -> None:
        """영상이 바뀌면 반드시 부른다.

        track_id가 영상 경계를 넘어가면 그 순간부터 그것은 임시 번호가
        아니라 사람을 가리키는 식별자가 된다. 그것은 이 프로젝트가 하지
        않기로 한 일이다.
        """
        self._tracks.clear()
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
        """한 프레임을 넣고 지금 확정된 사람들을 받는다."""
        high = [d for d in detections if d.confidence >= self.high_threshold]
        low = [
            d
            for d in detections
            if self.low_threshold <= d.confidence < self.high_threshold
        ]

        # 1단계: 모든 track 대 높은 신뢰도 상자.
        matches, unmatched_tracks, unmatched_high = greedy_match(
            self._tracks, high, self.iou_threshold
        )
        for track_index, detection_index in matches:
            self._tracks[track_index].apply(high[detection_index])

        # 2단계: 1단계에서 남은 track 대 낮은 신뢰도 상자.
        # 갓 생긴 TENTATIVE track까지 여기 넣으면 오탐이 약한 상자를 붙잡고
        # 계속 살아남는다. 한 번이라도 confirmed였던 track만 구제한다.
        rescuable = [i for i in unmatched_tracks if self._tracks[i].ever_confirmed]
        rescue_tracks = [self._tracks[i] for i in rescuable]
        rescue_matches, _, _ = greedy_match(
            rescue_tracks, low, self.low_iou_threshold
        )
        matched_rescuable: set[int] = set()
        for local_index, detection_index in rescue_matches:
            self._tracks[rescuable[local_index]].apply(low[detection_index])
            matched_rescuable.add(rescuable[local_index])

        for track_index in unmatched_tracks:
            if track_index not in matched_rescuable:
                self._tracks[track_index].mark_missed()

        # 짝을 못 찾은 높은 신뢰도 상자는 새 사람이다.
        for detection_index in unmatched_high:
            detection = high[detection_index]
            self._tracks.append(
                _Track(
                    track_id=self._next_id,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    state=TrackState.TENTATIVE,
                )
            )
            self._next_id += 1

        # 상태 갱신과 정리.
        alive: list[_Track] = []
        for track in self._tracks:
            if track.time_since_update > self.max_age:
                continue
            if track.time_since_update == 0 and (
                track.hits >= self.min_hits or track.ever_confirmed
            ):
                track.state = TrackState.CONFIRMED
                track.ever_confirmed = True
            alive.append(track)
        self._tracks = alive

        return [
            TrackedPerson(
                track_id=track.track_id,
                bbox=track.bbox,
                confidence=track.confidence,
                state=track.state,
            )
            for track in self._tracks
            if track.state is TrackState.CONFIRMED and track.time_since_update == 0
        ]

    def active_track_ids(self) -> set[int]:
        """아직 지워지지 않은 모든 track 번호. 진단용이다."""
        return {track.track_id for track in self._tracks}
