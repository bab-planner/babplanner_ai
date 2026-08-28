"""궤적 연속성을 강화한 tracker.

**이름을 정확히 쓴다.** 이것은 ByteTrack이 아니다. `tracking.py`의
`ByteTrackInspiredTracker`와 마찬가지로 high/low 2단계 연결 아이디어만
빌렸고, 나머지는 여기서 정한 것이다. Kalman filter도 쓰지 않는다. 그래서
이름이 `MotionAwareTracker`다 — 하는 일 그대로다.

**A5가 남긴 문제.** 통과 계수 누락 23건 중 다른 detector가 검출을 전부
회복시켰는데도 통과 사건은 43%만 살아났다. 남은 절반은 검출 문제가 아니라
**궤적이 끊기는 문제**다. track이 끊기면 새 번호를 받고, 새 번호는 선의
어느 쪽에 있었는지를 모른다(`crossing.py`의 `confirmed_side`가 None에서
시작한다). 그러면 그 사람이 선을 넘어도 사건이 나오지 않는다.

기존 tracker 대비 바뀐 것은 넷이다.

    1. 시간 기반 소멸        max_age(프레임) -> max_age_seconds
    2. 시간 비례 외삽        1프레임분 고정 -> 놓친 시간에 비례
    3. 속도 평활             직전 프레임 차이 -> 지수이동평균
    4. 전역 할당             greedy IoU -> Hungarian (assignment.py)

**1이 왜 중요한가.** 프레임 기반 소멸은 fps에 따라 정책이 달라진다. 30프레임은
10fps에서 3초, 60fps에서 0.5초다. 같은 설정 파일이 카메라마다 다른 tracker가
된다. A3부터 알고 있던 한계이고 여기서 고친다.

**2가 왜 중요한가.** 기존 `_Track.predicted_bbox()`는 놓친 시간과 무관하게
직전 이동량 한 번만 더한다. 10프레임을 놓쳐도 예측 상자는 1프레임 앞에
머문다. 사람이 계속 걷고 있으면 실제 위치는 10프레임분 앞에 있으므로 IoU가
0이 되고, 다시 나타난 그 사람은 새 번호를 받는다. **가림 구간이 길수록 이
오차가 커진다.** 통과 사건은 대개 사람이 카메라 앞을 가로지르는 동안 서로
가리는 구간에서 일어나므로 정확히 그 구간에서 끊긴다.

**3이 왜 필요한가.** 2를 하려면 속도가 믿을 만해야 한다. 직전 한 프레임의
차이는 상자 떨림이 그대로 들어와 시끄럽다. 그 값을 몇 프레임분 곱하면 예측이
엉뚱한 곳으로 날아간다. 지수이동평균으로 눌러 둔다.

**하지 않는 것.**

- 외형(re-id) 특징을 쓰지 않는다. 익명 계수라는 제약 그대로다.
- 놓친 구간의 **예측 위치를 통과 판정에 넣지 않는다.** 내보내는 것은
  이번 프레임에 실제로 갱신된 track뿐이다. 예측으로 선을 넘겼다고 세면
  없는 사건을 만들어낸다. 예측은 **다시 만났을 때 같은 번호를 주는**
  데에만 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cctv_counting.assignment import match_by_iou
from cctv_counting.detection import Detection
from cctv_counting.geometry import BoundingBox
from cctv_counting.tracking import TrackedPerson, TrackingError, TrackState


@dataclass
class _MotionTrack:
    """내부 상태. 밖으로 내보내지 않는다."""

    track_id: int
    bbox: BoundingBox
    confidence: float
    state: TrackState
    last_update_time: float
    hits: int = 1
    #: 초당 이동량(픽셀). 프레임당이 아니다. fps가 달라도 같은 값이 나온다.
    velocity: tuple[float, float] = (0.0, 0.0)
    ever_confirmed: bool = False
    observed: bool = True

    def seconds_since_update(self, now: float) -> float:
        return max(0.0, now - self.last_update_time)

    def predicted_bbox(self, now: float, max_coast_seconds: float) -> BoundingBox:
        """놓친 시간에 비례해 외삽한다.

        max_coast_seconds에서 자른다. 오래 놓친 track의 예측이 화면 밖으로
        날아가서 엉뚱한 사람과 겹치는 것을 막는다.
        """
        dx, dy = self.velocity
        if dx == 0.0 and dy == 0.0:
            return self.bbox
        elapsed = min(self.seconds_since_update(now), max_coast_seconds)
        if elapsed <= 0.0:
            return self.bbox
        return self.bbox.translated(dx * elapsed, dy * elapsed)

    def apply(self, detection: Detection, now: float, smoothing: float) -> None:
        elapsed = now - self.last_update_time
        if elapsed > 0.0:
            previous = self.bbox.center
            current = detection.bbox.center
            instant = (
                (current.x - previous.x) / elapsed,
                (current.y - previous.y) / elapsed,
            )
            if self.velocity == (0.0, 0.0):
                self.velocity = instant
            else:
                self.velocity = (
                    smoothing * instant[0] + (1.0 - smoothing) * self.velocity[0],
                    smoothing * instant[1] + (1.0 - smoothing) * self.velocity[1],
                )
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        self.hits += 1
        self.last_update_time = now
        self.observed = True

    def mark_missed(self) -> None:
        self.state = TrackState.LOST
        self.observed = False


@dataclass
class MotionAwareTracker:
    """시간 기반 소멸 + 시간 비례 외삽 + 전역 할당.

    high_threshold:     이 이상이면 믿을 만한 상자. 새 track은 여기서만 생긴다.
    low_threshold:      이 미만은 버린다. 그 사이는 2단계 구제에만 쓴다.
    iou_threshold:      1단계 매칭 최소 IoU.
    low_iou_threshold:  2단계 매칭 최소 IoU. 가려진 상자는 모양이 망가져
                        있어서 느슨하게 잡는다.
    max_age_seconds:    몇 **초**까지 못 찾아도 살려 둘 것인가. 프레임이
                        아니다. 25fps에서 1.2초 = 기존 max_age 30프레임.
    min_hits:           몇 번 맞아야 세기 시작할 것인가.
    velocity_smoothing: 속도 지수이동평균 계수. 1.0이면 평활 없음(직전 값
                        그대로), 0에 가까울수록 과거를 오래 기억한다.
    max_coast_seconds:  외삽을 이 시간에서 자른다. None이면 max_age_seconds.
    """

    high_threshold: float = 0.5
    low_threshold: float = 0.1
    iou_threshold: float = 0.3
    low_iou_threshold: float = 0.2
    max_age_seconds: float = 1.2
    min_hits: int = 1
    velocity_smoothing: float = 0.5
    max_coast_seconds: float | None = None

    _tracks: list[_MotionTrack] = field(default_factory=list, init=False, repr=False)
    _next_id: int = field(default=1, init=False, repr=False)
    _last_time: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "high_threshold",
            "low_threshold",
            "iou_threshold",
            "low_iou_threshold",
            "velocity_smoothing",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise TrackingError(f"{name}은(는) 0~1이어야 합니다: {value}")
        if self.low_threshold > self.high_threshold:
            raise TrackingError(
                "low_threshold가 high_threshold보다 클 수 없습니다: "
                f"{self.low_threshold} > {self.high_threshold}"
            )
        if not self.max_age_seconds > 0.0:
            raise TrackingError(
                f"max_age_seconds는 0보다 커야 합니다: {self.max_age_seconds}"
            )
        if self.min_hits < 1:
            raise TrackingError(f"min_hits는 1 이상이어야 합니다: {self.min_hits}")
        if self.max_coast_seconds is not None and self.max_coast_seconds < 0.0:
            raise TrackingError(
                f"max_coast_seconds는 0 이상이어야 합니다: {self.max_coast_seconds}"
            )

    @property
    def coast_limit(self) -> float:
        return (
            self.max_age_seconds
            if self.max_coast_seconds is None
            else self.max_coast_seconds
        )

    @property
    def next_track_id(self) -> int:
        return self._next_id

    def reset(self) -> None:
        """영상이 바뀌면 반드시 부른다. 번호가 영상 경계를 넘지 않게."""
        self._tracks.clear()
        self._next_id = 1
        self._last_time = None

    def update(
        self, detections: list[Detection], timestamp_seconds: float
    ) -> list[TrackedPerson]:
        """한 프레임을 넣고 지금 확정된 사람들을 받는다.

        기존 tracker와 달리 **관측 시각을 받는다.** 소멸도 외삽도 시간으로
        판단하기 때문이다. 같은 궤적이면 몇 fps로 샘플링하든 같은 결과가
        나와야 한다.
        """
        if timestamp_seconds != timestamp_seconds:  # NaN
            raise TrackingError("timestamp_seconds가 NaN입니다.")
        if self._last_time is not None and timestamp_seconds < self._last_time:
            raise TrackingError(
                f"시각이 거꾸로 갑니다: {timestamp_seconds} < {self._last_time}"
            )
        now = timestamp_seconds
        self._last_time = now

        high = [d for d in detections if d.confidence >= self.high_threshold]
        low = [
            d
            for d in detections
            if self.low_threshold <= d.confidence < self.high_threshold
        ]

        predicted = [t.predicted_bbox(now, self.coast_limit) for t in self._tracks]

        # 1단계: 모든 track 대 높은 신뢰도 상자. 전역 할당이다.
        matches, unmatched_tracks, unmatched_high = match_by_iou(
            predicted, [d.bbox for d in high], self.iou_threshold
        )
        for track_index, detection_index in matches:
            self._tracks[track_index].apply(
                high[detection_index], now, self.velocity_smoothing
            )

        # 2단계: 남은 track 대 낮은 신뢰도 상자.
        # 한 번이라도 confirmed였던 track만 구제한다. 갓 생긴 track까지
        # 넣으면 오탐이 약한 상자를 붙잡고 계속 살아남는다.
        rescuable = [i for i in unmatched_tracks if self._tracks[i].ever_confirmed]
        rescue_matches, _, _ = match_by_iou(
            [predicted[i] for i in rescuable],
            [d.bbox for d in low],
            self.low_iou_threshold,
        )
        rescued: set[int] = set()
        for local_index, detection_index in rescue_matches:
            track_index = rescuable[local_index]
            self._tracks[track_index].apply(
                low[detection_index], now, self.velocity_smoothing
            )
            rescued.add(track_index)

        for track_index in unmatched_tracks:
            if track_index not in rescued:
                self._tracks[track_index].mark_missed()

        # 짝을 못 찾은 높은 신뢰도 상자는 새 사람이다.
        for detection_index in unmatched_high:
            detection = high[detection_index]
            self._tracks.append(
                _MotionTrack(
                    track_id=self._next_id,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    state=TrackState.TENTATIVE,
                    last_update_time=now,
                )
            )
            self._next_id += 1

        alive: list[_MotionTrack] = []
        for track in self._tracks:
            if track.seconds_since_update(now) > self.max_age_seconds:
                continue
            if track.observed and (
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
            if track.state is TrackState.CONFIRMED and track.observed
        ]

    def active_track_ids(self) -> set[int]:
        """아직 지워지지 않은 모든 track 번호. 진단용이다."""
        return {track.track_id for track in self._tracks}
