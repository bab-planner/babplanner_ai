"""공유 detector + 과제별 정책(dual path).

**한 번 추론하고 두 갈래로 쓴다.**

    Video
      |
      +-- Person Detector  (프레임당 정확히 한 번)
            |
            +-- occupancy 갈래 : 높은 신뢰도만 남겨 그대로 센다 (추적 없음)
            |
            +-- entry 갈래     : 높은 + 낮은 신뢰도를 tracker에 넘긴다
                                  -> 선 통과 계수

**왜 정책을 갈라야 하는가.** A4/A5는 두 갈래가 같은 임계값을 썼다. 그런데
두 과제는 서로 다른 오차를 싫어한다.

    occupancy   오탐 하나가 그 프레임의 인원을 그대로 부풀린다.
                놓친 사람은 다음 프레임에 다시 셀 기회가 있다.
                -> 높은 임계값이 유리하다.

    entry       한 사람의 궤적이 한 프레임 끊기면 track이 쪼개지고,
                쪼개진 track은 선을 넘은 것을 모른다(사건 자체가 사라진다).
                놓친 상자를 되찾을 두 번째 기회가 없다.
                -> 낮은 신뢰도 상자라도 궤적을 잇는 데 쓰는 편이 유리하다.

A5의 관찰이 이것을 뒷받침한다. occupancy MAE로 고른 detector가 통과 계수에서
더 나빴다. 같은 축 하나로 두 과제를 고르면 한쪽은 반드시 손해를 본다.

**그렇다고 detector를 두 번 돌리지 않는다.** 추론이 전체 시간의 대부분이고,
두 번 돌리면 서비스에서 그대로 두 배가 된다. 갈라지는 것은 **필터 정책**이지
추론이 아니다. 이 모듈은 그 규칙을 코드로 못박는다 —
`detector_calls`가 프레임 수와 같아야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cctv_counting.crossing import Crossing, InDirection, LineCrossingCounter
from cctv_counting.detection import (
    COCO_PERSON_CLASS_ID,
    Anchor,
    Detection,
    PersonDetector,
    anchor_point,
    normalize_point,
)
from cctv_counting.geometry import Line, NormalizedRect
from cctv_counting.motion_tracking import MotionAwareTracker
from cctv_counting.tracking import ByteTrackInspiredTracker, TrackedPerson


class DualPathError(ValueError):
    """dual path 설정이 성립하지 않는다."""


class TrackerKind(str, Enum):
    """entry 갈래에서 어떤 tracker를 쓸 것인가."""

    #: A2~A5의 tracker. greedy IoU + 1프레임 등속 외삽 + 프레임 기반 소멸.
    BASELINE_GREEDY = "baseline_greedy"

    #: A6에서 추가. 전역 할당 + 시간 비례 외삽 + 초 기반 소멸.
    MOTION_AWARE = "motion_aware"


@dataclass(frozen=True)
class TaskPolicy:
    """과제별 임계값과 tracker 설정.

    occupancy_threshold      occupancy 갈래가 남기는 최소 신뢰도.
    tracker_high_threshold   tracker가 새 track을 만들 수 있는 신뢰도.
    tracker_low_threshold    tracker에 넘기는 최소 신뢰도. 이 아래는 버린다.
                             high와 같게 두면 낮은 신뢰도 구제가 꺼진다.
    """

    occupancy_threshold: float = 0.7
    tracker_high_threshold: float = 0.7
    tracker_low_threshold: float = 0.7
    tracker: TrackerKind = TrackerKind.BASELINE_GREEDY
    iou_threshold: float = 0.3
    low_iou_threshold: float = 0.2
    #: BASELINE_GREEDY에서만 쓴다. 프레임 단위다.
    max_age_frames: int = 30
    #: MOTION_AWARE에서만 쓴다. 초 단위다.
    max_age_seconds: float = 1.2
    min_hits: int = 1
    velocity_smoothing: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "occupancy_threshold",
            "tracker_high_threshold",
            "tracker_low_threshold",
            "iou_threshold",
            "low_iou_threshold",
            "velocity_smoothing",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise DualPathError(f"{name}은(는) 0~1이어야 합니다: {value}")
        if self.tracker_low_threshold > self.tracker_high_threshold:
            raise DualPathError(
                "tracker_low_threshold가 tracker_high_threshold보다 클 수 "
                f"없습니다: {self.tracker_low_threshold} > "
                f"{self.tracker_high_threshold}"
            )
        if self.max_age_frames < 1:
            raise DualPathError(
                f"max_age_frames는 1 이상이어야 합니다: {self.max_age_frames}"
            )
        if not self.max_age_seconds > 0.0:
            raise DualPathError(
                f"max_age_seconds는 0보다 커야 합니다: {self.max_age_seconds}"
            )
        if self.min_hits < 1:
            raise DualPathError(f"min_hits는 1 이상이어야 합니다: {self.min_hits}")

    @property
    def uses_low_confidence(self) -> bool:
        """entry 갈래가 낮은 신뢰도 상자를 실제로 받는가."""
        return self.tracker_low_threshold < self.tracker_high_threshold

    def as_dict(self) -> dict[str, Any]:
        """동결 기록에 그대로 실을 수 있는 형태."""
        return {
            "occupancyThreshold": self.occupancy_threshold,
            "trackerHighThreshold": self.tracker_high_threshold,
            "trackerLowThreshold": self.tracker_low_threshold,
            "tracker": self.tracker.value,
            "iouThreshold": self.iou_threshold,
            "lowIouThreshold": self.low_iou_threshold,
            "maxAgeFrames": self.max_age_frames,
            "maxAgeSeconds": self.max_age_seconds,
            "minHits": self.min_hits,
            "velocitySmoothing": self.velocity_smoothing,
        }


@dataclass(frozen=True)
class SceneGeometry:
    """카메라에 대한 기하 설정. 시퀀스마다 바꾸지 않는다."""

    frame_width: int
    frame_height: int
    line: Line
    in_direction: InDirection = InDirection.POSITIVE
    roi: NormalizedRect | None = None
    anchor: Anchor = Anchor.BOTTOM_CENTER
    band: float = 0.01
    min_confirm_distance: float = 0.03
    cooldown_seconds: float = 0.5

    def __post_init__(self) -> None:
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise DualPathError(
                f"프레임 크기가 올바르지 않습니다: {self.frame_width}x{self.frame_height}"
            )


@dataclass(frozen=True)
class DualPathFrame:
    """프레임 하나의 결과.

    occupancy와 entry가 **같은 raw detections**에서 나왔다는 것이 요점이다.
    """

    frame_index: int
    timestamp_seconds: float
    occupancy_count: int
    entries_total: int
    exits_total: int
    occupancy_detections: list[Detection]
    tracker_detections: list[Detection]
    tracked: list[TrackedPerson]
    crossings: list[Crossing]


@dataclass
class DualPathCounter:
    """공유 detector 위에 두 정책을 얹는다."""

    policy: TaskPolicy
    scene: SceneGeometry
    person_class_id: int = COCO_PERSON_CLASS_ID

    #: detector를 몇 번 불렀는가. 프레임 수와 같아야 한다.
    detector_calls: int = field(default=0, init=False)

    _tracker: Any = field(init=False, repr=False)
    _crossing: LineCrossingCounter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        policy = self.policy
        if policy.tracker is TrackerKind.MOTION_AWARE:
            self._tracker = MotionAwareTracker(
                high_threshold=policy.tracker_high_threshold,
                low_threshold=policy.tracker_low_threshold,
                iou_threshold=policy.iou_threshold,
                low_iou_threshold=policy.low_iou_threshold,
                max_age_seconds=policy.max_age_seconds,
                min_hits=policy.min_hits,
                velocity_smoothing=policy.velocity_smoothing,
            )
        else:
            self._tracker = ByteTrackInspiredTracker(
                high_threshold=policy.tracker_high_threshold,
                low_threshold=policy.tracker_low_threshold,
                iou_threshold=policy.iou_threshold,
                low_iou_threshold=policy.low_iou_threshold,
                max_age=policy.max_age_frames,
                min_hits=policy.min_hits,
            )
        self._crossing = LineCrossingCounter(
            line=self.scene.line,
            in_direction=self.scene.in_direction,
            band=self.scene.band,
            min_confirm_distance=self.scene.min_confirm_distance,
            cooldown_seconds=self.scene.cooldown_seconds,
        )

    def reset(self) -> None:
        """다음 영상으로 넘어가기 전에 부른다. 번호가 영상을 넘지 않게."""
        self._tracker.reset()
        self._crossing.reset()

    @property
    def entries(self) -> int:
        return self._crossing.entries

    @property
    def exits(self) -> int:
        return self._crossing.exits

    def active_track_ids(self) -> set[int]:
        return self._tracker.active_track_ids()

    def _eligible(self, detection: Detection, threshold: float) -> bool:
        if detection.class_id != self.person_class_id:
            return False
        if detection.confidence < threshold:
            return False
        if self.scene.roi is None:
            return True
        point = normalize_point(
            anchor_point(detection.bbox, self.scene.anchor),
            self.scene.frame_width,
            self.scene.frame_height,
        )
        return self.scene.roi.contains_normalized(point)

    def process_frame(
        self, frame_index: int, timestamp_seconds: float, frame: Any, detector: PersonDetector
    ) -> DualPathFrame:
        """영상 프레임 하나. **detector를 정확히 한 번 부른다.**"""
        self.detector_calls += 1
        return self.process_detections(
            frame_index, timestamp_seconds, detector.detect(frame)
        )

    def process_detections(
        self,
        frame_index: int,
        timestamp_seconds: float,
        raw: list[Detection],
    ) -> DualPathFrame:
        """이미 얻은 raw detections로 두 갈래를 돌린다.

        캐시된 상자로 실험을 다시 돌릴 때 쓰는 입구다. detector를 부르지
        않으므로 `detector_calls`도 늘지 않는다. 실험에서 같은 추론 결과를
        여러 설정에 재사용하는 것이 곧 "한 번만 추론한다"의 실현이다.
        """
        occupancy_detections = [
            d for d in raw if self._eligible(d, self.policy.occupancy_threshold)
        ]
        tracker_detections = [
            d for d in raw if self._eligible(d, self.policy.tracker_low_threshold)
        ]

        if self.policy.tracker is TrackerKind.MOTION_AWARE:
            tracked = self._tracker.update(tracker_detections, timestamp_seconds)
        else:
            tracked = self._tracker.update(tracker_detections)

        positions = {
            person.track_id: normalize_point(
                anchor_point(person.bbox, self.scene.anchor),
                self.scene.frame_width,
                self.scene.frame_height,
            )
            for person in tracked
        }
        crossings = self._crossing.update(
            frame_index, timestamp_seconds, positions
        )
        self._crossing.forget(self._tracker.active_track_ids())

        return DualPathFrame(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            # occupancy는 추적을 쓰지 않는다. A4/A5에서 추적이 occupancy를
            # 개선하지 않는다는 결과가 나왔고, 그 갈래는 그대로 둔다.
            occupancy_count=len(occupancy_detections),
            entries_total=self._crossing.entries,
            exits_total=self._crossing.exits,
            occupancy_detections=occupancy_detections,
            tracker_detections=tracker_detections,
            tracked=tracked,
            crossings=crossings,
        )
