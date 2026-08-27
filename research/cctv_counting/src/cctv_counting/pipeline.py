"""영상 한 편을 관측으로 바꾸는 흐름.

이 모듈은 영상 파일도 모르고 numpy도 모른다. detector 객체와 프레임 크기만
받는다. 그래서 무거운 의존성 없이 가짜 detector로 전체 흐름을 그대로 테스트할
수 있다. 실제 파일 입출력은 video.py 쪽이다.

두 baseline이 같은 코드를 지난다.

    A. detection only     tracker.enabled = false
    B. detection + tracking

A는 occupancy만 낸다. 입장 인원은 셀 수 없다. 같은 사람을 프레임 사이에서
이어 붙이지 못하면 "넘었다"는 사건 자체가 정의되지 않기 때문이다. 대신
선 근처 상자 수를 순진한 대리 지표로 같이 세어 둔다. 그 값이 실제 입장
인원보다 얼마나 부풀려지는지가 추적이 하는 일의 크기다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cctv_counting.config import CctvConfig
from cctv_counting.crossing import (
    Crossing,
    LineCrossingCounter,
    count_line_band_detections,
)
from cctv_counting.detection import (
    COCO_PERSON_CLASS_ID,
    Detection,
    PersonDetector,
    anchor_point,
    filter_detections,
    normalize_point,
)
from cctv_counting.observations import FrameObservation
from cctv_counting.reliability import ReliabilityPolicy, evaluate_reliability
from cctv_counting.tracking import ByteAssociationTracker, TrackedPerson


class PipelineError(ValueError):
    """pipeline을 돌릴 수 없다."""


@dataclass(frozen=True)
class FrameResult:
    """프레임 하나의 처리 결과.

    observation은 파일로 나가는 값이고, 나머지는 화면에 그리거나 디버깅할
    때 쓴다. tracked에는 track_id가 들어 있으므로 파일로 내보내지 않는다.
    """

    observation: FrameObservation
    detections: list[Detection]
    tracked: list[TrackedPerson]
    crossings: list[Crossing]


@dataclass
class CountingPipeline:
    """detector 하나 + 설정 하나로 영상을 훑는다."""

    detector: PersonDetector
    config: CctvConfig
    frame_width: int
    frame_height: int

    _tracker: ByteAssociationTracker | None = field(default=None, init=False, repr=False)
    _crossing: LineCrossingCounter | None = field(default=None, init=False, repr=False)
    _policy: ReliabilityPolicy = field(init=False, repr=False)
    _naive_line_hits: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise PipelineError(
                f"프레임 크기가 올바르지 않습니다: {self.frame_width}x{self.frame_height}"
            )

        tracker_config = self.config.tracker
        if tracker_config.enabled:
            self._tracker = ByteAssociationTracker(
                high_threshold=tracker_config.high_threshold,
                low_threshold=tracker_config.low_threshold,
                iou_threshold=tracker_config.iou_threshold,
                low_iou_threshold=tracker_config.low_iou_threshold,
                max_age=tracker_config.max_age,
                min_hits=tracker_config.min_hits,
            )

        line = self.config.line.to_line()
        if line is not None and self._tracker is not None:
            self._crossing = LineCrossingCounter(
                line=line,
                in_direction=self.config.line.in_direction,
                band=self.config.line.band,
                min_confirm_distance=self.config.line.min_confirm_distance,
                cooldown_frames=self.config.line.cooldown_frames,
            )

        self._policy = ReliabilityPolicy(
            min_mean_confidence=self.config.reliability.min_mean_confidence,
            crowded_track_count=self.config.reliability.crowded_track_count,
        )

    @property
    def naive_line_band_detections(self) -> int | None:
        """추적을 껐을 때만 의미가 있는 대리 지표."""
        if self.config.tracker.enabled:
            return None
        return self._naive_line_hits

    def reset(self) -> None:
        """다음 영상으로 넘어가기 전에 부른다. track 번호가 영상을 넘지 않게."""
        if self._tracker is not None:
            self._tracker.reset()
        if self._crossing is not None:
            self._crossing.reset()
        self._naive_line_hits = 0

    def process_frame(
        self, frame_index: int, timestamp_seconds: float, frame: Any
    ) -> FrameResult:
        """프레임 하나를 처리한다."""
        raw = self.detector.detect(frame)
        person_class_id = (
            self.config.detector.person_class_id
            if self.config.detector.person_class_id is not None
            else COCO_PERSON_CLASS_ID
        )
        detections = filter_detections(
            raw,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            min_confidence=self.config.detector.confidence_threshold,
            person_class_id=person_class_id,
            roi=self.config.roi.to_rect(),
            anchor=self.config.line.anchor,
        )

        crossings: list[Crossing] = []
        if self._tracker is None:
            # baseline A. 상자 개수가 곧 occupancy다.
            tracked: list[TrackedPerson] = []
            occupancy = len(detections)
            confidences = [d.confidence for d in detections]
            entries: int | None = None
            exits: int | None = None
            line = self.config.line.to_line()
            if line is not None:
                points = [
                    normalize_point(
                        anchor_point(d.bbox, self.config.line.anchor),
                        self.frame_width,
                        self.frame_height,
                    )
                    for d in detections
                ]
                self._naive_line_hits += count_line_band_detections(
                    line, points, self.config.line.band
                )
        else:
            tracked = self._tracker.update(detections)
            occupancy = len(tracked)
            confidences = [person.confidence for person in tracked]
            if self._crossing is not None:
                positions = {
                    person.track_id: normalize_point(
                        anchor_point(person.bbox, self.config.line.anchor),
                        self.frame_width,
                        self.frame_height,
                    )
                    for person in tracked
                }
                crossings = self._crossing.update(frame_index, positions)
                self._crossing.forget(self._tracker.active_track_ids())
                entries = self._crossing.entries
                exits = self._crossing.exits
            else:
                entries = None
                exits = None

        signal = evaluate_reliability(confidences, occupancy, self._policy)
        observation = FrameObservation(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            occupancy_count=occupancy,
            entries_total=entries,
            exits_total=exits,
            mean_detection_confidence=signal.mean_detection_confidence,
            min_detection_confidence=signal.min_detection_confidence,
            active_track_count=signal.active_track_count,
            reliability_flag=signal.flag,
        )
        return FrameResult(
            observation=observation,
            detections=detections,
            tracked=tracked,
            crossings=crossings,
        )
