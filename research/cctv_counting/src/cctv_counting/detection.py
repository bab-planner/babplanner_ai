"""Detector 경계.

detector 하나를 고정하지 않는다. 지금은 COCO pretrained 모델을 쓰지만,
나중에 급식소 영상으로 fine-tune한 모델이나 다른 backbone으로 갈아끼울
것이다. 그래서 downstream(추적, 계수, 지표)이 아는 것은 아래 두 가지뿐이다.

    Detection: 상자 하나, 신뢰도, 클래스
    PersonDetector: detect(frame) -> list[Detection]

이 이상 추상화하지 않는다. 등록기나 factory를 두면 구현이 두 개일 때부터
읽기 어려워진다. 구현이 늘어나면 그때 만든다.

얼굴, 속성, 신원에 해당하는 필드는 여기에 두지 않는다. 애초에 자리가
없으면 실수로 흘러들 수 없다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from cctv_counting.geometry import (
    BoundingBox,
    GeometryError,
    NormalizedPoint,
    NormalizedRect,
    Point,
    rect_contains,
)

#: COCO에서 사람 클래스. 실제 값은 모델 weight의 메타데이터에서 읽는 것이
#: 원칙이고, 이 상수는 그것을 확인할 수 없을 때의 기본값이다.
COCO_PERSON_CLASS_ID = 1


class DetectionError(ValueError):
    """탐지 결과가 성립하지 않는다."""


@dataclass(frozen=True)
class Detection:
    """프레임 하나에서 나온 상자 하나."""

    bbox: BoundingBox
    confidence: float
    class_id: int

    def __post_init__(self) -> None:
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise DetectionError(f"confidence가 숫자가 아닙니다: {self.confidence!r}")
        confidence = float(self.confidence)
        if math.isnan(confidence) or math.isinf(confidence):
            raise DetectionError(f"confidence가 유한하지 않습니다: {self.confidence!r}")
        if not 0.0 <= confidence <= 1.0:
            raise DetectionError(f"confidence는 0~1이어야 합니다: {confidence}")
        object.__setattr__(self, "confidence", confidence)
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int):
            raise DetectionError(f"class_id는 정수여야 합니다: {self.class_id!r}")


@runtime_checkable
class PersonDetector(Protocol):
    """프레임 하나를 받아 상자들을 돌려준다.

    frame의 타입은 backend에 맡긴다. 지금 구현은 numpy BGR 배열을 받지만
    이 경계는 그것을 요구하지 않는다. core 로직은 프레임을 만지지 않는다.
    """

    def detect(self, frame: Any) -> list[Detection]:  # pragma: no cover - 규약
        ...


class Anchor(str, Enum):
    """상자를 대표하는 점을 무엇으로 볼 것인가."""

    BOTTOM_CENTER = "BOTTOM_CENTER"
    CENTER = "CENTER"


def anchor_point(bbox: BoundingBox, anchor: Anchor) -> Point:
    if anchor is Anchor.CENTER:
        return bbox.center
    return bbox.bottom_center


def normalize_point(point: Point, width: int, height: int) -> NormalizedPoint:
    """픽셀 좌표를 0~1로 되돌린다.

    detector 출력이 프레임 밖으로 조금 삐져나오는 일이 있어서 잘라 넣는다.
    """
    if width <= 0 or height <= 0:
        raise GeometryError(f"프레임 크기가 올바르지 않습니다: {width}x{height}")
    return NormalizedPoint(
        min(1.0, max(0.0, point.x / width)),
        min(1.0, max(0.0, point.y / height)),
    )


def filter_detections(
    detections: list[Detection],
    *,
    frame_width: int,
    frame_height: int,
    min_confidence: float,
    person_class_id: int = COCO_PERSON_CLASS_ID,
    roi: NormalizedRect | None = None,
    anchor: Anchor = Anchor.BOTTOM_CENTER,
) -> list[Detection]:
    """사람만, 임계값 이상만, ROI 안쪽만 남긴다.

    ROI 판정은 상자 전체가 아니라 anchor 한 점으로 한다. 상자 전체를
    요구하면 ROI 경계에 반쯤 걸친 사람이 전부 빠지고, 상자가 조금이라도
    겹치면 통과시키면 ROI가 사실상 넓어진다. 발이 어디 있느냐가 "그 공간에
    있느냐"에 가장 가깝다.
    """
    if not 0.0 <= min_confidence <= 1.0:
        raise DetectionError(f"min_confidence는 0~1이어야 합니다: {min_confidence}")

    kept: list[Detection] = []
    for detection in detections:
        if detection.class_id != person_class_id:
            continue
        if detection.confidence < min_confidence:
            continue
        if roi is not None:
            point = normalize_point(
                anchor_point(detection.bbox, anchor), frame_width, frame_height
            )
            if not roi.contains_normalized(point):
                continue
        kept.append(detection)
    return kept


def roi_contains_bbox_anchor(
    bbox: BoundingBox,
    roi: NormalizedRect,
    *,
    frame_width: int,
    frame_height: int,
    anchor: Anchor = Anchor.BOTTOM_CENTER,
) -> bool:
    """추적된 상자가 아직 ROI 안인지. filter_detections와 같은 규칙을 쓴다."""
    point = normalize_point(anchor_point(bbox, anchor), frame_width, frame_height)
    return roi.contains_normalized(point)


__all__ = [
    "COCO_PERSON_CLASS_ID",
    "Anchor",
    "Detection",
    "DetectionError",
    "PersonDetector",
    "anchor_point",
    "filter_detections",
    "normalize_point",
    "rect_contains",
    "roi_contains_bbox_anchor",
]
