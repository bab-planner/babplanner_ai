"""Detection 경계와 필터."""

from __future__ import annotations

import pytest

from cctv_counting.detection import (
    COCO_PERSON_CLASS_ID,
    Anchor,
    Detection,
    DetectionError,
    PersonDetector,
    anchor_point,
    filter_detections,
    normalize_point,
)
from cctv_counting.geometry import BoundingBox, GeometryError, NormalizedRect, Point

from conftest import ScriptedDetector, person


def test_detection_rejects_confidence_above_one():
    with pytest.raises(DetectionError):
        Detection(BoundingBox(0, 0, 10, 10), 1.5, COCO_PERSON_CLASS_ID)


def test_detection_rejects_nan_confidence():
    with pytest.raises(DetectionError):
        Detection(BoundingBox(0, 0, 10, 10), float("nan"), COCO_PERSON_CLASS_ID)


def test_detection_rejects_non_integer_class():
    with pytest.raises(DetectionError):
        Detection(BoundingBox(0, 0, 10, 10), 0.9, 1.5)  # type: ignore[arg-type]


def test_scripted_detector_satisfies_the_protocol():
    """가짜 detector가 실제 detector와 같은 경계를 만족해야 한다."""
    assert isinstance(ScriptedDetector(script=[]), PersonDetector)


def test_confidence_threshold_drops_weak_boxes():
    detections = [person(0, 0, 10, 100, confidence=0.2), person(20, 0, 30, 100, 0.8)]
    kept = filter_detections(
        detections, frame_width=100, frame_height=100, min_confidence=0.5
    )
    assert [d.confidence for d in kept] == [0.8]


def test_non_person_classes_are_dropped():
    """의자를 사람으로 세면 빈 식당이 붐빈다."""
    detections = [person(0, 0, 10, 100, class_id=62), person(20, 0, 30, 100)]
    kept = filter_detections(
        detections, frame_width=100, frame_height=100, min_confidence=0.1
    )
    assert len(kept) == 1
    assert kept[0].class_id == COCO_PERSON_CLASS_ID


def test_roi_includes_person_whose_feet_are_inside():
    roi = NormalizedRect(0.0, 0.5, 1.0, 1.0)
    inside = person(400, 300, 460, 800)  # 발이 y=800 -> 0.8
    kept = filter_detections(
        [inside],
        frame_width=1000,
        frame_height=1000,
        min_confidence=0.1,
        roi=roi,
    )
    assert len(kept) == 1


def test_roi_excludes_person_whose_feet_are_outside():
    """머리가 ROI에 걸쳐도 발이 밖이면 그 공간에 있는 것이 아니다."""
    roi = NormalizedRect(0.0, 0.5, 1.0, 1.0)
    outside = person(400, 100, 460, 400)  # 발이 y=400 -> 0.4
    kept = filter_detections(
        [outside],
        frame_width=1000,
        frame_height=1000,
        min_confidence=0.1,
        roi=roi,
    )
    assert kept == []


def test_roi_uses_center_anchor_when_asked():
    roi = NormalizedRect(0.0, 0.0, 1.0, 0.5)
    subject = person(400, 100, 460, 500)  # 중심 y=300 -> 0.3, 발 y=500 -> 0.5
    by_center = filter_detections(
        [subject],
        frame_width=1000,
        frame_height=1000,
        min_confidence=0.1,
        roi=roi,
        anchor=Anchor.CENTER,
    )
    assert len(by_center) == 1


def test_no_roi_keeps_everything_in_frame():
    detections = [person(0, 0, 10, 100), person(900, 800, 990, 990)]
    kept = filter_detections(
        detections, frame_width=1000, frame_height=1000, min_confidence=0.1
    )
    assert len(kept) == 2


def test_anchor_point_selection():
    box = BoundingBox(0, 0, 100, 200)
    assert anchor_point(box, Anchor.BOTTOM_CENTER) == Point(50.0, 200.0)
    assert anchor_point(box, Anchor.CENTER) == Point(50.0, 100.0)


def test_normalize_point_clamps_boxes_that_leave_the_frame():
    """detector 상자가 프레임 밖으로 조금 삐져나오는 일이 실제로 있다."""
    assert normalize_point(Point(1010.0, -5.0), 1000, 1000).x == pytest.approx(1.0)
    assert normalize_point(Point(1010.0, -5.0), 1000, 1000).y == pytest.approx(0.0)


def test_normalize_point_rejects_bad_frame_size():
    with pytest.raises(GeometryError):
        normalize_point(Point(1.0, 1.0), 0, 100)


def test_filter_rejects_bad_threshold():
    with pytest.raises(DetectionError):
        filter_detections([], frame_width=10, frame_height=10, min_confidence=1.4)
