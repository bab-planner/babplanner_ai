"""좌표와 도형."""

from __future__ import annotations

import math

import pytest

from cctv_counting.geometry import (
    BoundingBox,
    GeometryError,
    Line,
    NormalizedPoint,
    NormalizedRect,
    Point,
    Side,
    rect_contains,
)


def test_normalized_point_rejects_out_of_range():
    with pytest.raises(GeometryError):
        NormalizedPoint(1.4, 0.5)


def test_normalized_point_rejects_nan():
    with pytest.raises(GeometryError):
        NormalizedPoint(float("nan"), 0.5)


def test_normalized_coordinates_scale_with_resolution():
    """같은 설정이 해상도가 달라도 같은 자리를 가리켜야 한다."""
    point = NormalizedPoint(0.5, 0.6)
    assert point.to_pixels(1920, 1080) == Point(960.0, 648.0)
    assert point.to_pixels(1280, 720) == Point(640.0, 432.0)


def test_normalized_rect_scales_with_resolution():
    roi = NormalizedRect(0.25, 0.5, 0.75, 1.0)
    box = roi.to_pixels(800, 600)
    assert (box.x1, box.y1, box.x2, box.y2) == (200.0, 300.0, 600.0, 600.0)


def test_bounding_box_rejects_inverted():
    with pytest.raises(GeometryError):
        BoundingBox(100, 100, 50, 200)


def test_bounding_box_rejects_zero_area():
    with pytest.raises(GeometryError):
        BoundingBox(100, 100, 100, 200)


def test_bottom_center_is_below_center():
    """바닥 판정에 쓰는 점은 몸통 중심보다 아래여야 한다."""
    box = BoundingBox(100, 100, 200, 300)
    assert box.center == Point(150.0, 200.0)
    assert box.bottom_center == Point(150.0, 300.0)


def test_iou_of_identical_boxes_is_one():
    box = BoundingBox(0, 0, 10, 10)
    assert box.iou(box) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert BoundingBox(0, 0, 10, 10).iou(BoundingBox(20, 20, 30, 30)) == 0.0


def test_iou_of_touching_boxes_is_zero():
    """모서리만 닿은 것은 겹친 것이 아니다."""
    assert BoundingBox(0, 0, 10, 10).iou(BoundingBox(10, 0, 20, 10)) == 0.0


def test_iou_half_overlap():
    left = BoundingBox(0, 0, 10, 10)
    right = BoundingBox(5, 0, 15, 10)
    assert left.iou(right) == pytest.approx(50 / 150)


def test_roi_boundary_counts_as_inside():
    """경계에 걸친 사람이 깜빡이면 occupancy가 흔들린다."""
    rect = BoundingBox(0, 0, 100, 100)
    assert rect_contains(rect, Point(100.0, 50.0))
    assert not rect_contains(rect, Point(100.5, 50.0))


def test_line_rejects_zero_length():
    with pytest.raises(GeometryError):
        Line(NormalizedPoint(0.5, 0.5), NormalizedPoint(0.5, 0.5))


def test_horizontal_line_sides():
    """start->end 방향 기준 부호. 화면 아래쪽이 POSITIVE다."""
    line = Line(NormalizedPoint(0.1, 0.6), NormalizedPoint(0.9, 0.6))
    assert line.side_of(NormalizedPoint(0.5, 0.8)) is Side.POSITIVE
    assert line.side_of(NormalizedPoint(0.5, 0.4)) is Side.NEGATIVE


def test_point_exactly_on_line_is_on_line():
    line = Line(NormalizedPoint(0.1, 0.6), NormalizedPoint(0.9, 0.6))
    assert line.side_of(NormalizedPoint(0.5, 0.6)) is Side.ON_LINE


def test_band_widens_the_on_line_zone():
    line = Line(NormalizedPoint(0.1, 0.6), NormalizedPoint(0.9, 0.6))
    near = NormalizedPoint(0.5, 0.605)
    assert line.side_of(near) is Side.POSITIVE
    assert line.side_of(near, band=0.01) is Side.ON_LINE


def test_reversing_the_line_flips_the_sides():
    """선을 반대로 그으면 안팎이 뒤집힌다. in_direction을 config에 둔 이유다."""
    forward = Line(NormalizedPoint(0.1, 0.6), NormalizedPoint(0.9, 0.6))
    backward = Line(NormalizedPoint(0.9, 0.6), NormalizedPoint(0.1, 0.6))
    point = NormalizedPoint(0.5, 0.8)
    assert forward.side_of(point) is Side.POSITIVE
    assert backward.side_of(point) is Side.NEGATIVE


def test_signed_distance_magnitude_is_geometric():
    """수직 거리여야 jitter 판정에 쓸 수 있다."""
    line = Line(NormalizedPoint(0.0, 0.5), NormalizedPoint(1.0, 0.5))
    assert line.signed_distance_normalized(NormalizedPoint(0.3, 0.7)) == pytest.approx(0.2)


def test_diagonal_line_signed_distance():
    line = Line(NormalizedPoint(0.0, 0.0), NormalizedPoint(1.0, 1.0))
    distance = line.signed_distance_normalized(NormalizedPoint(1.0, 0.0))
    assert distance == pytest.approx(-1.0 / math.sqrt(2))


def test_negative_band_is_rejected():
    line = Line(NormalizedPoint(0.1, 0.6), NormalizedPoint(0.9, 0.6))
    with pytest.raises(GeometryError):
        line.side_of(NormalizedPoint(0.5, 0.5), band=-0.1)
