"""Stage A6 — 과제별 정책 분리와 궤적 연속성 tracker.

무거운 추론은 여기서 돌리지 않는다. 지키는 것은 셋이다.

    1. occupancy와 entry가 **같은 추론**을 공유하면서 다른 정책을 쓴다.
    2. 새 tracker의 소멸과 외삽이 **프레임이 아니라 시간**에 걸려 있다.
    3. 할당이 1:1이고 결정적이다.

그리고 하나 더 — **re-identification을 하지 않는다**는 것이 코드 수준에서
확인되어야 한다. 그것이 이 프로젝트의 제약이고, 제약은 문서가 아니라 테스트가
지킨다.
"""

from __future__ import annotations

import pytest

from cctv_counting.assignment import AssignmentError, linear_sum_assignment, match_by_iou
from cctv_counting.crossing import InDirection
from cctv_counting.detection import Detection
from cctv_counting.dualpath import (
    DualPathCounter,
    DualPathError,
    SceneGeometry,
    TaskPolicy,
    TrackerKind,
)
from cctv_counting.geometry import BoundingBox, Line, NormalizedPoint, NormalizedRect
from cctv_counting.motion_tracking import MotionAwareTracker, _MotionTrack
from cctv_counting.tracking import (
    ByteTrackInspiredTracker,
    TrackState,
    TrackingError,
)

FRAME_WIDTH, FRAME_HEIGHT = 384, 288
LINE = Line(NormalizedPoint(0.5, 0.2), NormalizedPoint(0.5, 0.95))
SCENE = SceneGeometry(
    frame_width=FRAME_WIDTH,
    frame_height=FRAME_HEIGHT,
    line=LINE,
    in_direction=InDirection.POSITIVE,
    roi=NormalizedRect(0.0, 0.20, 1.0, 0.95),
)


def person(x: float, confidence: float = 0.9, width: float = 40.0) -> Detection:
    """화면 세로 중간을 걷는 사람 하나. x는 상자 왼쪽 픽셀."""
    return Detection(BoundingBox(x, 120.0, x + width, 240.0), confidence, 1)


# --- 전역 할당 -------------------------------------------------------------


def test_assignment_minimises_total_cost_not_the_first_pick():
    """greedy가 지는 전형적인 배치. 두 사람이 스쳐 지나갈 때 생긴다."""
    cost = [[0.45, 0.50], [0.48, 0.90]]
    assert linear_sum_assignment(cost) == [(0, 1), (1, 0)]


def test_assignment_is_one_to_one():
    pairs = linear_sum_assignment([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    rows = [r for r, _ in pairs]
    cols = [c for _, c in pairs]
    assert len(rows) == len(set(rows))
    assert len(cols) == len(set(cols))
    assert pairs == [(0, 1), (1, 0), (2, 2)]


def test_assignment_handles_more_rows_than_columns():
    pairs = linear_sum_assignment([[1.0, 9.0], [9.0, 1.0], [5.0, 5.0]])
    assert len(pairs) == 2
    assert {r for r, _ in pairs} <= {0, 1, 2}
    assert sorted(c for _, c in pairs) == [0, 1]


def test_assignment_is_deterministic_on_ties():
    cost = [[1.0, 1.0], [1.0, 1.0]]
    assert linear_sum_assignment(cost) == linear_sum_assignment(cost) == [(0, 0), (1, 1)]


def test_assignment_rejects_ragged_or_infinite_cost():
    with pytest.raises(AssignmentError):
        linear_sum_assignment([[1.0, 2.0], [3.0]])
    with pytest.raises(AssignmentError):
        linear_sum_assignment([[float("inf")]])


def test_iou_matching_drops_pairs_below_the_threshold():
    predicted = [BoundingBox(0, 0, 10, 10)]
    detections = [BoundingBox(100, 100, 110, 110)]
    matches, unmatched_tracks, unmatched_detections = match_by_iou(
        predicted, detections, 0.3
    )
    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_detections == [0]


def test_global_matching_keeps_both_tracks_where_greedy_drops_one():
    """A6가 전역 할당으로 바꾼 이유를 그대로 잰다."""
    from cctv_counting.tracking import _Track, greedy_match

    a = BoundingBox(0.0, 0.0, 100.0, 100.0)
    b = BoundingBox(30.0, 0.0, 130.0, 100.0)
    d1 = BoundingBox(20.0, 0.0, 120.0, 100.0)
    d2 = BoundingBox(60.0, 0.0, 160.0, 100.0)

    tracks = [
        _Track(1, a, 0.9, TrackState.CONFIRMED),
        _Track(2, b, 0.9, TrackState.CONFIRMED),
    ]
    detections = [Detection(d1, 0.9, 1), Detection(d2, 0.9, 1)]
    greedy, _, _ = greedy_match(tracks, detections, 0.3)
    global_matches, _, _ = match_by_iou([a, b], [d1, d2], 0.3)

    assert len(global_matches) == 2
    assert len(greedy) < len(global_matches)


# --- 시간 기반 운동 모델 ---------------------------------------------------


def test_motion_prediction_scales_with_elapsed_time():
    """기존 tracker는 몇 프레임을 놓쳐도 1프레임분만 외삽했다."""
    track = _MotionTrack(
        track_id=1,
        bbox=BoundingBox(0.0, 0.0, 10.0, 10.0),
        confidence=0.9,
        state=TrackState.CONFIRMED,
        last_update_time=0.0,
    )
    track.velocity = (100.0, 0.0)  # 초당 100픽셀

    near = track.predicted_bbox(0.1, max_coast_seconds=10.0)
    far = track.predicted_bbox(0.5, max_coast_seconds=10.0)
    assert near.x1 == pytest.approx(10.0)
    assert far.x1 == pytest.approx(50.0)


def test_motion_prediction_is_capped_by_max_coast_seconds():
    track = _MotionTrack(
        track_id=1,
        bbox=BoundingBox(0.0, 0.0, 10.0, 10.0),
        confidence=0.9,
        state=TrackState.CONFIRMED,
        last_update_time=0.0,
    )
    track.velocity = (100.0, 0.0)
    assert track.predicted_bbox(9.0, max_coast_seconds=0.4).x1 == pytest.approx(40.0)


def test_motion_tracker_recovers_the_same_id_after_a_detection_gap():
    """A6의 핵심 주장. 기존 tracker는 여기서 새 번호를 준다."""
    fps = 25.0
    gap = set(range(8, 14))
    motion = MotionAwareTracker(min_hits=1, max_age_seconds=1.2)
    baseline = ByteTrackInspiredTracker(min_hits=1, max_age=30)

    motion_ids: list[int] = []
    baseline_ids: list[int] = []
    for index in range(20):
        detections = [] if index in gap else [person(100.0 + index * 8.0)]
        motion_ids += [p.track_id for p in motion.update(detections, index / fps)]
        baseline_ids += [p.track_id for p in baseline.update(detections)]

    assert len(set(motion_ids)) == 1
    assert len(set(baseline_ids)) == 2


def test_track_expiry_is_measured_in_seconds():
    tracker = MotionAwareTracker(min_hits=1, max_age_seconds=1.0)
    tracker.update([person(100.0)], 0.0)
    tracker.update([], 0.5)
    assert tracker.active_track_ids() == {1}
    tracker.update([], 1.01)
    assert tracker.active_track_ids() == set()


@pytest.mark.parametrize("fps", [10.0, 25.0, 30.0, 60.0])
def test_expiry_is_fps_invariant(fps: float):
    """같은 궤적, 같은 공백 시간. fps만 다르다.

    사람은 1.5초 동안 보이지 않는다. max_age가 1.0초이므로 어느 fps에서도
    track은 지워지고, 다시 나타난 사람은 새 번호를 받아야 한다.
    """
    tracker = MotionAwareTracker(min_hits=1, max_age_seconds=1.0)
    seen: list[int] = []
    total = 3.0
    steps = int(total * fps)
    for step in range(steps + 1):
        now = step / fps
        visible = now <= 1.0 or now >= 2.5
        detections = [person(100.0 + now * 20.0)] if visible else []
        seen += [p.track_id for p in tracker.update(detections, now)]
    assert len(set(seen)) == 2


@pytest.mark.parametrize("fps", [10.0, 25.0, 30.0, 60.0])
def test_short_gap_survives_at_every_fps(fps: float):
    """0.4초 공백은 max_age 1.0초 안이다. 어느 fps에서도 같은 번호여야 한다."""
    tracker = MotionAwareTracker(min_hits=1, max_age_seconds=1.0)
    seen: list[int] = []
    for step in range(int(3.0 * fps) + 1):
        now = step / fps
        visible = not (1.0 < now < 1.4)
        detections = [person(100.0 + now * 20.0)] if visible else []
        seen += [p.track_id for p in tracker.update(detections, now)]
    assert len(set(seen)) == 1


def test_motion_tracker_rejects_backwards_time():
    tracker = MotionAwareTracker()
    tracker.update([person(100.0)], 1.0)
    with pytest.raises(TrackingError):
        tracker.update([], 0.5)


def test_motion_tracker_validates_its_configuration():
    with pytest.raises(TrackingError):
        MotionAwareTracker(max_age_seconds=0.0)
    with pytest.raises(TrackingError):
        MotionAwareTracker(high_threshold=0.3, low_threshold=0.5)
    with pytest.raises(TrackingError):
        MotionAwareTracker(min_hits=0)


def test_reset_clears_ids_so_they_never_cross_a_video_boundary():
    tracker = MotionAwareTracker(min_hits=1)
    tracker.update([person(100.0)], 0.0)
    tracker.reset()
    assert tracker.active_track_ids() == set()
    assert tracker.next_track_id == 1
    # 시각도 초기화된다. 다음 영상이 0초부터 시작해도 거꾸로 간다고 보지 않는다.
    tracker.update([person(100.0)], 0.0)


# --- re-identification을 하지 않는다 ---------------------------------------


def test_tracker_carries_no_appearance_or_identity_fields():
    """외형 특징이 코드에 들어오면 이 테스트가 먼저 깨진다."""
    tracker = MotionAwareTracker(min_hits=1)
    tracked = tracker.update([person(100.0)], 0.0)
    assert [p.track_id for p in tracked] == [1]

    forbidden = ("embedding", "feature", "appearance", "descriptor", "face", "reid")
    for name in vars(
        _MotionTrack(1, BoundingBox(0, 0, 1, 1), 0.9, TrackState.CONFIRMED, 0.0)
    ):
        assert not any(word in name.lower() for word in forbidden)
    for name in vars(tracked[0]):
        assert not any(word in name.lower() for word in forbidden)


def test_track_ids_do_not_survive_reset_across_videos():
    tracker = MotionAwareTracker(min_hits=1)
    first = tracker.update([person(100.0)], 0.0)[0].track_id
    tracker.reset()
    second = tracker.update([person(300.0)], 0.0)[0].track_id
    assert first == second == 1  # 같은 번호지만 같은 사람이 아니다


# --- 공유 detector와 과제별 정책 -------------------------------------------


class CountingDetector:
    """부른 횟수를 세는 가짜 detector."""

    def __init__(self, script: list[list[Detection]]) -> None:
        self.script = script
        self.calls = 0

    def detect(self, frame: object) -> list[Detection]:
        detections = self.script[self.calls] if self.calls < len(self.script) else []
        self.calls += 1
        return list(detections)


def test_task_policies_may_differ_between_occupancy_and_entry():
    policy = TaskPolicy(
        occupancy_threshold=0.7,
        tracker_high_threshold=0.5,
        tracker_low_threshold=0.3,
    )
    assert policy.occupancy_threshold != policy.tracker_high_threshold
    assert policy.uses_low_confidence


def test_policy_refuses_low_above_high():
    with pytest.raises(DualPathError):
        TaskPolicy(tracker_high_threshold=0.4, tracker_low_threshold=0.6)


def test_occupancy_branch_uses_only_its_own_threshold():
    """0.5짜리 상자는 occupancy에 들어가지 않지만 tracker에는 간다."""
    counter = DualPathCounter(
        policy=TaskPolicy(
            occupancy_threshold=0.7,
            tracker_high_threshold=0.5,
            tracker_low_threshold=0.3,
        ),
        scene=SCENE,
    )
    result = counter.process_detections(
        0, 0.0, [person(100.0, 0.9), person(200.0, 0.55), person(300.0, 0.35)]
    )
    assert result.occupancy_count == 1
    assert len(result.tracker_detections) == 3


def test_entry_branch_receives_low_confidence_boxes():
    counter = DualPathCounter(
        policy=TaskPolicy(
            occupancy_threshold=0.7,
            tracker_high_threshold=0.5,
            tracker_low_threshold=0.3,
        ),
        scene=SCENE,
    )
    result = counter.process_detections(0, 0.0, [person(100.0, 0.35)])
    assert [d.confidence for d in result.tracker_detections] == [pytest.approx(0.35)]
    assert result.occupancy_count == 0


def test_shared_policy_keeps_the_two_branches_identical():
    """P0 baseline. 임계값을 같게 두면 A4/A5와 같은 동작이어야 한다."""
    counter = DualPathCounter(
        policy=TaskPolicy(
            occupancy_threshold=0.7,
            tracker_high_threshold=0.7,
            tracker_low_threshold=0.7,
        ),
        scene=SCENE,
    )
    result = counter.process_detections(0, 0.0, [person(100.0, 0.9), person(200.0, 0.5)])
    assert result.occupancy_count == 1
    assert len(result.tracker_detections) == 1


def test_detector_runs_exactly_once_per_frame_for_both_branches():
    """추론을 두 번 돌리면 서비스에서 비용이 그대로 두 배가 된다."""
    frames = 12
    detector = CountingDetector([[person(100.0 + i * 6.0, 0.9)] for i in range(frames)])
    counter = DualPathCounter(
        policy=TaskPolicy(
            occupancy_threshold=0.7,
            tracker_high_threshold=0.5,
            tracker_low_threshold=0.3,
            tracker=TrackerKind.MOTION_AWARE,
        ),
        scene=SCENE,
    )
    occupancy_seen = 0
    tracked_seen = 0
    for index in range(frames):
        result = counter.process_frame(index, index / 25.0, object(), detector)
        occupancy_seen += result.occupancy_count
        tracked_seen += len(result.tracked)

    assert detector.calls == frames
    assert counter.detector_calls == frames
    # 두 갈래가 실제로 값을 냈다. 한쪽이 비어 있으면 공유의 의미가 없다.
    assert occupancy_seen > 0
    assert tracked_seen > 0


def test_cached_detections_do_not_count_as_detector_runs():
    """실험은 캐시된 상자를 재사용한다. 그것이 '한 번만 추론한다'의 실현이다."""
    counter = DualPathCounter(policy=TaskPolicy(), scene=SCENE)
    counter.process_detections(0, 0.0, [person(100.0)])
    assert counter.detector_calls == 0


def test_both_tracker_kinds_are_selectable_from_one_policy_object():
    for kind in (TrackerKind.BASELINE_GREEDY, TrackerKind.MOTION_AWARE):
        counter = DualPathCounter(
            policy=TaskPolicy(tracker=kind, min_hits=1), scene=SCENE
        )
        result = counter.process_detections(0, 0.0, [person(100.0)])
        assert len(result.tracked) == 1


def test_crossing_is_counted_once_per_person_crossing_the_line():
    """정책이 바뀌어도 통과 판정 규칙 자체는 그대로다."""
    counter = DualPathCounter(
        policy=TaskPolicy(
            occupancy_threshold=0.7,
            tracker_high_threshold=0.5,
            tracker_low_threshold=0.3,
            tracker=TrackerKind.MOTION_AWARE,
            min_hits=1,
        ),
        scene=SCENE,
    )
    # 오른쪽(x가 큰 쪽)에서 왼쪽으로 걸어간다. IN_DIRECTION=POSITIVE이므로 IN.
    for index in range(40):
        x = 300.0 - index * 6.0
        counter.process_detections(index, index / 25.0, [person(x, 0.9)])
    assert counter.entries == 1
    assert counter.exits == 0
