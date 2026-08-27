"""추적."""

from __future__ import annotations

import pytest

from cctv_counting.tracking import (
    ByteTrackInspiredTracker,
    TrackingError,
    TrackState,
)

from conftest import person


def _tracker(**overrides) -> ByteTrackInspiredTracker:
    defaults = dict(min_hits=2, max_age=3, iou_threshold=0.3)
    defaults.update(overrides)
    return ByteTrackInspiredTracker(**defaults)


def test_no_detections_yields_no_tracks():
    tracker = _tracker()
    assert tracker.update([]) == []
    assert tracker.update([]) == []


def test_track_needs_min_hits_before_it_counts():
    """한 프레임짜리 오탐이 바로 occupancy에 들어오면 안 된다."""
    tracker = _tracker(min_hits=3)
    box = person(100, 100, 160, 300)
    assert tracker.update([box]) == []
    assert tracker.update([box]) == []
    assert len(tracker.update([box])) == 1


def test_same_person_keeps_the_same_id_across_frames():
    tracker = _tracker(min_hits=1)
    first = tracker.update([person(100, 100, 160, 300)])
    second = tracker.update([person(105, 102, 165, 302)])
    assert first[0].track_id == second[0].track_id


def test_two_people_get_two_ids():
    tracker = _tracker(min_hits=1)
    tracked = tracker.update(
        [person(100, 100, 160, 300), person(500, 100, 560, 300)]
    )
    assert len({item.track_id for item in tracked}) == 2


def test_ids_are_deterministic_across_runs():
    """같은 입력이면 같은 번호가 나와야 실험을 재현할 수 있다."""

    def run() -> list[int]:
        tracker = _tracker(min_hits=1)
        ids: list[int] = []
        for offset in range(5):
            tracked = tracker.update(
                [
                    person(100 + offset, 100, 160 + offset, 300),
                    person(500 - offset, 100, 560 - offset, 300),
                ]
            )
            ids.extend(sorted(item.track_id for item in tracked))
        return ids

    assert run() == run()


def test_ids_start_at_one_and_increase():
    tracker = _tracker(min_hits=1)
    assert tracker.next_track_id == 1
    tracker.update([person(100, 100, 160, 300)])
    assert tracker.next_track_id == 2


def test_brief_occlusion_does_not_create_a_new_id():
    """가려진 사람이 새 번호를 받으면 입장 인원이 부풀려진다."""
    tracker = _tracker(min_hits=1, max_age=5)
    first = tracker.update([person(100, 100, 160, 300)])
    tracker.update([])
    tracker.update([])
    again = tracker.update([person(102, 100, 162, 300)])
    assert again[0].track_id == first[0].track_id


def test_track_disappears_after_max_age():
    tracker = _tracker(min_hits=1, max_age=2)
    tracker.update([person(100, 100, 160, 300)])
    for _ in range(4):
        tracker.update([])
    assert tracker.active_track_ids() == set()


def test_low_confidence_box_rescues_a_confirmed_track():
    """가려져 신뢰도가 잠깐 떨어졌다고 track이 끊기면 안 된다. BYTE의 핵심이다."""
    tracker = _tracker(min_hits=1, high_threshold=0.5, low_threshold=0.1)
    first = tracker.update([person(100, 100, 160, 300, confidence=0.9)])
    rescued = tracker.update([person(102, 100, 162, 300, confidence=0.25)])
    assert rescued and rescued[0].track_id == first[0].track_id


def test_low_confidence_box_alone_never_starts_a_track():
    """약한 상자만으로 새 사람을 만들면 오탐이 그대로 인원이 된다."""
    tracker = _tracker(min_hits=1, high_threshold=0.5, low_threshold=0.1)
    assert tracker.update([person(100, 100, 160, 300, confidence=0.2)]) == []
    assert tracker.active_track_ids() == set()


def test_boxes_below_low_threshold_are_ignored_entirely():
    tracker = _tracker(min_hits=1, high_threshold=0.5, low_threshold=0.3)
    tracker.update([person(100, 100, 160, 300, confidence=0.9)])
    tracker.update([person(102, 100, 162, 300, confidence=0.05)])
    assert tracker.update([]) == []


def test_distant_box_does_not_steal_an_existing_track():
    tracker = _tracker(min_hits=1)
    first = tracker.update([person(100, 100, 160, 300)])
    second = tracker.update([person(800, 100, 860, 300)])
    assert second[0].track_id != first[0].track_id


def test_reset_clears_ids_between_videos():
    """track 번호가 영상 경계를 넘으면 그것은 사람을 가리키는 식별자가 된다."""
    tracker = _tracker(min_hits=1)
    tracker.update([person(100, 100, 160, 300)])
    tracker.reset()
    assert tracker.active_track_ids() == set()
    assert tracker.next_track_id == 1
    fresh = tracker.update([person(700, 100, 760, 300)])
    assert fresh[0].track_id == 1


def test_only_confirmed_tracks_are_returned():
    tracker = _tracker(min_hits=5)
    tracker.update([person(100, 100, 160, 300)])
    assert tracker.update([person(101, 100, 161, 300)]) == []


def test_returned_person_exposes_center():
    tracker = _tracker(min_hits=1)
    tracked = tracker.update([person(100, 100, 200, 300)])
    assert tracked[0].center_x == pytest.approx(150.0)
    assert tracked[0].center_y == pytest.approx(200.0)
    assert tracked[0].state is TrackState.CONFIRMED


def test_rejects_low_threshold_above_high_threshold():
    with pytest.raises(TrackingError):
        ByteTrackInspiredTracker(high_threshold=0.3, low_threshold=0.6)


def test_rejects_zero_max_age():
    with pytest.raises(TrackingError):
        ByteTrackInspiredTracker(max_age=0)


def test_rejects_out_of_range_iou():
    with pytest.raises(TrackingError):
        ByteTrackInspiredTracker(iou_threshold=1.5)


def test_the_tracker_does_not_claim_to_be_bytetrack():
    """이름이 구현 사실과 어긋나면 나중에 나올 숫자가 잘못 읽힌다.

    이 tracker는 ByteTrack의 2단계 연결 아이디어를 빌렸을 뿐, 원 구현을
    옮기거나 알고리즘을 재현한 것이 아니다. 이름과 문서가 그렇게 말하고
    있는지를 값으로 고정한다.
    """
    name = ByteTrackInspiredTracker.__name__
    assert "Inspired" in name

    doc = ByteTrackInspiredTracker.__doc__ or ""
    assert "ByteTrack 자체가 아니다" in doc


def test_the_module_documents_what_was_simplified():
    """무엇을 단순화했는지가 코드 옆에 남아 있어야 한다."""
    import cctv_counting.tracking as tracking

    doc = tracking.__doc__ or ""
    assert "ByteTrack 구현이 아니다" in doc
    for simplification in ("Kalman", "Hungarian", "greedy"):
        assert simplification in doc


def test_no_kalman_or_hungarian_is_actually_used():
    """문서와 구현이 어긋나지 않는지 확인한다.

    문서 문자열에는 "Kalman을 쓰지 않는다"라고 적혀 있으므로 본문 검색으로는
    구분할 수 없다. 구문 트리에서 실제로 참조된 이름만 모아서 본다.
    """
    import ast
    from pathlib import Path

    import cctv_counting.tracking as tracking

    tree = ast.parse(Path(tracking.__file__).read_text(encoding="utf-8"))
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, ast.Import):
            referenced.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            referenced.add(node.module.split(".")[0])

    for forbidden in ("KalmanFilter", "linear_sum_assignment", "scipy", "numpy"):
        assert forbidden not in referenced
