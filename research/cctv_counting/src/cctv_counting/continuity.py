"""궤적 연속성 지표.

**왜 새 지표가 필요한가.** A5까지의 지표는 프레임 단위(검출 recall, occupancy
MAE)이거나 사건 단위(통과 precision/recall)였다. 둘 사이가 비어 있다. A5는
"검출을 전부 고쳤는데 통과 사건은 43%만 살아났다"에서 멈췄는데, 그 사이에서
무슨 일이 일어나는지 잴 도구가 없었기 때문이다.

여기 있는 것은 그 사이를 재는 지표다.

    fragmentation        정답 한 사람이 예측 track 몇 개로 쪼개지는가
    id switch            그 사람을 가리키던 번호가 몇 번 바뀌는가
    gap recovery         끊긴 뒤 **같은 번호로** 돌아오는 비율
    crossing window      선을 넘는 ±W초 동안 검출/추적이 얼마나 이어지는가

**MOT benchmark 전체를 구현하지 않는다.** MOTA/IDF1까지 가면 이 단계의
질문과 상관없는 숫자가 늘어난다. 계수와 직접 연결되는 것만 둔다.

**정답 대응은 전역 1:1이다.** greedy로 붙이면 두 사람이 겹칠 때 한 예측이
두 정답에 붙어 연속성이 실제보다 좋아 보인다. `assignment.py`를 쓴다.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from cctv_counting.assignment import linear_sum_assignment
from cctv_counting.benchmark import GroundTruthBox
from cctv_counting.geometry import BoundingBox


class ContinuityError(ValueError):
    """연속성 지표를 계산할 수 없다."""


#: 정답 상자와 예측 상자를 같은 사람으로 볼 최소 IoU. A5의 검출 채점과 같다.
DEFAULT_IOU_THRESHOLD = 0.5


def associate(
    truth: list[tuple[int, BoundingBox]],
    predicted: list[tuple[int, BoundingBox]],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[int, int]:
    """한 프레임의 (정답 id -> 예측 id) 대응. 전역 1:1이다.

    임계값 미만은 짝으로 인정하지 않는다. 한 예측이 두 정답에 붙는 일은
    구조적으로 일어나지 않는다.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ContinuityError(f"iou_threshold는 0~1이어야 합니다: {iou_threshold}")
    if not truth or not predicted:
        return {}
    ious = [[t.iou(p) for _, p in predicted] for _, t in truth]
    pairs = linear_sum_assignment([[1.0 - value for value in row] for row in ious])
    return {
        truth[r][0]: predicted[c][0]
        for r, c in pairs
        if ious[r][c] >= iou_threshold and ious[r][c] > 0.0
    }


@dataclass
class TrajectoryTrace:
    """시퀀스 하나에 대한 정답별 추적 이력.

    프레임 순서를 기억한다. 채점 프레임이 듬성듬성해도(frame_stride) 정답이
    있는 프레임만 이어 붙여 판단한다.
    """

    #: 정답 track이 존재한 프레임(오름차순).
    present: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    #: 정답 track이 예측 track에 덮인 프레임 -> 예측 track id.
    covered: dict[int, dict[int, int]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    #: 정답 track이 **원시 검출**에 덮인 프레임. 추적과 무관하다.
    detected: dict[int, set[int]] = field(default_factory=lambda: defaultdict(set))

    def observe(
        self,
        frame_index: int,
        truth: list[tuple[int, BoundingBox]],
        tracked: list[tuple[int, BoundingBox]],
        detections: list[BoundingBox],
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        """프레임 하나를 기록한다."""
        for track_id, _ in truth:
            self.present[track_id].append(frame_index)

        mapping = associate(truth, tracked, iou_threshold)
        for gt_id, predicted_id in mapping.items():
            self.covered[gt_id][frame_index] = predicted_id

        detection_mapping = associate(
            truth,
            list(enumerate(detections)),
            iou_threshold,
        )
        for gt_id in detection_mapping:
            self.detected[gt_id].add(frame_index)

    def gt_track_ids(self) -> list[int]:
        return sorted(self.present)


def fragments_for(trace: TrajectoryTrace, gt_track_id: int) -> int:
    """정답 한 사람의 궤적이 **끊긴 조각 몇 개**로 덮이는가.

    완전히 덮이면 1이다. 한 번도 덮이지 않으면 0이다. 중간이 끊길 때마다
    늘어난다. 번호가 바뀌는 것과 끊기는 것은 다른 사건이므로, 끊김 없이
    번호만 바뀌는 경우도 조각을 하나 더 센다.
    """
    frames = trace.present.get(gt_track_id, [])
    covered = trace.covered.get(gt_track_id, {})
    fragments = 0
    previous_id: int | None = None
    previous_covered = False
    for frame in frames:
        current = covered.get(frame)
        if current is None:
            previous_covered = False
            previous_id = None
            continue
        if not previous_covered or current != previous_id:
            fragments += 1
        previous_covered = True
        previous_id = current
    return fragments


def id_switches_for(trace: TrajectoryTrace, gt_track_id: int) -> int:
    """그 사람을 가리키던 예측 번호가 바뀐 횟수.

    사이에 끊긴 구간이 있어도 **덮인 프레임끼리** 이어서 비교한다. 끊겼다가
    다른 번호로 돌아오는 것이 통과 계수를 깨뜨리는 바로 그 사건이기 때문이다.
    """
    covered = trace.covered.get(gt_track_id, {})
    switches = 0
    previous: int | None = None
    for frame in sorted(covered):
        current = covered[frame]
        if previous is not None and current != previous:
            switches += 1
        previous = current
    return switches


@dataclass(frozen=True)
class Gap:
    """정답 한 사람이 추적에서 사라졌다가 돌아온 구간 하나."""

    gt_track_id: int
    start_frame: int
    end_frame: int
    frames: int
    seconds: float
    before_id: int
    after_id: int

    @property
    def recovered(self) -> bool:
        """같은 번호로 돌아왔는가."""
        return self.before_id == self.after_id


def gaps_for(
    trace: TrajectoryTrace, gt_track_id: int, fps: float
) -> list[Gap]:
    """덮인 구간 사이의 빈 구간을 모두 찾는다.

    시작 전과 끝난 뒤의 빈 구간은 세지 않는다. 그것은 아직 안 나타났거나
    이미 나간 것이지 **끊긴 것**이 아니다.
    """
    if fps <= 0:
        raise ContinuityError(f"fps는 0보다 커야 합니다: {fps}")
    frames = trace.present.get(gt_track_id, [])
    covered = trace.covered.get(gt_track_id, {})
    result: list[Gap] = []
    pending: list[int] = []
    previous_id: int | None = None
    previous_frame: int | None = None
    for frame in frames:
        current = covered.get(frame)
        if current is None:
            if previous_id is not None:
                pending.append(frame)
            continue
        if pending and previous_id is not None and previous_frame is not None:
            result.append(
                Gap(
                    gt_track_id=gt_track_id,
                    start_frame=previous_frame,
                    end_frame=frame,
                    frames=len(pending),
                    seconds=(frame - previous_frame) / fps,
                    before_id=previous_id,
                    after_id=current,
                )
            )
        pending = []
        previous_id = current
        previous_frame = frame
    return result


@dataclass(frozen=True)
class ContinuitySummary:
    """시퀀스 하나의 연속성 요약."""

    gt_tracks: int
    covered_tracks: int
    total_fragments: int
    id_switches: int
    gaps: int
    recovered_gaps: int

    @property
    def fragments_per_gt_track(self) -> float | None:
        """덮인 정답 track 하나가 평균 몇 조각으로 쪼개지는가. 1.0이 이상적."""
        if self.covered_tracks == 0:
            return None
        return self.total_fragments / self.covered_tracks

    @property
    def gap_recovery_rate(self) -> float | None:
        """끊긴 뒤 같은 번호로 돌아온 비율. 끊김이 없으면 정의되지 않는다."""
        if self.gaps == 0:
            return None
        return self.recovered_gaps / self.gaps


def summarize(trace: TrajectoryTrace, fps: float) -> ContinuitySummary:
    total_fragments = 0
    switches = 0
    gaps = 0
    recovered = 0
    covered_tracks = 0
    for gt_id in trace.gt_track_ids():
        fragments = fragments_for(trace, gt_id)
        if fragments:
            covered_tracks += 1
        total_fragments += fragments
        switches += id_switches_for(trace, gt_id)
        for gap in gaps_for(trace, gt_id, fps):
            gaps += 1
            recovered += 1 if gap.recovered else 0
    return ContinuitySummary(
        gt_tracks=len(trace.present),
        covered_tracks=covered_tracks,
        total_fragments=total_fragments,
        id_switches=switches,
        gaps=gaps,
        recovered_gaps=recovered,
    )


@dataclass(frozen=True)
class CrossingWindow:
    """정답 통과 사건 하나 주변의 연속성.

    **전체 영상 추적 성능보다 이쪽이 제품 과제에 가깝다.** 사람이 화면
    구석에서 얼마나 잘 추적되는지는 통과 계수와 상관이 없다. 선을 넘는
    그 몇 초가 전부다.
    """

    gt_track_id: int
    frame_index: int
    window_frames: int
    detection_coverage: float | None
    track_coverage: float | None
    dominant_track_share: float | None
    distinct_track_ids: int
    spans_both_sides: bool


def crossing_window(
    trace: TrajectoryTrace,
    gt_frames: dict[int, list[GroundTruthBox]],
    *,
    gt_track_id: int,
    frame_index: int,
    window_seconds: float,
    fps: float,
    frame_width: int,
    line_x_normalized: float,
    min_confirm_distance: float,
) -> CrossingWindow:
    """정답 통과 사건 하나의 ±window_seconds 구간을 잰다.

    분모는 **그 사람이 정답에 존재하는 프레임**이다. 사람이 아직 화면에
    없는 프레임을 분모에 넣으면 못 찾은 것처럼 보인다.
    """
    if window_seconds <= 0:
        raise ContinuityError(
            f"window_seconds는 0보다 커야 합니다: {window_seconds}"
        )
    if fps <= 0:
        raise ContinuityError(f"fps는 0보다 커야 합니다: {fps}")

    half = window_seconds * fps
    low = frame_index - half
    high = frame_index + half
    window = [f for f in trace.present.get(gt_track_id, []) if low <= f <= high]

    covered = trace.covered.get(gt_track_id, {})
    detected = trace.detected.get(gt_track_id, set())
    if not window:
        return CrossingWindow(
            gt_track_id=gt_track_id,
            frame_index=frame_index,
            window_frames=0,
            detection_coverage=None,
            track_coverage=None,
            dominant_track_share=None,
            distinct_track_ids=0,
            spans_both_sides=False,
        )

    ids_in_window = [covered[f] for f in window if f in covered]
    counts: dict[int, int] = defaultdict(int)
    for value in ids_in_window:
        counts[value] += 1
    dominant_id: int | None = None
    if counts:
        dominant_id = max(sorted(counts), key=lambda key: counts[key])

    left = right = False
    if dominant_id is not None:
        for frame in window:
            if covered.get(frame) != dominant_id:
                continue
            for box in gt_frames.get(frame, []):
                if box.track_id != gt_track_id:
                    continue
                offset = box.bbox.bottom_center.x / frame_width - line_x_normalized
                if offset <= -min_confirm_distance:
                    left = True
                elif offset >= min_confirm_distance:
                    right = True

    return CrossingWindow(
        gt_track_id=gt_track_id,
        frame_index=frame_index,
        window_frames=len(window),
        detection_coverage=sum(1 for f in window if f in detected) / len(window),
        track_coverage=len(ids_in_window) / len(window),
        dominant_track_share=(
            counts[dominant_id] / len(window) if dominant_id is not None else 0.0
        ),
        distinct_track_ids=len(counts),
        spans_both_sides=left and right,
    )
