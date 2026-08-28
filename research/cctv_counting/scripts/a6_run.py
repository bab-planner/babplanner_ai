"""A6 — 과제별 정책과 궤적 연속성이 통과 계수를 살리는가.

A5에서 남은 것은 하나다. **검출을 고쳐도 통과 사건의 절반 이상이 돌아오지
않는다.** 그 절반은 궤적이 끊기는 문제라는 것이 A5의 결론이었고, 여기서
그것을 직접 잰다.

**detector는 시퀀스당 한 번만 돌린다.** 상자를 캐시해 두고 정책 조합은 그
위에서 다시 계산한다. A4/A5와 같은 이유다 — 같은 조합을 여러 번 돌릴 때
결과가 흔들리면 방법 비교가 깨진다. 그리고 이번 단계의 핵심 주장(occupancy와
entry가 **같은 추론**을 공유한다)이 실험 구조에도 그대로 있어야 한다.

    .venv-cctv\\Scripts\\python.exe research\\cctv_counting\\scripts\\a6_run.py ^
      --split DEV --out research\\cctv_counting\\experiments\\a6_continuity\\development.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from a4_run import (  # noqa: E402
    ANCHOR,
    BAND,
    COOLDOWN_SECONDS,
    EVENT_TOLERANCE_SECONDS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    IN_DIRECTION,
    LINE,
    LINE_X,
    MIN_CONFIRM_DISTANCE,
    ROI,
    SAMPLE_EVERY,
    SEQUENCES,
    SOURCE_FPS,
)

from cctv_counting.attribution import attribute, tally  # noqa: E402
from cctv_counting.benchmark import (  # noqa: E402
    crossing_ground_truth,
    occupancy_ground_truth,
    parse_cvml,
)
from cctv_counting.continuity import (  # noqa: E402
    TrajectoryTrace,
    crossing_window,
    gaps_for,
    summarize,
)
from cctv_counting.detection import (  # noqa: E402
    Detection,
    anchor_point,
    normalize_point,
)
from cctv_counting.dualpath import (  # noqa: E402
    DualPathCounter,
    SceneGeometry,
    TaskPolicy,
    TrackerKind,
)
from cctv_counting.geometry import BoundingBox  # noqa: E402
from cctv_counting.metrics import (  # noqa: E402
    TimedEvent,
    match_events,
    occupancy_bias,
    occupancy_mae,
    occupancy_rmse,
)
from cctv_counting.sizeeval import MatchSummary, SizeBin, accumulate  # noqa: E402

SHOP = Path("research/cctv_counting/datasets/raw/caviar")
LOBBY = Path("research/cctv_counting/datasets/raw/caviar_lobby")
CORRIDOR = Path("research/cctv_counting/datasets/raw/caviar_corridor")

#: A5에서 선택된 detector. **A6에서는 고정한다.**
#: 이번 단계의 축은 tracking mechanism이지 detector가 아니다.
DETECTOR_TAG = "E0_mobilenet_R0"
DETECTOR_MODEL = "fasterrcnn_mobilenet_v3_large_fpn"

#: 통과 사건 주변을 얼마나 볼 것인가. **결과를 보기 전에 고정했다.**
#: 25fps에서 ±1초 = ±25프레임. 사람이 걷는 속도에서 선을 넘기 직전과 직후를
#: 담기에 충분하고, 옆 사람의 다른 통과까지 삼키지 않는 크기다.
CROSSING_WINDOW_SECONDS = 1.0

#: A6 FINAL. corridor(along the hallway) 시점, 계열별 두 번째 촬영본 전부.
#: 규칙은 datasets/a6_dataset_gate.md 에서 다운로드 전에 고정했다.
CORRIDOR_SEQUENCES: list[tuple[str, str]] = [
    ("EnterExitCrossingPaths2cor", "ceecp2gt.xml"),
    ("OneLeaveShop2cor", "cols2gt.xml"),
    ("OneLeaveShopReenter2cor", "colsr2gt.xml"),
    ("OneShopOneWait2cor", "cosow2gt.xml"),
    ("OneStopEnter2cor", "cose2gt.xml"),
    ("OneStopMoveEnter2cor", "cosme2gt.xml"),
    ("OneStopMoveNoEnter2cor", "cosmne2gt.xml"),
    ("OneStopNoEnter2cor", "cosne2gt.xml"),
    ("ShopAssistant2cor", "csa2gt.xml"),
    ("ThreePastShop2cor", "c3ps2gt.xml"),
    ("TwoEnterShop2cor", "c2es2gt.xml"),
    ("TwoLeaveShop2cor", "c2ls2gt.xml"),
]

#: occupancy 갈래 후보. 추적과 무관하므로 tracker 설정과 곱하지 않는다.
OCCUPANCY_THRESHOLDS = [0.5, 0.7]

#: **결과를 보기 전에 고정한 정책 행렬.** Cartesian product를 돌리지 않는다.
#: A5의 실패와 직접 연결된 것만 남긴다.
POLICIES: dict[str, TaskPolicy] = {
    # A4/A5가 쓴 그대로. occupancy와 entry가 같은 임계값 0.7을 공유하고,
    # tracker는 greedy + 프레임 기반 소멸이다.
    "P0_baseline": TaskPolicy(
        occupancy_threshold=0.7,
        tracker_high_threshold=0.7,
        tracker_low_threshold=0.7,
        tracker=TrackerKind.BASELINE_GREEDY,
        max_age_frames=30,
        min_hits=1,
    ),
    # 임계값만 갈랐다. tracker는 그대로. 이 줄이 RQ-A6-1의 답이다.
    "P1_dual_threshold": TaskPolicy(
        occupancy_threshold=0.7,
        tracker_high_threshold=0.5,
        tracker_low_threshold=0.3,
        tracker=TrackerKind.BASELINE_GREEDY,
        max_age_frames=30,
        min_hits=1,
    ),
    # 임계값 분리 + 궤적 연속성 tracker. 이 줄이 RQ-A6-3의 답이다.
    "P2_motion": TaskPolicy(
        occupancy_threshold=0.7,
        tracker_high_threshold=0.5,
        tracker_low_threshold=0.3,
        tracker=TrackerKind.MOTION_AWARE,
        max_age_seconds=1.2,
        min_hits=1,
    ),
    # 소멸 시간만 늘린다. 25fps 기준 1.2초 -> 2.0초.
    "P3_motion_long_age": TaskPolicy(
        occupancy_threshold=0.7,
        tracker_high_threshold=0.5,
        tracker_low_threshold=0.3,
        tracker=TrackerKind.MOTION_AWARE,
        max_age_seconds=2.0,
        min_hits=1,
    ),
    # 낮은 신뢰도 구간을 더 넓게 연다. H6를 세게 시험한다.
    "P4_motion_low01": TaskPolicy(
        occupancy_threshold=0.7,
        tracker_high_threshold=0.5,
        tracker_low_threshold=0.1,
        tracker=TrackerKind.MOTION_AWARE,
        max_age_seconds=1.2,
        min_hits=1,
    ),
}

SCENE = SceneGeometry(
    frame_width=FRAME_WIDTH,
    frame_height=FRAME_HEIGHT,
    line=LINE,
    in_direction=IN_DIRECTION,
    roi=ROI,
    anchor=ANCHOR,
    band=BAND,
    min_confirm_distance=MIN_CONFIRM_DISTANCE,
    cooldown_seconds=COOLDOWN_SECONDS,
)


def sequences_for(split: str) -> list[tuple[str, str, Path]]:
    """split 이름을 (시퀀스, GT 파일, 디렉터리) 목록으로 바꾼다."""
    if split == "DEV":
        return [(s, g, SHOP) for s, g, k in SEQUENCES if k == "DEV"]
    if split == "ANALYSIS":
        return [(s, g, SHOP) for s, g, k in SEQUENCES if k == "TEST"]
    if split == "ANALYSIS_LOBBY":
        listing = json.loads(
            (
                Path("research/cctv_counting/experiments/a5_small_person")
                / "final_sequences.json"
            ).read_text(encoding="utf-8")
        )
        return [(row["sequence"], row["gt"], LOBBY) for row in listing["sequences"]]
    if split == "FINAL":
        return [(s, g, CORRIDOR) for s, g in CORRIDOR_SEQUENCES]
    raise SystemExit(f"알 수 없는 split: {split}")


def cache_path(root: Path, sequence: str) -> Path:
    return root / f"{sequence}.{DETECTOR_TAG}.detections.json"


def load_detections(
    root: Path, sequence: str, device: str
) -> tuple[dict[int, list[Detection]], float, int]:
    """캐시가 있으면 읽고, 없으면 **한 번** 돌려서 캐시한다."""
    cache = cache_path(root, sequence)
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return (
            {
                int(k): [
                    Detection(BoundingBox(*b[:4]), b[4], int(b[5])) for b in v
                ]
                for k, v in raw["frames"].items()
            },
            raw["elapsedSeconds"],
            raw["frameCount"],
        )

    import cv2

    from cctv_counting.detectors.torchvision_detector import TorchvisionPersonDetector

    detector = TorchvisionPersonDetector(model_name=DETECTOR_MODEL, device=device)
    capture = cv2.VideoCapture(str(root / f"{sequence}.mpg"))
    frames: dict[int, list[Detection]] = {}
    index = 0
    started = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames[index] = detector.detect(frame)
        index += 1
    elapsed = time.perf_counter() - started
    capture.release()

    cache.write_text(
        json.dumps(
            {
                "sequence": sequence,
                "tag": DETECTOR_TAG,
                "elapsedSeconds": elapsed,
                "frameCount": index,
                "frames": {
                    str(k): [
                        [
                            d.bbox.x1,
                            d.bbox.y1,
                            d.bbox.x2,
                            d.bbox.y2,
                            d.confidence,
                            d.class_id,
                        ]
                        for d in v
                    ]
                    for k, v in frames.items()
                },
            }
        ),
        encoding="utf-8",
    )
    return frames, elapsed, index


def in_roi(box: BoundingBox) -> bool:
    point = normalize_point(anchor_point(box, ANCHOR), FRAME_WIDTH, FRAME_HEIGHT)
    return ROI.contains_normalized(point)


def occupancy_row(detections, gt_frames, threshold: float) -> dict:
    """occupancy 갈래. **추적을 쓰지 않는다.**

    검출 지표(recall, 오탐/frame)를 같이 낸다. A5에서 낮은 MAE가 오차 상쇄
    때문일 수 있다는 것을 확인했으므로, MAE 하나만으로 고르지 않기 위해서다.
    """
    scored = sorted(set(detections) & set(gt_frames))
    summary = MatchSummary()
    for index in scored:
        truth = [
            b.bbox
            for b in gt_frames[index]
            if not b.occluded and in_roi(b.bbox)
        ]
        kept = [
            d.bbox
            for d in detections[index]
            if d.confidence >= threshold and in_roi(d.bbox)
        ]
        accumulate(
            summary, kept, truth, image_width=FRAME_WIDTH, image_height=FRAME_HEIGHT
        )

    gt_occ = occupancy_ground_truth(gt_frames, SAMPLE_EVERY)
    actual, predicted = [], []
    for frame in sorted(gt_occ):
        if frame not in detections:
            continue
        actual.append(gt_occ[frame])
        predicted.append(
            sum(
                1
                for d in detections[frame]
                if d.confidence >= threshold and in_roi(d.bbox)
            )
        )

    return {
        "threshold": threshold,
        "scoredFrames": len(scored),
        "detection": {
            "gtTotal": summary.gt_total,
            "gtMatched": summary.gt_matched,
            "recall": summary.recall,
            "precision": summary.precision,
            "falsePositivesPerFrame": summary.false_positives_per_frame,
            "bySize": {
                b.value: {
                    "total": summary.by_size[b].total,
                    "matched": summary.by_size[b].matched,
                    "recall": summary.by_size[b].recall,
                }
                for b in SizeBin
            },
        },
        "occupancy": {
            "samples": len(actual),
            "mae": occupancy_mae(actual, predicted) if actual else None,
            "rmse": occupancy_rmse(actual, predicted) if actual else None,
            "bias": occupancy_bias(actual, predicted) if actual else None,
        },
    }


def entry_row(detections, gt_frames, name: str, policy: TaskPolicy) -> dict:
    """entry 갈래. 통과 계수 + 궤적 연속성 + 놓친 사건 원인."""
    scored = sorted(set(detections) & set(gt_frames))
    counter = DualPathCounter(policy=policy, scene=SCENE)
    trace = TrajectoryTrace()

    events: list[TimedEvent] = []
    tracker_seconds = 0.0
    for index in scored:
        timestamp = index / SOURCE_FPS
        started = time.perf_counter()
        result = counter.process_detections(index, timestamp, detections[index])
        tracker_seconds += time.perf_counter() - started
        for crossing in result.crossings:
            events.append(
                TimedEvent(
                    timestamp,
                    "IN" if crossing.event.value == "ENTRY" else "OUT",
                )
            )
        truth = [
            (b.track_id, b.bbox) for b in gt_frames[index] if in_roi(b.bbox)
        ]
        trace.observe(
            index,
            truth,
            [(p.track_id, p.bbox) for p in result.tracked],
            [d.bbox for d in result.tracker_detections],
        )

    gt_events = crossing_ground_truth(
        gt_frames,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        line_x_normalized=LINE_X,
        min_confirm_distance=MIN_CONFIRM_DISTANCE,
    )

    crossing: dict[str, dict] = {}
    matched_frames: set[tuple[int, str, int]] = set()
    for direction in ("IN", "OUT"):
        predicted = [e for e in events if e.direction == direction]
        actual_events = [e for e in gt_events if e.direction.value == direction]
        actual = [
            TimedEvent(e.frame_index / SOURCE_FPS, direction) for e in actual_events
        ]
        result = match_events(predicted, actual, EVENT_TOLERANCE_SECONDS)
        # 같은 프레임에 같은 방향 사건이 둘 있을 수 있다(두 사람이 나란히
        # 넘는 경우). 시각으로 대조하면 하나만 맞았는데 둘 다 맞은 것으로
        # 센다. 짝지어진 정답을 **순서대로 하나씩** 소비한다.
        remaining = list(actual_events)
        for _, hit in result.matched:
            for position, event in enumerate(remaining):
                if event.frame_index / SOURCE_FPS == hit.timestamp_seconds:
                    matched_frames.add((event.frame_index, direction, event.track_id))
                    remaining.pop(position)
                    break
        crossing[direction] = {
            "actual": len(actual),
            "predicted": len(predicted),
            "matched": len(result.matched),
            "falsePositives": len(result.false_positives),
            "missed": len(result.missed),
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "countError": abs(len(predicted) - len(actual)),
        }

    # 사건별 창 연속성과, 놓친 사건의 원인.
    max_age = (
        policy.max_age_seconds
        if policy.tracker is TrackerKind.MOTION_AWARE
        else policy.max_age_frames / SOURCE_FPS
    )
    per_event: list[dict] = []
    attributions = []
    for event in gt_events:
        direction = event.direction.value
        window = crossing_window(
            trace,
            gt_frames,
            gt_track_id=event.track_id,
            frame_index=event.frame_index,
            window_seconds=CROSSING_WINDOW_SECONDS,
            fps=SOURCE_FPS,
            frame_width=FRAME_WIDTH,
            line_x_normalized=LINE_X,
            min_confirm_distance=MIN_CONFIRM_DISTANCE,
        )
        recovered = (event.frame_index, direction, event.track_id) in matched_frames
        half = CROSSING_WINDOW_SECONDS * SOURCE_FPS
        overlapping = [
            gap
            for gap in gaps_for(trace, event.track_id, SOURCE_FPS)
            if gap.start_frame <= event.frame_index + half
            and gap.end_frame >= event.frame_index - half
        ]
        row = {
            "frameIndex": event.frame_index,
            "direction": direction,
            "gtTrackId": event.track_id,
            "normalizedHeight": event.normalized_height,
            "matched": recovered,
            "windowFrames": window.window_frames,
            "detectionCoverage": window.detection_coverage,
            "trackCoverage": window.track_coverage,
            "dominantTrackShare": window.dominant_track_share,
            "distinctTrackIds": window.distinct_track_ids,
            "spansBothSides": window.spans_both_sides,
        }
        if not recovered:
            attribution = attribute(window, overlapping, max_age_seconds=max_age)
            row["cause"] = attribution.cause.value
            row["longestGapSeconds"] = attribution.longest_gap_seconds
            attributions.append(attribution)
        per_event.append(row)

    continuity = summarize(trace, SOURCE_FPS)
    return {
        "policy": name,
        "config": policy.as_dict(),
        "scoredFrames": len(scored),
        "crossing": crossing,
        "continuity": {
            "gtTracks": continuity.gt_tracks,
            "coveredTracks": continuity.covered_tracks,
            "totalFragments": continuity.total_fragments,
            "fragmentsPerGtTrack": continuity.fragments_per_gt_track,
            "idSwitches": continuity.id_switches,
            "gaps": continuity.gaps,
            "recoveredGaps": continuity.recovered_gaps,
            "gapRecoveryRate": continuity.gap_recovery_rate,
        },
        "events": per_event,
        "missedCauses": tally(attributions),
        "trackerSeconds": tracker_seconds,
        "trackerMsPerFrame": (
            tracker_seconds * 1000.0 / len(scored) if scored else None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=["DEV", "ANALYSIS", "ANALYSIS_LOBBY", "FINAL"],
        required=True,
    )
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    sequences = sequences_for(args.split)
    rows = []
    for sequence, gt_name, root in sequences:
        detections, elapsed, frame_count = load_detections(
            root, sequence, args.device
        )
        gt_frames = parse_cvml(root / gt_name)
        row = {
            "sequence": sequence,
            "detectorTag": DETECTOR_TAG,
            "detectSeconds": elapsed,
            "detectFps": frame_count / elapsed if elapsed else None,
            "frameCount": frame_count,
            "gtFrames": len(gt_frames),
            "occupancyRows": [
                occupancy_row(detections, gt_frames, t) for t in OCCUPANCY_THRESHOLDS
            ],
            "entryRows": [
                entry_row(detections, gt_frames, name, POLICIES[name])
                for name in args.policies
            ],
        }
        rows.append(row)
        best = row["entryRows"][0]
        print(
            f"  {sequence:28} frames {frame_count:>5}  "
            f"occMAE(0.7) {row['occupancyRows'][1]['occupancy']['mae']:.3f}  "
            f"IN {best['crossing']['IN']['predicted']}/"
            f"{best['crossing']['IN']['actual']}  "
            f"OUT {best['crossing']['OUT']['predicted']}/"
            f"{best['crossing']['OUT']['actual']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "split": args.split,
                "detector": {"tag": DETECTOR_TAG, "model": DETECTOR_MODEL},
                "crossingWindowSeconds": CROSSING_WINDOW_SECONDS,
                "eventToleranceSeconds": EVENT_TOLERANCE_SECONDS,
                "occupancyThresholds": OCCUPANCY_THRESHOLDS,
                "policies": {k: POLICIES[k].as_dict() for k in args.policies},
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n결과: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
