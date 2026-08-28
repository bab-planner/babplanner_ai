"""통과 사건을 놓치는 이유가 사람이 작아서인지 확인한다 (A5 §35, H4).

A4에서 통과 recall이 0.4대였다. 원인 후보는 여럿이다. 검출 누락일 수도,
연결 끊김일 수도, 선 근처 궤적 문제일 수도 있다. "작아서 못 봤다"고
말하려면 세 단계가 모두 이어져야 한다.

    1. 놓친 통과가 작은 사람에 몰려 있다
    2. 더 나은 detector가 바로 그 사람들의 검출을 회복한다
    3. 회복된 검출이 실제로 통과 사건까지 회복시킨다

1만 보이고 3이 없으면 "작은 사람이 문제"가 아니라 "작은 사람은 원래
어렵다"는 동어반복이다. 세 단계를 각각 수치로 낸다.

이 분석은 ANALYSIS set(A4에서 이미 본 시퀀스)에서 한다. FINAL set을 여기에
쓰면 그 집합은 더 이상 손대지 않은 test가 아니게 된다.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from a5_run import (  # noqa: E402
    ANCHOR,
    BAND,
    CANDIDATES,
    COOLDOWN_SECONDS,
    EVENT_TOLERANCE_SECONDS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    IN_DIRECTION,
    LINE,
    LINE_X,
    MIN_CONFIRM_DISTANCE,
    SOURCE_FPS,
    THRESHOLD,
    detect_sequence,
    sequences_for,
)
from cctv_counting.benchmark import (  # noqa: E402
    crossing_ground_truth,
    parse_cvml,
)
from cctv_counting.crossing import LineCrossingCounter  # noqa: E402
from cctv_counting.detection import anchor_point, normalize_point  # noqa: E402
from cctv_counting.sizeeval import (  # noqa: E402
    MATCH_IOU,
    SizeBin,
    match_frame,
    size_bin,
)
from cctv_counting.tracking import ByteTrackInspiredTracker  # noqa: E402


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def crossing_events(detections: dict, frames: list[int]) -> list[tuple[float, str, float]]:
    """예측 통과 사건. (초, 방향, 그 순간 상자의 정규화 높이).

    높이를 함께 뽑는 이유는 예측된 사건과 정답 사건을 크기별로 비교하기
    위해서다. 사건을 만든 track의 마지막 상자 높이를 쓴다.
    """
    tracker = ByteTrackInspiredTracker(min_hits=1, max_age=30)
    counter = LineCrossingCounter(
        line=LINE, in_direction=IN_DIRECTION, band=BAND,
        min_confirm_distance=MIN_CONFIRM_DISTANCE, cooldown_seconds=COOLDOWN_SECONDS,
    )
    out: list[tuple[float, str, float]] = []
    for index in frames:
        kept = [d for d in detections[index] if d.confidence >= THRESHOLD]
        tracked = tracker.update(kept)
        heights = {p.track_id: p.bbox.height / FRAME_HEIGHT for p in tracked}
        positions = {
            p.track_id: normalize_point(
                anchor_point(p.bbox, ANCHOR), FRAME_WIDTH, FRAME_HEIGHT
            )
            for p in tracked
        }
        for crossing in counter.update(index, index / SOURCE_FPS, positions):
            out.append((
                index / SOURCE_FPS,
                "IN" if crossing.event.value == "ENTRY" else "OUT",
                heights.get(crossing.track_id, 0.0),
            ))
        counter.forget(tracker.active_track_ids())
    return out


def match_by_time(
    predicted: list[tuple[float, str, float]],
    truth: list,
    tolerance: float,
) -> tuple[set[int], set[int]]:
    """방향이 같고 시간이 가까운 쌍을 1:1로 잇는다. (맞힌 정답 index, 쓰인 예측 index)."""
    pairs = []
    for pi, (pt, pd, _) in enumerate(predicted):
        for ti, event in enumerate(truth):
            if event.direction.value != pd:
                continue
            gap = abs(pt - event.frame_index / SOURCE_FPS)
            if gap <= tolerance:
                pairs.append((gap, pi, ti))
    pairs.sort()
    used_p: set[int] = set()
    used_t: set[int] = set()
    for _, pi, ti in pairs:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
    return used_t, used_p


def track_detection_recall(
    detections: dict,
    gt_frames: dict,
    track_id: int,
    scored: list[int],
) -> tuple[int, int]:
    """정답 track 하나가 얼마나 검출되었는지. (맞힌 프레임, 볼 수 있었던 프레임)."""
    total = matched_count = 0
    for index in scored:
        boxes = [
            b for b in gt_frames.get(index, [])
            if not b.occluded
        ]
        target = [b for b in boxes if b.track_id == track_id]
        if not target:
            continue
        total += 1
        kept = [d.bbox for d in detections[index] if d.confidence >= THRESHOLD]
        matches, _, _ = match_frame(kept, [b.bbox for b in target], MATCH_IOU)
        matched_count += len(matches)
    return matched_count, total


def analyse(sequence: str, gt_name: str, root: Path, detectors: dict) -> dict:
    gt_frames = parse_cvml(root / gt_name)
    truth = crossing_ground_truth(
        gt_frames, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT,
        line_x_normalized=LINE_X, min_confirm_distance=MIN_CONFIRM_DISTANCE,
    )

    per_candidate: dict[str, dict] = {}
    hit_by_candidate: dict[str, set[int]] = {}
    detections_by_candidate: dict[str, dict] = {}
    scored_frames: list[int] = []

    for tag, detector in detectors.items():
        detections, *_ = detect_sequence(root, sequence, tag, detector, f"{sequence}.mpg")
        scored = sorted(set(detections) & set(gt_frames))
        scored_frames = scored
        detections_by_candidate[tag] = detections

        predicted = crossing_events(detections, scored)
        hit, _ = match_by_time(predicted, truth, EVENT_TOLERANCE_SECONDS)
        hit_by_candidate[tag] = hit

        matched_h = [truth[i].normalized_height for i in sorted(hit)]
        missed_h = [
            e.normalized_height for i, e in enumerate(truth) if i not in hit
        ]
        by_size = {b.value: {"total": 0, "matched": 0} for b in SizeBin}
        for i, event in enumerate(truth):
            bucket = by_size[size_bin(event.normalized_height).value]
            bucket["total"] += 1
            if i in hit:
                bucket["matched"] += 1

        per_candidate[tag] = {
            "gtEvents": len(truth),
            "predictedEvents": len(predicted),
            "matched": len(hit),
            "recall": len(hit) / len(truth) if truth else None,
            "medianMatchedHeight": median(matched_h),
            "medianMissedHeight": median(missed_h),
            "matchedHeights": [round(h, 4) for h in matched_h],
            "missedHeights": [round(h, 4) for h in missed_h],
            "bySize": {
                k: {**v, "recall": (v["matched"] / v["total"]) if v["total"] else None}
                for k, v in by_size.items()
            },
        }

    # --- H4 2단계와 3단계 -------------------------------------------------
    # baseline이 놓친 사건만 본다. 그 사건을 만든 사람의 검출이 대조군에서
    # 살아났는지, 그리고 사건까지 살아났는지를 각각 센다.
    chain = None
    tags = list(detectors)
    if len(tags) == 2:
        base, contrast = tags
        missed_by_base = [
            i for i in range(len(truth)) if i not in hit_by_candidate[base]
        ]
        recovered = [
            i for i in missed_by_base if i in hit_by_candidate[contrast]
        ]
        rows = []
        for i in missed_by_base:
            event = truth[i]
            b_hit, b_total = track_detection_recall(
                detections_by_candidate[base], gt_frames, event.track_id, scored_frames
            )
            c_hit, c_total = track_detection_recall(
                detections_by_candidate[contrast], gt_frames, event.track_id,
                scored_frames,
            )
            rows.append({
                "trackId": event.track_id,
                "frame": event.frame_index,
                "direction": event.direction.value,
                "normalizedHeight": round(event.normalized_height, 4),
                "baseTrackRecall": (b_hit / b_total) if b_total else None,
                "contrastTrackRecall": (c_hit / c_total) if c_total else None,
                "eventRecovered": i in hit_by_candidate[contrast],
            })
        chain = {
            "baseline": base,
            "contrast": contrast,
            "missedByBaseline": len(missed_by_base),
            "eventsRecoveredByContrast": len(recovered),
            "rows": rows,
        }

    return {
        "sequence": sequence,
        "gtEvents": len(truth),
        "candidates": per_candidate,
        "chain": chain,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["DEV", "ANALYSIS"], default="ANALYSIS",
                        help="FINAL은 쓰지 않는다. 여기서 보면 test가 아니게 된다.")
    parser.add_argument("--baseline", default="E0_mobilenet_R0")
    parser.add_argument("--contrast", default="E3_resnet50_R0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from cctv_counting.detectors.torchvision_detector import TorchvisionPersonDetector

    detectors = {}
    for tag in (args.baseline, args.contrast):
        spec = CANDIDATES[tag]
        detectors[tag] = TorchvisionPersonDetector(
            model_name=spec["model"], device=args.device,
            min_size=spec["min"], max_size=spec["max"],
        )

    results = []
    for sequence, gt_name, root in sequences_for(args.split):
        print(f"  {sequence} ...", flush=True)
        results.append(analyse(sequence, gt_name, root, detectors))

    payload = {
        "split": args.split,
        "threshold": THRESHOLD,
        "eventToleranceSeconds": EVENT_TOLERANCE_SECONDS,
        "baseline": args.baseline,
        "contrast": args.contrast,
        "note": (
            "H4는 세 단계가 모두 이어질 때만 성립한다. "
            "1) 놓친 사건이 작은 사람에 몰림 "
            "2) 대조 detector가 그 사람 검출을 회복 "
            "3) 회복된 검출이 통과 사건까지 회복"
        ),
        "sequences": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"기록: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
