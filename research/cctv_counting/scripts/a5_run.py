"""A5 — 해상도 / detector 용량이 작은 사람 검출을 살리는가.

A4에서 occupancy 오차의 절반이 검출 누락이었고, 통과 계수 recall이 0.4까지
떨어진 직접 원인도 누락이었다. 여기서는 그 누락이 **작은 사람에 몰려 있는지**,
그리고 **실효 해상도를 올리거나 더 큰 모델을 쓰면 회복되는지** 본다.

detector를 조건마다 한 번씩만 돌리고 상자를 캐시한다. 같은 조건을 다시
평가할 때 결과가 흔들리면 방법 비교가 깨지기 때문이다.

    .venv-cctv\\Scripts\\python.exe research\\cctv_counting\\scripts\\a5_run.py ^
      --split DEV --out research\\cctv_counting\\experiments\\a5_small_person\\development.json
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
    FRAME_HEIGHT,
    FRAME_WIDTH,
    IN_DIRECTION,
    LINE,
    LINE_X,
    MIN_CONFIRM_DISTANCE,
    ROI,
    SAMPLE_EVERY,
    SOURCE_FPS,
    SEQUENCES,
)

from cctv_counting.benchmark import (  # noqa: E402
    crossing_ground_truth,
    occupancy_ground_truth,
    parse_cvml,
)
from cctv_counting.crossing import LineCrossingCounter  # noqa: E402
from cctv_counting.detection import Detection, anchor_point, normalize_point  # noqa: E402
from cctv_counting.geometry import BoundingBox  # noqa: E402
from cctv_counting.metrics import (  # noqa: E402
    TimedEvent,
    match_events,
    occupancy_bias,
    occupancy_mae,
    occupancy_rmse,
)
from cctv_counting.sizeeval import (  # noqa: E402
    MatchSummary,
    SizeBin,
    accumulate,
    height_histogram,
)
from cctv_counting.tracking import ByteTrackInspiredTracker  # noqa: E402

SHOP = Path("research/cctv_counting/datasets/raw/caviar")
LOBBY = Path("research/cctv_counting/datasets/raw/caviar_lobby")
EVENT_TOLERANCE_SECONDS = 1.5

#: A4에서 dev로 고른 값. A5의 축은 해상도와 모델이므로 임계값은 고정한다.
THRESHOLD = 0.7

#: 실험 조건. **결과를 보기 전에 고정했다.**
#: 완전 조합(cartesian)을 돌리지 않는다. 축 하나씩만 움직인다.
CANDIDATES: dict[str, dict] = {
    "E0_mobilenet_R0": {"model": "fasterrcnn_mobilenet_v3_large_fpn", "min": None, "max": None},
    "E1_mobilenet_R1": {"model": "fasterrcnn_mobilenet_v3_large_fpn", "min": 1200, "max": 2000},
    "E2_mobilenet_R2": {"model": "fasterrcnn_mobilenet_v3_large_fpn", "min": 1600, "max": 2666},
    "E3_resnet50_R0": {"model": "fasterrcnn_resnet50_fpn_v2", "min": None, "max": None},
    "E4_resnet50_R1": {"model": "fasterrcnn_resnet50_fpn_v2", "min": 1200, "max": 2000},
}


def sequences_for(split: str) -> list[tuple[str, str, Path]]:
    if split in {"DEV", "ANALYSIS"}:
        want = "DEV" if split == "DEV" else "TEST"
        return [(s, g, SHOP) for s, g, k in SEQUENCES if k == want]
    if split == "FINAL":
        listing = json.loads(
            Path("research/cctv_counting/experiments/a5_small_person/final_sequences.json")
            .read_text(encoding="utf-8")
        )
        return [(row["sequence"], row["gt"], LOBBY) for row in listing["sequences"]]
    raise SystemExit(f"알 수 없는 split: {split}")


def cache_path(root: Path, sequence: str, tag: str) -> Path:
    return root / f"{sequence}.{tag}.detections.json"


def detect_sequence(root: Path, sequence: str, tag: str, detector, video: str):
    import cv2
    import torch

    cache = cache_path(root, sequence, tag)
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return (
            {
                int(k): [Detection(BoundingBox(*b[:4]), b[4], int(b[5])) for b in v]
                for k, v in raw["frames"].items()
            },
            raw["elapsedSeconds"],
            raw["frameCount"],
            raw.get("peakVramMb"),
        )

    capture = cv2.VideoCapture(str(root / video))
    frames: dict[int, list[Detection]] = {}
    index = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames[index] = detector.detect(frame)
        index += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    capture.release()
    peak = (
        round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        if torch.cuda.is_available()
        else None
    )

    cache.write_text(
        json.dumps(
            {
                "sequence": sequence,
                "tag": tag,
                "elapsedSeconds": elapsed,
                "frameCount": index,
                "peakVramMb": peak,
                "frames": {
                    str(k): [
                        [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2, d.confidence, d.class_id]
                        for d in v
                    ]
                    for k, v in frames.items()
                },
            }
        ),
        encoding="utf-8",
    )
    return frames, elapsed, index, peak


def kept(detections: list[Detection]) -> list[Detection]:
    out = []
    for d in detections:
        if d.class_id != 1 or d.confidence < THRESHOLD:
            continue
        point = normalize_point(anchor_point(d.bbox, ANCHOR), FRAME_WIDTH, FRAME_HEIGHT)
        if ROI.contains_normalized(point):
            out.append(d)
    return out


def in_roi(box: BoundingBox) -> bool:
    point = normalize_point(anchor_point(box, ANCHOR), FRAME_WIDTH, FRAME_HEIGHT)
    return ROI.contains_normalized(point)


def evaluate(detections, gt_frames, sequence: str) -> dict:
    """검출 성능, occupancy, 통과 계수를 한 번에 낸다."""
    # **정답이 있는 프레임만 채점한다.**
    # CAVIAR 로비 세트는 영상보다 annotation이 짧다(대부분 1프레임, 한
    # 시퀀스는 107프레임). 정답이 없는 프레임을 "사람 0명"으로 두면 그
    # 구간의 검출이 전부 오탐으로 세여 precision이 조용히 나빠진다.
    scored = sorted(set(detections) & set(gt_frames))
    summary = MatchSummary()
    for index in scored:
        truth = [
            b.bbox
            for b in gt_frames[index]
            # 가려진 사람은 검출 recall 분모에서 뺀다. 보이지 않는 것을
            # 못 찾은 것을 detector 실패로 세지 않는다.
            if not b.occluded and in_roi(b.bbox)
        ]
        accumulate(
            summary,
            [d.bbox for d in kept(detections[index])],
            truth,
            image_width=FRAME_WIDTH,
            image_height=FRAME_HEIGHT,
        )

    gt_occ = occupancy_ground_truth(gt_frames, SAMPLE_EVERY)
    actual, predicted = [], []
    for frame in sorted(gt_occ):
        if frame not in detections:
            continue
        actual.append(gt_occ[frame])
        predicted.append(len(kept(detections[frame])))

    tracker = ByteTrackInspiredTracker(min_hits=1, max_age=30)
    counter = LineCrossingCounter(
        line=LINE, in_direction=IN_DIRECTION, band=BAND,
        min_confirm_distance=MIN_CONFIRM_DISTANCE, cooldown_seconds=COOLDOWN_SECONDS,
    )
    events: list[TimedEvent] = []
    for index in scored:
        tracked = tracker.update(kept(detections[index]))
        positions = {
            p.track_id: normalize_point(
                anchor_point(p.bbox, ANCHOR), FRAME_WIDTH, FRAME_HEIGHT
            )
            for p in tracked
        }
        for crossing in counter.update(index, index / SOURCE_FPS, positions):
            events.append(
                TimedEvent(
                    index / SOURCE_FPS,
                    "IN" if crossing.event.value == "ENTRY" else "OUT",
                )
            )
        counter.forget(tracker.active_track_ids())

    gt_events = crossing_ground_truth(
        gt_frames, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT,
        line_x_normalized=LINE_X, min_confirm_distance=MIN_CONFIRM_DISTANCE,
    )
    crossing: dict[str, dict] = {}
    for direction in ("IN", "OUT"):
        p = [e for e in events if e.direction == direction]
        a = [
            TimedEvent(e.frame_index / SOURCE_FPS, direction)
            for e in gt_events
            if e.direction.value == direction
        ]
        m = match_events(p, a, EVENT_TOLERANCE_SECONDS)
        crossing[direction] = {
            "actual": len(a), "predicted": len(p), "matched": len(m.matched),
            "falsePositives": len(m.false_positives), "missed": len(m.missed),
            "precision": m.precision, "recall": m.recall, "f1": m.f1,
            "countError": abs(len(p) - len(a)),
        }

    return {
        "sequence": sequence,
        "frames": summary.frames,
        "scoredFrames": len(scored),
        "detectedFrames": len(detections),
        "gtFrames": len(gt_frames),
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
            # 상자별 높이를 그대로 남기면 기록이 수백 KB가 된다.
            # 눈금 0.001 히스토그램이면 이 단계의 어떤 수치도 달라지지 않는다.
            "missedHeightHistogram": height_histogram(summary.missed_heights),
            "matchedHeightHistogram": height_histogram(summary.matched_heights),
        },
        "occupancy": {
            "samples": len(actual),
            "mae": occupancy_mae(actual, predicted) if actual else None,
            "rmse": occupancy_rmse(actual, predicted) if actual else None,
            "bias": occupancy_bias(actual, predicted) if actual else None,
            "absErrors": [abs(a - p) for a, p in zip(actual, predicted)],
        },
        "crossing": crossing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["DEV", "ANALYSIS", "FINAL"], required=True)
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from cctv_counting.detectors.torchvision_detector import TorchvisionPersonDetector

    sequences = sequences_for(args.split)
    results = []

    for tag in args.candidates:
        spec = CANDIDATES[tag]
        detector = TorchvisionPersonDetector(
            model_name=spec["model"], device=args.device,
            min_size=spec["min"], max_size=spec["max"],
        )
        shape = detector.transformed_shape(FRAME_HEIGHT, FRAME_WIDTH)
        print(f"\n### {tag}: {detector.describe()['model']} "
              f"min={detector.effective_min_size} max={detector.effective_max_size} "
              f"-> {FRAME_HEIGHT}x{FRAME_WIDTH} 가 {shape[0]}x{shape[1]} 로 들어감")

        for sequence, gt_name, root in sequences:
            detections, elapsed, frame_count, peak = detect_sequence(
                root, sequence, tag, detector, f"{sequence}.mpg"
            )
            gt_frames = parse_cvml(root / gt_name)
            row = evaluate(detections, gt_frames, sequence)
            row.update({
                "candidate": tag, "model": spec["model"],
                "minSize": detector.effective_min_size,
                "maxSize": detector.effective_max_size,
                "transformedShape": list(shape),
                "detectSeconds": elapsed,
                "detectFps": frame_count / elapsed if elapsed else None,
                "peakVramMb": peak,
                "threshold": THRESHOLD,
            })
            results.append(row)
            d = row["detection"]
            small = d["bySize"]["SMALL"]["recall"]
            print(f"  {sequence:28} recall {d['recall']:.3f}  "
                  f"small {('%.3f'%small) if small is not None else ' N/A '}  "
                  f"FP/f {d['falsePositivesPerFrame']:.3f}  "
                  f"MAE {row['occupancy']['mae']:.3f}  {row['detectFps']:.1f}fps")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"split": args.split, "threshold": THRESHOLD, "rows": results},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n결과: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
