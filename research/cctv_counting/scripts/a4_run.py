"""A4 — CAVIAR 시퀀스에서 occupancy / crossing을 평가한다.

**detector를 시퀀스당 한 번만 돌린다.** 상자를 모두 캐시해 두고, 임계값과
계수 방법 조합은 그 위에서 다시 계산한다. 이유가 둘이다.

1. 같은 조합을 여러 번 돌릴 때 detector 결과가 완전히 같아야 방법 간 비교가
   깨끗하다. 두 번 돌려서 미세하게 달라지면 그 차이가 방법 차이로 보인다.
2. detector가 전체 시간의 대부분이다. 조합마다 다시 돌리면 낭비다.

`score_floor`가 캐시 임계값보다 낮으므로, 캐시에서 더 높은 임계값으로 거르는
것은 처음부터 그 임계값으로 돌린 것과 정확히 같다.

    .venv-cctv\\Scripts\\python.exe research\\cctv_counting\\scripts\\a4_run.py ^
      --split DEV --out research\\cctv_counting\\experiments\\a4_cross_video\\dev_metrics.json
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

from cctv_counting.benchmark import (
    crossing_ground_truth,
    occupancy_ground_truth,
    parse_cvml,
)
from cctv_counting.confirmation import TemporalConfirmationCounter
from cctv_counting.crossing import InDirection, LineCrossingCounter
from cctv_counting.detection import Anchor, Detection, anchor_point, normalize_point
from cctv_counting.geometry import BoundingBox, Line, NormalizedPoint, NormalizedRect
from cctv_counting.metrics import (
    TimedEvent,
    match_events,
    occupancy_bias,
    occupancy_mae,
    occupancy_rmse,
)
from cctv_counting.tracking import ByteTrackInspiredTracker

RAW = Path("research/cctv_counting/datasets/raw/caviar")

#: 계열당 하나, 알파벳 순. 앞 5개 DEV, 나머지 8개 TEST.
#: 이 목록과 분할은 datasets/a4_dataset_gate.md 에서 고정했다.
SEQUENCES: list[tuple[str, str, str]] = [
    ("EnterExitCrossingPaths1front", "feecp1gt.xml", "DEV"),
    ("OneLeaveShop1front", "fols1gt.xml", "DEV"),
    ("OneLeaveShopReenter1front", "folsr1gt.xml", "DEV"),
    ("OneShopOneWait1front", "fosow1gt.xml", "DEV"),
    ("OneStopEnter1front", "fose1gt.xml", "DEV"),
    ("OneStopMoveEnter1front", "fosme1gt.xml", "TEST"),
    ("OneStopMoveNoEnter1front", "fosmne1gt.xml", "TEST"),
    ("OneStopNoEnter1front", "fosne1gt.xml", "TEST"),
    ("ShopAssistant1front", "fsa1gt.xml", "TEST"),
    ("ThreePastShop1front", "f3ps1gt.xml", "TEST"),
    ("TwoEnterShop1front", "f2es1gt.xml", "TEST"),
    ("TwoLeaveShop1front", "f2ls1gt.xml", "TEST"),
    ("WalkByShop1front", "fwbs1gt.xml", "TEST"),
]

FRAME_WIDTH, FRAME_HEIGHT = 384, 288
SOURCE_FPS = 25.0

#: 정답 발 위치 분포(y 0.234~0.892)에서 바깥으로 반올림해 잡았다.
#: CAVIAR는 통로 배우만 annotation하므로, 그 범위 밖 검출은 정답과 비교할
#: 대상이 없다. 예측을 보고 정한 값이 아니다.
ROI = NormalizedRect(0.0, 0.20, 1.0, 0.95)

#: 세로선. 시퀀스마다 손으로 고르지 않는다. 전부 화면 중앙.
LINE_X = 0.5
LINE = Line(NormalizedPoint(LINE_X, 0.20), NormalizedPoint(LINE_X, 0.95))
IN_DIRECTION = InDirection.POSITIVE  # 오른쪽 -> 왼쪽이 IN
MIN_CONFIRM_DISTANCE = 0.03
BAND = 0.01
COOLDOWN_SECONDS = 0.5

#: 모든 시퀀스에 같은 표본 간격. 25fps에서 1초에 한 번.
SAMPLE_EVERY = 25

#: A3에서 고정한 값을 그대로 쓴다. 정답 간격 1초 + 확정 지연.
EVENT_TOLERANCE_SECONDS = 1.5

ANCHOR = Anchor.BOTTOM_CENTER


def _cache_path(sequence: str) -> Path:
    return RAW / f"{sequence}.detections.json"


def detect_sequence(sequence: str, detector, device: str) -> tuple[dict[int, list[Detection]], float, int]:
    """시퀀스 전체를 훑어 사람 상자를 캐시한다. (상자, 소요초, 프레임수)"""
    import cv2

    cache = _cache_path(sequence)
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        boxes = {
            int(k): [
                Detection(BoundingBox(*b[:4]), b[4], int(b[5])) for b in v
            ]
            for k, v in raw["frames"].items()
        }
        return boxes, raw["elapsedSeconds"], raw["frameCount"]

    capture = cv2.VideoCapture(str(RAW / f"{sequence}.mpg"))
    frames: dict[int, list[Detection]] = {}
    index = 0
    started = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames[index] = detector.detect(frame)
        index += 1
    capture.release()
    elapsed = time.perf_counter() - started

    cache.write_text(
        json.dumps(
            {
                "sequence": sequence,
                "device": device,
                "elapsedSeconds": elapsed,
                "frameCount": index,
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
    return frames, elapsed, index


def _kept(detections: list[Detection], threshold: float) -> list[Detection]:
    """클래스 / 신뢰도 / ROI 필터. filter_detections와 같은 규칙이다."""
    out = []
    for d in detections:
        if d.class_id != 1 or d.confidence < threshold:
            continue
        point = normalize_point(anchor_point(d.bbox, ANCHOR), FRAME_WIDTH, FRAME_HEIGHT)
        if ROI.contains_normalized(point):
            out.append(d)
    return out


def count_occupancy(
    frames: dict[int, list[Detection]],
    threshold: float,
    method: str,
    min_hits: int,
    max_age: int,
    confirm_n: int,
) -> tuple[dict[int, int], float]:
    """프레임별 인원수와, detector를 뺀 계수 단계의 소요 시간."""
    tracker = confirmer = None
    if method == "tracking":
        tracker = ByteTrackInspiredTracker(min_hits=min_hits, max_age=max_age)
    elif method == "confirmation":
        confirmer = TemporalConfirmationCounter(min_observations=confirm_n)

    counts: dict[int, int] = {}
    started = time.perf_counter()
    for index in sorted(frames):
        kept = _kept(frames[index], threshold)
        if tracker is not None:
            counts[index] = len(tracker.update(kept))
        elif confirmer is not None:
            counts[index] = len(confirmer.update(kept))
        else:
            counts[index] = len(kept)
    return counts, time.perf_counter() - started


def count_crossings(
    frames: dict[int, list[Detection]],
    threshold: float,
    min_hits: int,
    max_age: int,
) -> list[TimedEvent]:
    """추적 + 선 통과. 입장 계수는 이 경로만 할 수 있다."""
    tracker = ByteTrackInspiredTracker(min_hits=min_hits, max_age=max_age)
    counter = LineCrossingCounter(
        line=LINE,
        in_direction=IN_DIRECTION,
        band=BAND,
        min_confirm_distance=MIN_CONFIRM_DISTANCE,
        cooldown_seconds=COOLDOWN_SECONDS,
    )
    events: list[TimedEvent] = []
    for index in sorted(frames):
        kept = _kept(frames[index], threshold)
        tracked = tracker.update(kept)
        positions = {
            person.track_id: normalize_point(
                anchor_point(person.bbox, ANCHOR), FRAME_WIDTH, FRAME_HEIGHT
            )
            for person in tracked
        }
        timestamp = index / SOURCE_FPS
        for crossing in counter.update(index, timestamp, positions):
            events.append(TimedEvent(timestamp, crossing.event.value.replace("ENTRY", "IN").replace("EXIT", "OUT")))
        counter.forget(tracker.active_track_ids())
    return events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["DEV", "TEST"], required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7])
    parser.add_argument("--min-hits", type=int, nargs="+", default=[3])
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--confirm", type=int, nargs="+", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from cctv_counting.detectors.torchvision_detector import TorchvisionPersonDetector

    detector = TorchvisionPersonDetector(
        model_name="fasterrcnn_mobilenet_v3_large_fpn", device=args.device
    )
    print("detector:", detector.describe())

    sequences = [s for s in SEQUENCES if s[2] == args.split]
    results: list[dict] = []

    for sequence, gt_name, _ in sequences:
        gt_frames = parse_cvml(RAW / gt_name)
        gt_occ = occupancy_ground_truth(gt_frames, SAMPLE_EVERY)
        gt_events = crossing_ground_truth(
            gt_frames,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            line_x_normalized=LINE_X,
            min_confirm_distance=MIN_CONFIRM_DISTANCE,
        )
        detections, detect_seconds, frame_count = detect_sequence(
            sequence, detector, args.device
        )
        print(f"\n=== {sequence} ({frame_count} frames, {detect_seconds:.1f}s detect) ===")

        for threshold in args.thresholds:
            configs = [("detection_only", 0, 0)]
            configs += [("confirmation", 0, n) for n in args.confirm]
            configs += [("tracking", h, 0) for h in args.min_hits]

            for method, min_hits, confirm_n in configs:
                counts, count_seconds = count_occupancy(
                    detections, threshold, method, min_hits or 3, args.max_age, confirm_n or 3
                )
                actual = [gt_occ[f] for f in sorted(gt_occ) if f in counts]
                predicted = [counts[f] for f in sorted(gt_occ) if f in counts]
                row = {
                    "sequence": sequence,
                    "frames": frame_count,
                    "threshold": threshold,
                    "method": method,
                    "minHits": min_hits or None,
                    "confirmN": confirm_n or None,
                    "samples": len(actual),
                    "gtMean": sum(actual) / len(actual),
                    "predMean": sum(predicted) / len(predicted),
                    "mae": occupancy_mae(actual, predicted),
                    "rmse": occupancy_rmse(actual, predicted),
                    "bias": occupancy_bias(actual, predicted),
                    "absErrors": [abs(a - p) for a, p in zip(actual, predicted)],
                    "detectSeconds": detect_seconds,
                    "countSeconds": count_seconds,
                    "detectFps": frame_count / detect_seconds if detect_seconds else None,
                    "countMsPerFrame": count_seconds / frame_count * 1000.0,
                }

                if method == "tracking":
                    predicted_events = count_crossings(
                        detections, threshold, min_hits or 3, args.max_age
                    )
                    for direction in ("IN", "OUT"):
                        p = [e for e in predicted_events if e.direction == direction]
                        a = [
                            TimedEvent(e.frame_index / SOURCE_FPS, direction)
                            for e in gt_events
                            if e.direction.value == direction
                        ]
                        m = match_events(p, a, EVENT_TOLERANCE_SECONDS)
                        row[f"crossing{direction}"] = {
                            "actual": len(a),
                            "predicted": len(p),
                            "matched": len(m.matched),
                            "falsePositives": len(m.false_positives),
                            "missed": len(m.missed),
                            "precision": m.precision,
                            "recall": m.recall,
                            "f1": m.f1,
                            "countError": abs(len(p) - len(a)),
                        }
                results.append(row)
                tag = f"{method}" + (f"(min_hits={min_hits})" if min_hits else "") + (f"(N={confirm_n})" if confirm_n else "")
                print(f"  conf={threshold} {tag:26} MAE {row['mae']:6.3f} bias {row['bias']:+6.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"split": args.split, "rows": results}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"\n결과: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
