"""A4 — occupancy 오차를 검출 오탐과 누락으로 쪼갠다. 그리고 정지 오탐을 센다.

occupancy 오차는 상자 개수의 차이다. 그 차이는 **없는 사람을 센 것(FP)** 과
**있는 사람을 놓친 것(FN)** 의 합에서 나온다. 둘은 원인도 대책도 다르다.

    bias = (프레임당 FP 평균) - (프레임당 FN 평균)

**임계값은 결과를 보기 전에 정했다.**

    IoU >= 0.5          정답 상자와 검출 상자가 같은 사람이라고 본다
    2.0초 이상 지속     "정지 오탐"으로 본다 (25fps에서 50프레임)

정지 오탐 판정은 **분석일 뿐 필터가 아니다.** "오래 있으면 사람이 아니다"를
파이프라인에 넣지 않는다. 실제로 오래 서 있는 사람이 있기 때문이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cctv_counting.benchmark import parse_cvml
from cctv_counting.detection import Detection, anchor_point, normalize_point
from cctv_counting.geometry import BoundingBox

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a4_run import ANCHOR, FRAME_HEIGHT, FRAME_WIDTH, RAW, ROI, SEQUENCES  # noqa: E402

MATCH_IOU = 0.5
STATIONARY_SECONDS = 2.0
SOURCE_FPS = 25.0


def _kept(detections: list[Detection], threshold: float) -> list[Detection]:
    out = []
    for d in detections:
        if d.class_id != 1 or d.confidence < threshold:
            continue
        point = normalize_point(anchor_point(d.bbox, ANCHOR), FRAME_WIDTH, FRAME_HEIGHT)
        if ROI.contains_normalized(point):
            out.append(d)
    return out


def _greedy_iou_match(pred: list[BoundingBox], truth: list[BoundingBox]):
    """IoU가 큰 쌍부터 확정. 짝 없는 예측은 FP, 짝 없는 정답은 FN."""
    pairs = []
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            iou = p.iou(t)
            if iou >= MATCH_IOU:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_p, used_t = set(), set()
    for _, i, j in pairs:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
    return (
        [i for i in range(len(pred)) if i not in used_p],
        [j for j in range(len(truth)) if j not in used_t],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["DEV", "TEST"], default="TEST")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = []
    for sequence, gt_name, split in SEQUENCES:
        if split != args.split:
            continue
        cache = RAW / f"{sequence}.detections.json"
        if not cache.exists():
            print(f"건너뜀 (검출 캐시 없음): {sequence}")
            continue
        raw = json.loads(cache.read_text(encoding="utf-8"))
        detections = {
            int(k): [Detection(BoundingBox(*b[:4]), b[4], int(b[5])) for b in v]
            for k, v in raw["frames"].items()
        }
        gt_frames = parse_cvml(RAW / gt_name)

        fp_total = fn_total = frames = 0
        # 정지 오탐 추적: 직전 프레임의 미매칭 검출과 겹치면 지속으로 본다.
        streaks: list[tuple[BoundingBox, int]] = []
        stationary_hits = 0
        longest = 0

        for index in sorted(detections):
            truth = [b.bbox for b in gt_frames.get(index, [])]
            truth = [
                t
                for t in truth
                if ROI.contains_normalized(
                    normalize_point(anchor_point(t, ANCHOR), FRAME_WIDTH, FRAME_HEIGHT)
                )
            ]
            pred = [d.bbox for d in _kept(detections[index], args.threshold)]
            fp_idx, fn_idx = _greedy_iou_match(pred, truth)
            fp_total += len(fp_idx)
            fn_total += len(fn_idx)
            frames += 1

            unmatched = [pred[i] for i in fp_idx]
            new_streaks = []
            for box in unmatched:
                best = 0
                for prev_box, length in streaks:
                    if box.iou(prev_box) >= MATCH_IOU:
                        best = max(best, length)
                new_streaks.append((box, best + 1))
            for _, length in new_streaks:
                longest = max(longest, length)
                if length == int(STATIONARY_SECONDS * SOURCE_FPS):
                    stationary_hits += 1
            streaks = new_streaks

        rows.append(
            {
                "sequence": sequence,
                "frames": frames,
                "falsePositivesPerFrame": fp_total / frames,
                "falseNegativesPerFrame": fn_total / frames,
                "impliedBias": (fp_total - fn_total) / frames,
                "stationaryFalsePositiveRuns": stationary_hits,
                "longestFalsePositiveRunFrames": longest,
                "longestFalsePositiveRunSeconds": longest / SOURCE_FPS,
            }
        )
        print(
            f"{sequence:28} FP/frame {rows[-1]['falsePositivesPerFrame']:5.3f}  "
            f"FN/frame {rows[-1]['falseNegativesPerFrame']:5.3f}  "
            f"bias~ {rows[-1]['impliedBias']:+6.3f}  "
            f"정지오탐 {stationary_hits:2d}건  최장 {longest/SOURCE_FPS:5.1f}s"
        )

    total_fp = sum(r["falsePositivesPerFrame"] * r["frames"] for r in rows)
    total_fn = sum(r["falseNegativesPerFrame"] * r["frames"] for r in rows)
    total_frames = sum(r["frames"] for r in rows)
    summary = {
        "split": args.split,
        "threshold": args.threshold,
        "matchIou": MATCH_IOU,
        "stationarySeconds": STATIONARY_SECONDS,
        "thresholdsFixedBeforeResults": True,
        "microFalsePositivesPerFrame": total_fp / total_frames,
        "microFalseNegativesPerFrame": total_fn / total_frames,
        "microImpliedBias": (total_fp - total_fn) / total_frames,
        "sequences": rows,
    }
    print(
        f"\n전체: FP/frame {summary['microFalsePositivesPerFrame']:.3f}, "
        f"FN/frame {summary['microFalseNegativesPerFrame']:.3f}, "
        f"bias~ {summary['microImpliedBias']:+.3f}"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"결과: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
