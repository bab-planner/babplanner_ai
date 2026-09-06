"""C4 — 허용 subset에서 사전학습 detector의 검출·occupancy를 잰다.

**모델을 바꾸지 않는다.** torchvision COCO 사전학습 Faster R-CNN 그대로다.
먼저 baseline 숫자를 확보하고, 부족하면 그때 fine-tuning을 검토한다. 숫자를
보기 전에 모델을 고르면 무엇 때문에 좋아졌는지 말할 수 없다.

평가는 C2가 다시 나눈 split만 쓴다. 원본 split은 MOT 시퀀스가 갈려 있어
쓰지 않는다.

**이 결과는 external-public 벤치마크다.** 허용 subset이 COCO뿐이라 감시
카메라 영상이 아니라 웹 사진이고, 급식소 천장 카메라로의 전이를 보장하지
않는다. 이 한계를 산출물에 같이 적는다.

    .venv-waste\\Scripts\\python.exe research\\cctv_counting\\scripts\\c4_occupancy_baseline.py ^
      --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cctv_counting.assignment import match_by_iou  # noqa: E402
from cctv_counting.geometry import BoundingBox  # noqa: E402
from cctv_counting.metrics import score_occupancy  # noqa: E402

ROOT = Path("research/cctv_counting/datasets/raw/crowd_cctv")
PREPARED = ROOT / "prepared" / "annotations.jsonl"
OUT = Path("research/cctv_counting/experiments/c4_occupancy_baseline")

BENCHMARK = "external-public"

#: 상자를 맞다고 볼 최소 겹침. 사람 계수에서 흔히 쓰는 값이고 여기서
#: 새로 고르지 않는다.
IOU_THRESHOLD = 0.5

#: 사람 수를 셀 때 쓰는 신뢰도. 하나를 고르지 않고 곡선을 낸다 — 급식소에
#: 맞는 값은 급식소 영상을 보고 정할 일이지 COCO에서 정할 일이 아니다.
SCORE_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#: 혼잡할수록 무너지는지 보려고 나눈다. 급식소에서 중요한 것은 평균이
#: 아니라 붐빌 때다.
CROWD_BINS = ((0, 0), (1, 2), (3, 5), (6, 10), (11, 10_000))


def bin_label(count: int) -> str:
    for low, high in CROWD_BINS:
        if low <= count <= high:
            return f"{low}" if low == high else (f"{low}-{high}" if high < 10_000 else f"{low}+")
    return "?"


def load_frames(path: Path, split: str | None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [row for row in rows if split is None or row["split"] == split]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=PREPARED)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split", default="test", help="C2가 만든 split. all이면 전부")
    parser.add_argument("--model", default="fasterrcnn_mobilenet_v3_large_fpn")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    if not args.annotations.exists():
        print(f"준비된 주석이 없습니다: {args.annotations}")
        print("먼저 c1_fetch_crowd_cctv.py 와 c2_prepare_external.py 를 실행하세요.")
        return 1

    frames = load_frames(args.annotations, None if args.split == "all" else args.split)
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        print(f"'{args.split}' 에 프레임이 없습니다.")
        return 1

    licenses = {row["licenseClass"] for row in frames}
    if licenses != {"permissive"}:
        # 비상업 데이터가 제품 성능 근거로 새어 들어가는 것을 여기서 막는다.
        print(f"허용되지 않은 라이선스 등급이 섞여 있습니다: {sorted(licenses)}")
        return 1

    import cv2

    from cctv_counting.detectors.torchvision_detector import TorchvisionPersonDetector

    detector = TorchvisionPersonDetector(model_name=args.model, device=args.device)
    print(f"detector: {json.dumps(detector.describe(), ensure_ascii=False)}")
    print(f"split '{args.split}' 프레임 {len(frames)}장  (benchmark: {BENCHMARK})\n")

    truth_counts: list[int] = []
    per_frame: list[dict] = []
    for index, row in enumerate(frames, start=1):
        image = cv2.imread(str(args.root / row["imagePath"]))
        if image is None:
            print(f"이미지를 열 수 없습니다: {row['imagePath']}")
            return 1
        detections = detector.detect(image)
        truth_counts.append(row["personCount"])
        per_frame.append(
            {
                "frameId": row["frameId"],
                "source": row["source"],
                "truth": row["personCount"],
                "boxes": row["boxes"],
                "detections": [
                    {
                        "box": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                        "score": d.confidence,
                    }
                    for d in detections
                ],
            }
        )
        if index % 50 == 0:
            print(f"  {index}/{len(frames)}")

    results = []
    for threshold in SCORE_GRID:
        predicted_counts: list[int] = []
        true_positive = false_positive = false_negative = 0
        by_bin: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for item in per_frame:
            kept = [d for d in item["detections"] if d["score"] >= threshold]
            predicted_counts.append(len(kept))
            by_bin[bin_label(item["truth"])].append((item["truth"], len(kept)))

            truth_boxes = [BoundingBox(*b) for b in item["boxes"]]
            pred_boxes = [BoundingBox(*d["box"]) for d in kept]
            matches, unmatched_truth, unmatched_pred = match_by_iou(
                truth_boxes, pred_boxes, IOU_THRESHOLD
            )
            true_positive += len(matches)
            false_negative += len(unmatched_truth)
            false_positive += len(unmatched_pred)

        occupancy = score_occupancy(truth_counts, predicted_counts)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        )
        results.append(
            {
                "scoreThreshold": threshold,
                "occupancyMae": occupancy.mae,
                "occupancyRmse": occupancy.rmse,
                "occupancyBias": occupancy.bias,
                "boxPrecision": precision,
                "boxRecall": recall,
                "truePositive": true_positive,
                "falsePositive": false_positive,
                "falseNegative": false_negative,
                "byCrowdLevel": {
                    label: {
                        "frames": len(pairs),
                        "meanTruth": sum(t for t, _ in pairs) / len(pairs),
                        "mae": sum(abs(p - t) for t, p in pairs) / len(pairs),
                        "bias": sum(p - t for t, p in pairs) / len(pairs),
                    }
                    for label, pairs in sorted(by_bin.items())
                },
            }
        )

    print(f"{'score':>6} {'MAE':>7} {'Bias':>8} {'precision':>10} {'recall':>8}")
    for row in results:
        print(
            f"{row['scoreThreshold']:6.2f} {row['occupancyMae']:7.3f} "
            f"{row['occupancyBias']:+8.3f} "
            f"{(row['boxPrecision'] or 0):10.3f} {(row['boxRecall'] or 0):8.3f}"
        )

    best = min(results, key=lambda row: row["occupancyMae"])
    print(f"\nMAE 최저: score {best['scoreThreshold']:.2f} → {best['occupancyMae']:.3f}명")
    print("혼잡도별 (MAE 최저 임계값 기준)")
    for label, scores in best["byCrowdLevel"].items():
        print(
            f"  실제 {label:>6}명  프레임 {scores['frames']:5}  "
            f"MAE {scores['mae']:6.3f}  bias {scores['bias']:+6.3f}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(
        json.dumps(
            {
                "benchmark": BENCHMARK,
                "dataset": "crowd_cctv permissive subset (COCO only)",
                "registry": "research/cctv_counting/datasets/c1_crowd_cctv_registry.json",
                "split": args.split,
                "splitSource": "C2 group-wise resplit (제공 split 미사용)",
                "frames": len(frames),
                "sources": dict(Counter(row["source"] for row in frames)),
                "detector": detector.describe(),
                "detectorChanged": False,
                "iouThreshold": IOU_THRESHOLD,
                "results": results,
                "limitation": (
                    "허용 subset이 COCO뿐이라 감시 카메라 영상이 아니다. 웹 사진에서 "
                    "잰 값이며 급식소 천장 카메라로의 전이를 보장하지 않는다."
                ),
                "claimLimit": (
                    "external-public 벤치마크다. 밥플래너 성능이 아니다. "
                    "fine-tuning 여부는 이 숫자를 보고 나서 정한다."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n{args.out / 'metrics.json'}")
    print("이 숫자는 external-public 벤치마크다. 밥플래너 성능이 아니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
