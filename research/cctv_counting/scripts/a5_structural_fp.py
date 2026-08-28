"""해상도를 올리면 구조적 오탐이 줄어드는가 늘어나는가 (A5 §42, H1 재검증).

A4의 H1은 "오탐 상당수가 **한 자리에 계속 머무는** 구조물"이라는 것이었다.
사람이 아닌 물체를 사람으로 보는 것이니, 해상도를 올리면 두 방향 모두
가능하다. 더 잘 보여서 아니라고 판단할 수도 있고, 더 잘 보여서 작은 물체가
새로 사람처럼 보일 수도 있다. 어느 쪽인지 세어 본다.

**총 오탐 수만 비교하면 답이 안 나온다.** 한 자리에 200프레임 머무는 오탐
하나와, 여기저기 튀는 오탐 200개는 총계가 같지만 원인이 다르다. 그래서
오탐을 위치로 묶고, 묶음이 몇 프레임에 걸쳐 지속되는지를 본다.

    구조적(persistent)   같은 자리에서 전체 프레임의 20% 이상 나타난 묶음
    산발적(transient)    그 밖

격자는 프레임을 16x12로 나눈 것이다. CAVIAR 384x288에서 한 칸이 24x24px다.
사람 한 명이 서 있는 자리 정도의 크기이고, 이 값은 결과를 보기 전에 정했다.

계산은 캐시된 검출 상자만 쓴다. GPU를 다시 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from a5_run import (  # noqa: E402
    CANDIDATES,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    THRESHOLD,
    cache_path,
    sequences_for,
)
from cctv_counting.benchmark import parse_cvml  # noqa: E402
from cctv_counting.detection import Detection  # noqa: E402
from cctv_counting.geometry import BoundingBox  # noqa: E402
from cctv_counting.sizeeval import MATCH_IOU, match_frame, size_bin  # noqa: E402

#: 격자 칸 수. 결과를 보기 전에 정했다.
GRID_X, GRID_Y = 16, 12

#: 이 비율 이상의 프레임에 나타난 격자 칸을 구조적이라고 부른다.
PERSISTENT_RATIO = 0.20


def load_cache(root: Path, sequence: str, tag: str) -> dict[int, list[Detection]] | None:
    path = cache_path(root, sequence, tag)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(k): [Detection(BoundingBox(*b[:4]), b[4], int(b[5])) for b in v]
        for k, v in raw["frames"].items()
    }


def cell(bbox: BoundingBox) -> tuple[int, int]:
    """상자 발끝이 속한 격자 칸. 발끝을 쓰는 이유는 바닥 위치가 물체의 자리를 가장 잘 나타내기 때문이다."""
    foot = bbox.bottom_center
    cx = min(GRID_X - 1, max(0, int(foot.x / FRAME_WIDTH * GRID_X)))
    cy = min(GRID_Y - 1, max(0, int(foot.y / FRAME_HEIGHT * GRID_Y)))
    return cx, cy


def analyse_sequence(detections: dict, gt_frames: dict) -> dict:
    scored = sorted(set(detections) & set(gt_frames))
    per_cell: dict[tuple[int, int], int] = defaultdict(int)
    fp_total = 0
    fp_by_size: dict[str, int] = defaultdict(int)

    for index in scored:
        truth = [
            b.bbox for b in gt_frames[index]
            if not b.occluded
        ]
        predicted = [d.bbox for d in detections[index] if d.confidence >= THRESHOLD]
        _, unmatched_p, _ = match_frame(predicted, truth, MATCH_IOU)
        for i in unmatched_p:
            fp_total += 1
            per_cell[cell(predicted[i])] += 1
            fp_by_size[size_bin(predicted[i].height / FRAME_HEIGHT).value] += 1

    frames = len(scored)
    threshold_frames = frames * PERSISTENT_RATIO
    persistent = {c: n for c, n in per_cell.items() if n >= threshold_frames}
    persistent_fp = sum(persistent.values())

    return {
        "scoredFrames": frames,
        "falsePositives": fp_total,
        "fpPerFrame": fp_total / frames if frames else None,
        "occupiedCells": len(per_cell),
        "persistentCells": len(persistent),
        "persistentFalsePositives": persistent_fp,
        # 구조적 오탐이 전체 오탐에서 차지하는 비율. H1이 말하는 값이다.
        "persistentShare": (persistent_fp / fp_total) if fp_total else None,
        "fpBySize": dict(fp_by_size),
        "topCells": sorted(
            ({"cell": list(c), "frames": n} for c, n in per_cell.items()),
            key=lambda r: -r["frames"],
        )[:5],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["DEV", "ANALYSIS", "FINAL"], default="DEV")
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    results: dict[str, dict] = {}
    for tag in args.candidates:
        rows = []
        for sequence, gt_name, root in sequences_for(args.split):
            detections = load_cache(root, sequence, tag)
            if detections is None:
                print(f"  캐시 없음, 건너뜀: {sequence} / {tag}")
                continue
            gt_frames = parse_cvml(root / gt_name)
            rows.append({"sequence": sequence, **analyse_sequence(detections, gt_frames)})
        if not rows:
            continue
        fp = sum(r["falsePositives"] for r in rows)
        persistent = sum(r["persistentFalsePositives"] for r in rows)
        frames = sum(r["scoredFrames"] for r in rows)
        by_size: dict[str, int] = defaultdict(int)
        for r in rows:
            for k, v in r["fpBySize"].items():
                by_size[k] += v
        results[tag] = {
            "model": CANDIDATES[tag]["model"],
            "minSize": CANDIDATES[tag]["min"],
            "sequences": len(rows),
            "falsePositives": fp,
            "fpPerFrame": fp / frames if frames else None,
            "persistentFalsePositives": persistent,
            "persistentShare": (persistent / fp) if fp else None,
            "transientFalsePositives": fp - persistent,
            "fpBySize": dict(by_size),
            "perSequence": rows,
        }

    payload = {
        "split": args.split,
        "grid": [GRID_X, GRID_Y],
        "persistentRatio": PERSISTENT_RATIO,
        "threshold": THRESHOLD,
        "note": (
            "구조적 오탐 = 한 격자 칸에서 전체 프레임의 20% 이상 나타난 오탐. "
            "총 오탐 수만 비교하면 한 자리에 머무는 오탐과 산발적 오탐을 "
            "구분할 수 없다."
        ),
        "candidates": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"{'candidate':22s} {'FP/frame':>9s} {'구조적':>7s} {'비율':>7s}  크기별")
    for tag, r in results.items():
        share = f"{r['persistentShare']:.2f}" if r["persistentShare"] is not None else "-"
        sizes = " ".join(f"{k}={v}" for k, v in sorted(r["fpBySize"].items()))
        print(f"{tag:22s} {r['fpPerFrame']:9.3f} {r['persistentFalsePositives']:7d} "
              f"{share:>7s}  {sizes}")
    print(f"기록: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
