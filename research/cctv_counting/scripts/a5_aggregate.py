"""a5_run.py의 시퀀스별 결과를 후보별 요약으로 접는다.

**macro와 micro를 모두 낸다.** 시퀀스마다 프레임 수와 사람 수가 다르기
때문이다. micro만 쓰면 긴 시퀀스가 결론을 지배하고, macro만 쓰면 사람이
두 명 지나간 시퀀스와 스무 명 지나간 시퀀스가 같은 무게를 갖는다. 둘이
어긋나면 그 사실 자체가 보고할 내용이다.

통계의 독립 단위는 프레임이 아니라 **시퀀스**다. 같은 시퀀스의 이웃
프레임은 서로 거의 같으므로, 프레임을 표본으로 세면 표본 수가 실제보다
수백 배 커 보인다.
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

from cctv_counting.sizeeval import (  # noqa: E402
    histogram_fraction_below,
    histogram_median,
    histogram_min,
)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict]) -> dict:
    """한 후보의 시퀀스별 결과를 접는다."""
    det_total = sum(r["detection"]["gtTotal"] for r in rows)
    det_matched = sum(r["detection"]["gtMatched"] for r in rows)

    size = {}
    for bin_name in ("SMALL", "MEDIUM", "LARGE"):
        total = sum(r["detection"]["bySize"][bin_name]["total"] for r in rows)
        matched = sum(r["detection"]["bySize"][bin_name]["matched"] for r in rows)
        per_sequence = [
            r["detection"]["bySize"][bin_name]["recall"]
            for r in rows
            if r["detection"]["bySize"][bin_name]["recall"] is not None
        ]
        size[bin_name] = {
            "gtTotal": total,
            "matched": matched,
            # 표본이 없으면 None. 0.0으로 두면 "전부 놓쳤다"로 읽힌다.
            "microRecall": (matched / total) if total else None,
            "macroRecall": mean(per_sequence),
            "sequencesWithSamples": len(per_sequence),
        }

    seq_mae = [r["occupancy"]["mae"] for r in rows if r["occupancy"]["mae"] is not None]
    seq_bias = [r["occupancy"]["bias"] for r in rows if r["occupancy"]["bias"] is not None]
    all_abs = [e for r in rows for e in r["occupancy"]["absErrors"]]

    crossing = {}
    for direction in ("IN", "OUT"):
        actual = sum(r["crossing"][direction]["actual"] for r in rows)
        predicted = sum(r["crossing"][direction]["predicted"] for r in rows)
        matched = sum(r["crossing"][direction]["matched"] for r in rows)
        precision = (matched / predicted) if predicted else None
        recall = (matched / actual) if actual else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else None
        )
        crossing[direction] = {
            "actual": actual, "predicted": predicted, "matched": matched,
            "microPrecision": precision, "microRecall": recall, "microF1": f1,
        }

    # 시퀀스별 히스토그램을 더한 뒤에 중앙값을 낸다. 시퀀스별 중앙값의
    # 평균은 전체 중앙값이 아니다.
    missed: dict[str, int] = {}
    matched_h: dict[str, int] = {}
    for r in rows:
        for target, key in ((missed, "missedHeightHistogram"),
                            (matched_h, "matchedHeightHistogram")):
            for bucket, n in r["detection"][key].items():
                target[bucket] = target.get(bucket, 0) + n

    return {
        "sequences": len(rows),
        "scoredFrames": sum(r["scoredFrames"] for r in rows),
        "detection": {
            "gtTotal": det_total,
            "microRecall": (det_matched / det_total) if det_total else None,
            "macroRecall": mean(
                [r["detection"]["recall"] for r in rows
                 if r["detection"]["recall"] is not None]
            ),
            "microFpPerFrame": (
                sum(r["detection"]["falsePositivesPerFrame"] * r["scoredFrames"]
                    for r in rows if r["detection"]["falsePositivesPerFrame"] is not None)
                / sum(r["scoredFrames"] for r in rows)
            ),
            "bySize": size,
            "medianMissedHeight": histogram_median(missed),
            "medianMatchedHeight": histogram_median(matched_h),
            "missedBelowSmallMax": histogram_fraction_below(missed, 0.10),
            "minMatchedHeight": histogram_min(matched_h),
        },
        "occupancy": {
            "macroMae": mean(seq_mae),
            "macroBias": mean(seq_bias),
            "microMae": mean(all_abs),
            "worstSequenceMae": max(seq_mae) if seq_mae else None,
            "bestSequenceMae": min(seq_mae) if seq_mae else None,
            # 시퀀스가 독립 단위다. 시퀀스 간 표준편차를 함께 적는다.
            "maeStdevAcrossSequences": (
                statistics.stdev(seq_mae) if len(seq_mae) > 1 else None
            ),
        },
        "crossing": crossing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = json.loads(args.result.read_text(encoding="utf-8"))
    by_candidate: dict[str, list[dict]] = {}
    meta: dict[str, dict] = {}
    for row in payload["rows"]:
        by_candidate.setdefault(row["candidate"], []).append(row)
        meta.setdefault(row["candidate"], {
            "model": row["model"],
            "shape": f"{row['transformedShape'][0]}x{row['transformedShape'][1]}",
            "detectFps": row["detectFps"],
            "peakVramMb": row["peakVramMb"],
            "threshold": row["threshold"],
        })

    out = {
        "split": payload.get("split"),
        "candidates": {
            tag: {**meta[tag], **summarize(rows)}
            for tag, rows in by_candidate.items()
        },
    }
    args.out.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for tag, s in out["candidates"].items():
        d, o = s["detection"], s["occupancy"]
        print(
            f"{tag:20s} recall micro {d['microRecall']:.3f} "
            f"macro {d['macroRecall']:.3f} | "
            f"SMALL {d['bySize']['SMALL']['microRecall']} | "
            f"occ MAE macro {o['macroMae']:.3f} micro {o['microMae']:.3f}"
        )
    print(f"기록: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
