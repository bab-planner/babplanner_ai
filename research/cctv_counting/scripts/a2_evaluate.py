"""Stage A2 정량 평가.

수동 정답이 붙은 구간에 대해서만 채점한다. 정답이 없는 구간의 예측은
비교 대상이 아니므로 아예 계산에 넣지 않는다.

    .venv-cctv\\Scripts\\python.exe research\\cctv_counting\\scripts\\a2_evaluate.py ^
      --observations outputs\\a2_tracking\\observations.csv ^
      --occupancy-gt research\\cctv_counting\\evaluation\\a2_occupancy_ground_truth.csv ^
      --crossing-gt research\\cctv_counting\\evaluation\\a2_crossing_ground_truth.csv ^
      --video-id a2-demo-vtest --end-seconds 29

`evaluate.py`와 나누어 둔 이유는 이쪽이 구간 제한과 신뢰도 계층 비교를
하기 때문이다. 둘 다 지표 계산은 `cctv_counting.metrics`를 그대로 쓴다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cctv_counting.groundtruth import (
    Direction,
    read_crossing_ground_truth,
    read_occupancy_ground_truth,
)
from cctv_counting.metrics import score_occupancy
from cctv_counting.observations import read_observations
from cctv_counting.reliability import ReliabilityFlag

EXIT_OK = 0
EXIT_FAILED = 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--occupancy-gt", type=Path, required=True)
    parser.add_argument("--crossing-gt", type=Path, default=None)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--end-seconds", type=float, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    observations = read_observations(args.observations)
    by_frame = {o.frame_index: o for o in observations}

    gt = [
        row
        for row in read_occupancy_ground_truth(args.occupancy_gt)
        if row.video_id == args.video_id and row.timestamp_seconds <= args.end_seconds
    ]
    matched = [(row, by_frame[row.frame_index]) for row in gt if row.frame_index in by_frame]
    if len(matched) != len(gt):
        print(f"경고: 정답 {len(gt)}건 중 {len(matched)}건만 관측과 맞춰졌다.")
    if not matched:
        print("맞춰진 표본이 없다.", file=sys.stderr)
        return EXIT_FAILED

    actual = [float(r.actual_count) for r, _ in matched]
    predicted = [float(o.occupancy_count) for _, o in matched]
    scores = score_occupancy(actual, predicted)

    label = args.label or args.observations.parent.name
    print(f"=== {label} ===")
    print(f"정답 구간 0 ~ {args.end_seconds:.0f}초, {len(matched)}개 시점\n")
    print("## Occupancy")
    print(scores.render())
    print(f"실제 평균 {sum(actual)/len(actual):.2f}명, 예측 평균 {sum(predicted)/len(predicted):.2f}명")

    print("\n시점별 (t / 실제 / 예측 / 오차 / 신뢰)")
    for row, obs in matched:
        diff = obs.occupancy_count - row.actual_count
        print(
            f"  {row.timestamp_seconds:5.1f}s  {row.actual_count:2d}  {obs.occupancy_count:2d}  "
            f"{diff:+3d}   {obs.reliability_flag.value}"
        )

    payload: dict = {
        "label": label,
        "windowEndSeconds": args.end_seconds,
        "samples": len(matched),
        "occupancy": {
            "mae": scores.mae,
            "rmse": scores.rmse,
            "bias": scores.bias,
            "actualMean": sum(actual) / len(actual),
            "predictedMean": sum(predicted) / len(predicted),
        },
    }

    # 신뢰도 계층. RQ3의 첫 탐색이다. 보정된 불확실성이 아니라
    # heuristic reliability stratification일 뿐이다.
    groups: dict[str, list[float]] = {}
    for row, obs in matched:
        groups.setdefault(obs.reliability_flag.value, []).append(
            abs(obs.occupancy_count - row.actual_count)
        )
    print("\n## Reliability stratification (heuristic)")
    strat = {}
    for flag in ReliabilityFlag:
        errs = groups.get(flag.value, [])
        if not errs:
            print(f"  {flag.value:16} 표본 0 — INSUFFICIENT SAMPLE")
            strat[flag.value] = {"samples": 0, "meanAbsError": None}
            continue
        mean_err = sum(errs) / len(errs)
        note = "" if len(errs) >= 5 else "  (표본 부족 — 해석 불가)"
        print(f"  {flag.value:16} 표본 {len(errs):2d}  평균 절대 오차 {mean_err:.2f}{note}")
        strat[flag.value] = {"samples": len(errs), "meanAbsError": mean_err}
    payload["reliabilityStratification"] = strat

    # 입장/퇴장. 창 끝 시점의 누적값과 정답 사건 수를 비교한다.
    if args.crossing_gt is not None and args.crossing_gt.exists():
        events = [
            e
            for e in read_crossing_ground_truth(args.crossing_gt)
            if e.video_id == args.video_id and e.timestamp_seconds <= args.end_seconds
        ]
        actual_in = sum(1 for e in events if e.direction is Direction.IN)
        actual_out = sum(1 for e in events if e.direction is Direction.OUT)
        end_frame = max(r.frame_index for r, _ in matched)
        end_obs = by_frame[end_frame]

        print(f"\n## Entry / Exit (0 ~ {args.end_seconds:.0f}초 누적)")
        if end_obs.entries_total is None:
            print("  측정 안 됨 — 추적을 껐으므로 통과 사건을 셀 수 없다.")
            payload["entry"] = None
        else:
            in_err = abs(end_obs.entries_total - actual_in)
            out_err = abs((end_obs.exits_total or 0) - actual_out)
            print(f"  IN   실제 {actual_in:2d}  예측 {end_obs.entries_total:2d}  절대 오차 {in_err}")
            print(f"  OUT  실제 {actual_out:2d}  예측 {end_obs.exits_total:2d}  절대 오차 {out_err}")
            payload["entry"] = {
                "actualIn": actual_in,
                "predictedIn": end_obs.entries_total,
                "absoluteErrorIn": in_err,
                "actualOut": actual_out,
                "predictedOut": end_obs.exits_total,
                "absoluteErrorOut": out_err,
            }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"\n결과: {args.out}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
