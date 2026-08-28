"""A3 H3 — 입장/퇴장 오차를 사건 단위로 쪼갠다.

개수 오차만으로는 "없는 것을 셌는지" "있는 것을 놓쳤는지" 알 수 없다.
정답 사건과 예측 사건을 시각으로 맞춰 보고 어느 쪽인지 나눈다.

예측 사건 시각은 observations.csv의 누적값이 올라가는 프레임에서 얻는다.
별도 실행 없이 이미 저장된 결과에서 그대로 복원된다.

    .venv-cctv\\Scripts\\python.exe research\\cctv_counting\\scripts\\a3_event_analysis.py ^
      --observations outputs\\a3_C_track_minhits1\\observations.csv ^
      --crossing-gt research\\cctv_counting\\evaluation\\a3_holdout_crossing_ground_truth.csv ^
      --video-id a2-demo-vtest --start-seconds 30 --end-seconds 79
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cctv_counting.groundtruth import read_crossing_ground_truth
from cctv_counting.metrics import TimedEvent, match_events
from cctv_counting.observations import read_observations

#: 정답이 1초 간격 표본에서 나왔고 확정 거리 때문에 예측도 조금 늦는다.
#: **결과를 보기 전에 정한 값이다.**
DEFAULT_TOLERANCE_SECONDS = 1.5


def predicted_events(path: Path, start: float, end: float) -> list[TimedEvent]:
    """누적 열이 올라가는 지점을 사건으로 되돌린다."""
    rows = sorted(read_observations(path), key=lambda o: o.frame_index)
    events: list[TimedEvent] = []
    prev_in = prev_out = 0
    for row in rows:
        if row.entries_total is None:
            continue
        for direction, current, previous in (
            ("IN", row.entries_total, prev_in),
            ("OUT", row.exits_total or 0, prev_out),
        ):
            for _ in range(max(0, current - previous)):
                if start <= row.timestamp_seconds <= end:
                    events.append(TimedEvent(row.timestamp_seconds, direction))
        prev_in = row.entries_total
        prev_out = row.exits_total or 0
    return events


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--crossing-gt", type=Path, required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float, required=True)
    parser.add_argument(
        "--tolerance-seconds", type=float, default=DEFAULT_TOLERANCE_SECONDS
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    predicted = predicted_events(args.observations, args.start_seconds, args.end_seconds)
    actual = [
        TimedEvent(e.timestamp_seconds, e.direction.value)
        for e in read_crossing_ground_truth(args.crossing_gt)
        if e.video_id == args.video_id
        and args.start_seconds <= e.timestamp_seconds <= args.end_seconds
    ]

    print(f"구간 {args.start_seconds:.0f} ~ {args.end_seconds:.0f}초, "
          f"허용 오차 ±{args.tolerance_seconds}초 (사전 고정)")
    payload: dict = {
        "windowStartSeconds": args.start_seconds,
        "windowEndSeconds": args.end_seconds,
        "toleranceSeconds": args.tolerance_seconds,
        "toleranceFixedBeforeSeeingResults": True,
        "directions": {},
    }

    for direction in ("IN", "OUT"):
        p = [e for e in predicted if e.direction == direction]
        a = [e for e in actual if e.direction == direction]
        result = match_events(p, a, args.tolerance_seconds)
        print(f"\n## {direction}")
        print(f"  정답 {len(a)}건, 예측 {len(p)}건")
        print(f"  맞음 {len(result.matched)}  없는 것을 셈 {len(result.false_positives)}  "
              f"놓침 {len(result.missed)}")

        def fmt(value: float | None) -> str:
            return "N/A" if value is None else f"{value:.3f}"

        small = len(a) < 10
        note = "  (표본이 작다. exploratory metric으로만 읽는다)" if small else ""
        print(f"  precision {fmt(result.precision)}  recall {fmt(result.recall)}  "
              f"F1 {fmt(result.f1)}{note}")
        if result.false_positives:
            print("  없는 것을 센 시각:",
                  ", ".join(f"{e.timestamp_seconds:.1f}s" for e in result.false_positives))
        if result.missed:
            print("  놓친 시각:",
                  ", ".join(f"{e.timestamp_seconds:.1f}s" for e in result.missed))

        payload["directions"][direction] = {
            "actual": len(a),
            "predicted": len(p),
            "matched": len(result.matched),
            "falsePositives": len(result.false_positives),
            "missed": len(result.missed),
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "exploratoryOnlySmallSample": small,
            "falsePositiveTimes": [e.timestamp_seconds for e in result.false_positives],
            "missedTimes": [e.timestamp_seconds for e in result.missed],
        }

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"\n결과: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
