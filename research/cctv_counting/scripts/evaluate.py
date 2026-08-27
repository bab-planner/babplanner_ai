"""정답과 예측을 맞춰 지표를 낸다.

    .venv\\Scripts\\python.exe research\\cctv_counting\\scripts\\evaluate.py ^
      --video-id demo-01 ^
      --observations outputs\\cctv_demo\\observations.csv ^
      --occupancy-gt research\\cctv_counting\\datasets\\annotations\\occupancy_ground_truth.csv ^
      --crossing-gt research\\cctv_counting\\datasets\\annotations\\crossing_ground_truth.csv

정답 파일이 비어 있으면 지표를 내지 않고 그렇다고 말한다. 0건으로 계산한
MAE 0.0은 완벽한 성능처럼 보이지만 아무것도 재지 않은 것이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cctv_counting.groundtruth import (
    entries_of,
    read_crossing_ground_truth,
    read_occupancy_ground_truth,
)
from cctv_counting.metrics import (
    MetricError,
    entry_count_error,
    render_condition_table,
    score_by_condition,
    score_occupancy,
)
from cctv_counting.observations import read_observations

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True, help="정답 파일의 video_id")
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--occupancy-gt", type=Path, default=None)
    parser.add_argument("--crossing-gt", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.observations.exists():
        print(f"관측 파일이 없습니다: {args.observations}", file=sys.stderr)
        return EXIT_USAGE

    observations = read_observations(args.observations)
    by_frame = {item.frame_index: item for item in observations}
    print(f"관측 {len(observations)}프레임")

    measured_anything = False

    if args.occupancy_gt is not None and args.occupancy_gt.exists():
        rows = [
            row
            for row in read_occupancy_ground_truth(args.occupancy_gt)
            if row.video_id == args.video_id
        ]
        matched = [(row, by_frame[row.frame_index]) for row in rows if row.frame_index in by_frame]
        missing = len(rows) - len(matched)

        print(f"\n## Occupancy\n정답 {len(rows)}건, 관측과 맞춰진 것 {len(matched)}건")
        if missing:
            print(f"주의: 관측에 없는 프레임 {missing}건은 계산에서 빠졌다.")

        if not matched:
            print("측정 안 됨 — 맞춰진 정답이 없다.")
        else:
            measured_anything = True
            scores = score_occupancy(
                [row.actual_count for row, _ in matched],
                [obs.occupancy_count for _, obs in matched],
            )
            print(scores.render())

            tagged = [
                (row.condition, float(row.actual_count), float(obs.occupancy_count))
                for row, obs in matched
                if row.condition
            ]
            print("\n### 조건별")
            if not tagged:
                print("조건 annotation이 없다. 조건별 표는 만들지 않는다.")
            else:
                print(render_condition_table(score_by_condition(tagged)))
    else:
        print("\n## Occupancy\n측정 안 됨 — 정답 파일이 지정되지 않았거나 없다.")

    if args.crossing_gt is not None and args.crossing_gt.exists():
        rows = read_crossing_ground_truth(args.crossing_gt)
        for_video = [row for row in rows if row.video_id == args.video_id]
        actual_entries = entries_of(rows, args.video_id)
        predicted = observations[-1].entries_total if observations else None

        print(f"\n## Entry count\n정답 사건 {len(for_video)}건")
        if not for_video:
            print("측정 안 됨 — 이 영상에 대한 정답 사건이 없다.")
        elif predicted is None:
            print("측정 안 됨 — 관측에 입장 인원이 없다(추적을 껐다).")
        else:
            measured_anything = True
            try:
                error = entry_count_error(actual_entries, predicted)
            except MetricError as failure:
                print(f"계산 실패: {failure}", file=sys.stderr)
                return EXIT_FAILED
            print(f"실제 {actual_entries}명, 예측 {predicted}명, 절대 오차 {error}명")
    else:
        print("\n## Entry count\n측정 안 됨 — 정답 파일이 지정되지 않았거나 없다.")

    if not measured_anything:
        print("\n아무 지표도 계산하지 못했다. 정답 데이터가 필요하다.")
        return EXIT_FAILED
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
