"""통과 사건 하나를 프레임 단위로 펼쳐 본다.

**자동 분류를 그대로 믿지 않기 위해서 있다.** `attribution.py`의 규칙은
86건을 사람이 전부 볼 수 없어서 두는 1차 분류다. 대표 사건 몇 개는 여기서
프레임 단위로 직접 읽고, 자동 분류가 실제로 그 사건을 설명하는지 확인한다.

한 줄이 한 프레임이다.

    frame   정답 발끝 x(정규화)   정답이 선의 어느 쪽인가
            그 사람에게 붙은 원시 검출의 신뢰도
            그 사람에게 붙은 예측 track 번호

track 번호가 바뀌는 지점, 검출이 끊기는 지점이 눈으로 보인다.

    .venv-cctv\\Scripts\\python.exe research\\cctv_counting\\scripts\\a6_inspect.py ^
      --split ANALYSIS --sequence ThreePastShop1front --policy P0_baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from a4_run import (  # noqa: E402
    ANCHOR,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    LINE_X,
    MIN_CONFIRM_DISTANCE,
    SOURCE_FPS,
)
from a6_run import (  # noqa: E402
    CROSSING_WINDOW_SECONDS,
    POLICIES,
    SCENE,
    in_roi,
    load_detections,
    sequences_for,
)

from cctv_counting.benchmark import crossing_ground_truth, parse_cvml  # noqa: E402
from cctv_counting.continuity import associate  # noqa: E402
from cctv_counting.dualpath import DualPathCounter  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--policy", default="P0_baseline")
    parser.add_argument("--track", type=int, help="정답 track id. 없으면 전부.")
    parser.add_argument("--pad", type=float, default=CROSSING_WINDOW_SECONDS)
    args = parser.parse_args(argv)

    match = [row for row in sequences_for(args.split) if row[0] == args.sequence]
    if not match:
        raise SystemExit(f"{args.split}에 {args.sequence}가 없습니다.")
    sequence, gt_name, root = match[0]

    detections, _, _ = load_detections(root, sequence, "cpu")
    gt_frames = parse_cvml(root / gt_name)
    scored = sorted(set(detections) & set(gt_frames))
    policy = POLICIES[args.policy]

    counter = DualPathCounter(policy=policy, scene=SCENE)
    per_frame: dict[int, dict[int, int]] = {}
    detected: dict[int, dict[int, float]] = {}
    for index in scored:
        result = counter.process_detections(index, index / SOURCE_FPS, detections[index])
        truth = [(b.track_id, b.bbox) for b in gt_frames[index] if in_roi(b.bbox)]
        per_frame[index] = associate(
            truth, [(p.track_id, p.bbox) for p in result.tracked]
        )
        hits = associate(
            truth, list(enumerate([d.bbox for d in result.tracker_detections]))
        )
        detected[index] = {
            gt_id: result.tracker_detections[position].confidence
            for gt_id, position in hits.items()
        }

    events = crossing_ground_truth(
        gt_frames,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        line_x_normalized=LINE_X,
        min_confirm_distance=MIN_CONFIRM_DISTANCE,
    )
    events = [e for e in events if args.track is None or e.track_id == args.track]
    if not events:
        raise SystemExit("해당 조건의 정답 통과 사건이 없습니다.")

    half = int(args.pad * SOURCE_FPS)
    for event in events:
        print(
            f"\n=== {sequence} / {args.policy} / GT track {event.track_id} "
            f"/ {event.direction.value} @ frame {event.frame_index} ==="
        )
        print("frame     x_norm  side   detConf  trackId")
        for index in range(event.frame_index - half, event.frame_index + half + 1):
            if index not in per_frame:
                continue
            box = next(
                (b for b in gt_frames[index] if b.track_id == event.track_id), None
            )
            if box is None:
                continue
            x = box.bbox.bottom_center.x / FRAME_WIDTH
            offset = x - LINE_X
            side = (
                "  -  "
                if abs(offset) < MIN_CONFIRM_DISTANCE
                else (" left" if offset < 0 else "right")
            )
            confidence = detected[index].get(event.track_id)
            track_id = per_frame[index].get(event.track_id)
            marker = " <<<" if index == event.frame_index else ""
            occluded = "*" if box.occluded else " "
            print(
                f"{index:>5}{occluded}   {x:0.3f}  {side}   "
                f"{('%.2f' % confidence) if confidence else '  -  '}   "
                f"{track_id if track_id is not None else '  -'}{marker}"
            )
    print("\n* = 정답이 occluded로 표시한 프레임. ANCHOR =", ANCHOR.value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
