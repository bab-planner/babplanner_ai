"""추적 비용만 따로 잰다.

**detector 추론은 두 갈래가 공유하므로 비교 대상이 아니다.** A6에서 바뀐 것은
tracker뿐이고, 그래서 재야 하는 것도 tracker가 더 쓰는 시간뿐이다.

같은 캐시 상자를 두 tracker에 그대로 넣고 `update()`만 잰다. 영상 디코딩도
검출도 파일 입출력도 들어가지 않는다.

    .venv-cctv\\Scripts\\python.exe research\\cctv_counting\\scripts\\a6_latency.py ^
      --split FINAL --out research\\cctv_counting\\experiments\\a6_continuity\\latency.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from a4_run import ANCHOR, FRAME_HEIGHT, FRAME_WIDTH, ROI, SOURCE_FPS  # noqa: E402
from a6_run import POLICIES, load_detections, sequences_for  # noqa: E402

from cctv_counting.detection import anchor_point, normalize_point  # noqa: E402
from cctv_counting.motion_tracking import MotionAwareTracker  # noqa: E402
from cctv_counting.tracking import ByteTrackInspiredTracker  # noqa: E402

REPEATS = 3


def eligible(detections, threshold: float):
    kept = []
    for detection in detections:
        if detection.class_id != 1 or detection.confidence < threshold:
            continue
        point = normalize_point(
            anchor_point(detection.bbox, ANCHOR), FRAME_WIDTH, FRAME_HEIGHT
        )
        if ROI.contains_normalized(point):
            kept.append(detection)
    return kept


def measure(frames: list[list], policy_name: str) -> dict:
    policy = POLICIES[policy_name]
    best: float | None = None
    for _ in range(REPEATS):
        if policy.tracker.value == "motion_aware":
            tracker = MotionAwareTracker(
                high_threshold=policy.tracker_high_threshold,
                low_threshold=policy.tracker_low_threshold,
                iou_threshold=policy.iou_threshold,
                low_iou_threshold=policy.low_iou_threshold,
                max_age_seconds=policy.max_age_seconds,
                min_hits=policy.min_hits,
                velocity_smoothing=policy.velocity_smoothing,
            )
        else:
            tracker = ByteTrackInspiredTracker(
                high_threshold=policy.tracker_high_threshold,
                low_threshold=policy.tracker_low_threshold,
                iou_threshold=policy.iou_threshold,
                low_iou_threshold=policy.low_iou_threshold,
                max_age=policy.max_age_frames,
                min_hits=policy.min_hits,
            )
        prepared = [eligible(f, policy.tracker_low_threshold) for f in frames]
        started = time.perf_counter()
        for index, detections in enumerate(prepared):
            if policy.tracker.value == "motion_aware":
                tracker.update(detections, index / SOURCE_FPS)
            else:
                tracker.update(detections)
        elapsed = time.perf_counter() - started
        best = elapsed if best is None else min(best, elapsed)

    boxes = sum(len(eligible(f, policy.tracker_low_threshold)) for f in frames)
    return {
        "policy": policy_name,
        "tracker": policy.tracker.value,
        "frames": len(frames),
        "boxesPerFrame": boxes / len(frames) if frames else None,
        "msPerFrame": best * 1000.0 / len(frames) if frames else None,
        "framesPerSecond": len(frames) / best if best else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="FINAL")
    parser.add_argument("--policies", nargs="+", default=["P0_baseline", "P2_motion"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    frames: list[list] = []
    detector_fps: list[float] = []
    for sequence, _, root in sequences_for(args.split):
        detections, elapsed, count = load_detections(root, sequence, "cpu")
        frames += [detections[k] for k in sorted(detections)]
        if elapsed:
            detector_fps.append(count / elapsed)

    rows = [measure(frames, name) for name in args.policies]
    payload = {
        "split": args.split,
        "repeats": REPEATS,
        "note": (
            "tracker.update()만 잰다. detector 추론은 두 갈래가 공유하므로 "
            "비교 대상이 아니다. 3회 중 최소값을 쓴다."
        ),
        "detectorFpsMean": statistics.fmean(detector_fps) if detector_fps else None,
        "sourceFps": SOURCE_FPS,
        "rows": rows,
    }

    print(f"split {args.split}   프레임 {len(frames)}")
    print(f"detector {payload['detectorFpsMean']:.1f} fps (source {SOURCE_FPS} fps)")
    for row in rows:
        print(
            f"  {row['policy']:20} {row['tracker']:16} "
            f"{row['msPerFrame']:.4f} ms/frame   "
            f"{row['framesPerSecond']:.0f} fps   "
            f"박스 {row['boxesPerFrame']:.2f}/frame"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n결과: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
