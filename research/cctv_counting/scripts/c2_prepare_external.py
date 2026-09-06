"""C2/C3 — 내려받은 subset을 시퀀스 단위로 다시 나누고 공통 스키마로 내보낸다.

제공된 split을 쓰지 않는다. 그쪽은 MOT 시퀀스 11개 전부가 train·val·test에
동시에 들어 있고 프레임 간격 최빈값이 1이다. 연속 프레임이 갈렸다는 뜻이고,
인접 프레임은 사실상 같은 이미지라 그 split으로 잰 성능은 실제보다 높다.

여기서 만든 것만 아래 단계가 쓴다. 원본 디렉터리의 train/val/test 구조는
읽기만 하고 의미로 쓰지 않는다.

    .venv-waste\\Scripts\\python.exe research\\cctv_counting\\scripts\\c2_prepare_external.py
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

from cctv_counting.external import (  # noqa: E402
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    PERMISSIVE,
    AdapterError,
    read_yolo_frame,
    split_by_group,
)

ROOT = Path("research/cctv_counting/datasets/raw/crowd_cctv")
OUT = Path("research/cctv_counting/datasets/raw/crowd_cctv/prepared")
SUMMARY = Path("research/cctv_counting/datasets/c2_crowd_cctv_prepared.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    # 원본이 나눠 둔 폴더를 가로질러 전부 모은다. 그 구분은 여기서 버린다.
    images = sorted(args.root.glob("images/*/*.jpg"))
    if not images:
        print(f"이미지가 없습니다: {args.root}/images/*/*.jpg")
        print("먼저 c1_fetch_crowd_cctv.py 를 실행하세요.")
        return 1

    frames = []
    problems: dict[str, list[str]] = {}
    for image_path in images:
        label_path = args.root / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
        # C1이 허용 subset만 받았으므로 여기 있는 것은 전부 permissive다.
        frame, issues = read_yolo_frame(
            image_path, label_path, license_class=PERMISSIVE, root=args.root
        )
        frames.append(frame)
        if issues:
            problems[frame.frame_id] = issues

    counts = Counter(frame.source for frame in frames)
    groups = {frame.group_key for frame in frames}
    print(f"프레임 {len(frames)}장")
    for source in sorted(counts):
        print(f"  {source:14} {counts[source]}")
    print(f"  분할 그룹      {len(groups)}")

    try:
        assignment = split_by_group(frames, seed=args.seed)
    except AdapterError as error:
        print(f"분할할 수 없습니다: {error}")
        return 1

    # 한 그룹이 두 split에 걸치면 재분할한 의미가 없다. 여기서 확인한다.
    by_group: dict[str, set[str]] = defaultdict(set)
    for frame in frames:
        by_group[frame.group_key].add(assignment[frame.frame_id])
    spanning = sorted(key for key, splits in by_group.items() if len(splits) > 1)
    if spanning:
        print(f"같은 그룹이 여러 split에 있습니다: {spanning[:5]}")
        return 1

    split_counts = Counter(assignment.values())
    people = sum(frame.person_count for frame in frames)
    empty = sum(1 for frame in frames if frame.person_count == 0)
    print(f"\n분할 (seed {args.seed}, 그룹 단위)")
    for name in ("train", "val", "test"):
        rows = [f for f in frames if assignment[f.frame_id] == name]
        print(
            f"  {name:6} 프레임 {len(rows):5}  그룹 {len({f.group_key for f in rows}):5}  "
            f"사람 {sum(f.person_count for f in rows):6}"
        )
    print(f"\n사람 상자 {people}개, 사람 없는 프레임 {empty}장")
    if problems:
        reasons = Counter(
            issue.split(": ", 1)[-1] for issues in problems.values() for issue in issues
        )
        print(f"버린 라벨 줄이 있는 프레임 {len(problems)}장")
        for reason, count in reasons.most_common(5):
            print(f"  {reason}  {count}줄")

    args.out.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "frameId": frame.frame_id,
            "source": frame.source,
            "sequence": frame.sequence,
            "frameIndex": frame.frame_index,
            "imagePath": frame.image_path,
            "width": frame.width,
            "height": frame.height,
            "licenseClass": frame.license_class,
            "groupKey": frame.group_key,
            "split": assignment[frame.frame_id],
            "personCount": frame.person_count,
            "boxes": [[b.x1, b.y1, b.x2, b.y2] for b in frame.boxes],
        }
        for frame in frames
    ]
    annotations = args.out / "annotations.jsonl"
    annotations.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n",
        encoding="utf-8",
    )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(
            {
                "datasetKey": "crowd_cctv_permissive",
                "benchmark": "external-public",
                "licenseClass": PERMISSIVE,
                "registry": "research/cctv_counting/datasets/c1_crowd_cctv_registry.json",
                "frames": len(frames),
                "sources": dict(sorted(counts.items())),
                "groups": len(groups),
                "personBoxes": people,
                "framesWithoutPeople": empty,
                "framesWithDroppedLabelLines": len(problems),
                "split": {
                    "method": "group-wise, seed-deterministic",
                    "groupKey": "source:sequence (시퀀스가 없으면 프레임 자신)",
                    "seed": args.seed,
                    "ratios": list(DEFAULT_RATIOS),
                    "counts": dict(sorted(split_counts.items())),
                    "providedSplitUsed": False,
                    "providedSplitReason": (
                        "MOT 시퀀스 11개 전부가 train·val·test에 동시에 있고 프레임 간격 "
                        "최빈값이 1이다. 연속 프레임이 갈렸다."
                    ),
                },
                "schema": {
                    "boxes": "픽셀 좌표 [x1, y1, x2, y2]. 이미지 밖은 잘라 넣었다.",
                    "classes": "person(0)과 pedestrian(1)을 사람 하나로 합쳤다.",
                    "trackIds": False,
                },
                "note": "annotations.jsonl 은 커밋하지 않는다. 이 요약만 남긴다.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n{annotations}")
    print(f"{args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
