"""데이터 상태를 센다. **읽기만 한다.**

이 단계에서 가장 중요한 질문 하나에 답하기 위한 도구다.

    지금 학습에 쓸 수 있는 표본이 몇 개인가?

매니페스트가 있으면 그것을 읽고, 없으면 이미지 디렉터리만 훑는다. 둘 다
없으면 **0이라고 말한다.** 문서에 스키마가 있다는 것과 데이터가 있다는 것은
다른 말이다.

    .venv\\Scripts\\python.exe research\\waste_estimation\\scripts\\audit_dataset.py ^
      --manifest data\\processed\\pilot-2026-01\\manifest.csv ^
      --images data\\processed\\pilot-2026-01\\images
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SERVICE_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SERVICE_SRC) not in sys.path:
    sys.path.insert(0, str(_SERVICE_SRC))

from waste_estimation.audit import audit_images, audit_samples  # noqa: E402
from waste_estimation.schemas import (  # noqa: E402
    SampleSource,
    WasteSample,
    WasteSchemaError,
)


def read_manifest(path: Path, source: SampleSource) -> tuple[list[WasteSample], list[str]]:
    """매니페스트를 연구 표본으로 옮긴다.

    열 이름은 `docs/waste_dataset_spec.md`를 따른다. 읽지 못한 행은 버리지
    않고 이유와 함께 돌려준다. 조용히 줄어든 표본 수는 나중에 아무도 설명할
    수 없다.
    """
    samples: list[WasteSample] = []
    problems: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=2):
            sample_id = (row.get("sample_id") or "").strip()
            # 분할 단위는 서비스 규칙과 같다. 참가자 키가 있으면 그것을,
            # 없으면 촬영 회차를 쓴다(dataset/validation.py::group_key_of).
            participant = (row.get("participant_group_id") or "").strip()
            session = (row.get("capture_session_id") or "").strip()
            group_id = (
                f"participant:{participant}" if participant else f"session:{session}"
            )
            raw_rate = (row.get("waste_rate") or "").strip()
            try:
                rate = float(raw_rate) if raw_rate else None
                samples.append(
                    WasteSample(
                        sample_id=sample_id,
                        source=source,
                        group_id=group_id,
                        waste_rate=rate,
                        image_reference=(row.get("image_path") or "").strip() or None,
                    )
                )
            except (ValueError, WasteSchemaError) as error:
                problems.append(f"{path.name}:{index} {error}")
    return samples, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="매니페스트 CSV")
    parser.add_argument("--images", type=Path, help="이미지 디렉터리")
    parser.add_argument(
        "--source",
        default=SampleSource.PILOT.value,
        choices=[item.value for item in SampleSource],
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    if args.manifest is None and args.images is None:
        parser.error("--manifest 또는 --images 중 하나는 필요합니다.")

    report: dict[str, object] = {}

    if args.manifest is not None:
        if not args.manifest.exists():
            print(f"매니페스트가 없습니다: {args.manifest}")
            print("표본 0건. 학습을 시작할 수 없습니다.")
            report["manifest"] = {"path": str(args.manifest), "exists": False}
        else:
            samples, problems = read_manifest(
                args.manifest, SampleSource(args.source)
            )
            audit = audit_samples(samples)
            report["manifest"] = {
                "path": str(args.manifest),
                "exists": True,
                "unreadableRows": problems,
                **audit.as_dict(),
            }
            print(f"매니페스트  {args.manifest}")
            print(f"  전체 표본        {audit.total}")
            print(f"  정답 있는 표본   {audit.labeled}")
            print(f"  정답 없는 표본   {audit.unlabeled}")
            print(f"  그룹 수          {audit.groups}  (가장 큰 그룹 {audit.largest_group}건)")
            print(f"  중복 sample_id   {len(audit.duplicate_sample_ids)}")
            print(f"  읽지 못한 행     {len(problems)}")
            distribution = audit.distribution
            if distribution.count:
                print(
                    f"  잔반률           min {distribution.minimum:.3f} / "
                    f"median {distribution.median:.3f} / "
                    f"mean {distribution.mean:.3f} / max {distribution.maximum:.3f}"
                )
                for start, count in distribution.histogram:
                    print(f"    [{start:.1f}, {start + 0.1:.1f})  {count}")
            else:
                print("  잔반률           표본 없음")
            print(
                f"  학습 가능        {'예' if audit.usable_for_training else '아니오'}"
            )

    if args.images is not None:
        image_audit = audit_images(args.images)
        report["images"] = image_audit.as_dict()
        print(f"\n이미지  {args.images}")
        if not image_audit.exists:
            print("  디렉터리가 없습니다. 이미지 0장.")
        else:
            print(f"  파일 수          {image_audit.image_count}")
            print(f"  고유 이미지      {image_audit.unique_images}")
            print(f"  중복 묶음        {len(image_audit.duplicate_groups)}")
            print(f"  열지 못한 파일   {len(image_audit.unreadable)}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n기록: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
