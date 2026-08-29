"""W6 — 통제된 연출 파일럿의 상태를 센다. **읽기만 한다.**

`audit_dataset.py`와 목적이 다르다. 그쪽은 "학습에 쓸 표본이 몇 개인가"를
묻고, 여기는 "**평가를 시작해도 되는가**"를 묻는다. 관문을 통과하지 못하면
`PILOT_INSUFFICIENT`로 끝나고, 그 상태에서는 모델 평가 스크립트가 돌지 않는다.

표본 12개로 잰 MAE를 보고서에 적으면 그 숫자는 반드시 성능으로 인용된다.
그럴 바에는 재지 않는 게 낫다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w6_pilot_audit.py ^
      --manifest research\\waste_estimation\\data\\private\\w6_pilot\\manifest.csv ^
      --images research\\waste_estimation\\data\\private\\w6_pilot\\images
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SERVICE = Path(__file__).resolve().parents[3] / "src"
if str(_SERVICE) not in sys.path:
    sys.path.insert(0, str(_SERVICE))

from waste_estimation.pilot import (  # noqa: E402
    HIGH_WASTE_THRESHOLD,
    MIN_FOOD_GROUPS,
    MIN_HIGH_WASTE,
    MIN_VALID_SAMPLES,
    PilotError,
    evaluate_gate,
    load_manifest,
    repeatability,
    summarize_repeatability,
)

#: 데이터가 아직 없을 때 쓰는 상태값. 보고서와 같은 문자열을 쓴다.
BLOCKED = "BLOCKED_PENDING_CONTROLLED_DATA"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    if not args.manifest.exists():
        # **데이터가 없다는 것을 0으로 말한다.** 스키마가 있다는 것과 표본이
        # 있다는 것은 다른 말이다.
        print(f"STATUS: {BLOCKED}")
        print(f"매니페스트가 없습니다: {args.manifest}")
        print("수집 절차: docs/waste_controlled_pilot_runbook.md")
        print("템플릿: docs/templates/waste_controlled_pilot_manifest_template.csv")
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(
                    {
                        "status": BLOCKED,
                        "manifest": str(args.manifest),
                        "validSamples": 0,
                        "note": (
                            "원본 파일럿 데이터가 없다. 표본 수를 추정하지 않는다."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return 0

    try:
        samples, rejections = load_manifest(args.manifest, args.images)
    except PilotError as error:
        print(f"STATUS: PILOT_INSUFFICIENT\n매니페스트를 읽을 수 없습니다: {error}")
        return 1

    # 반복 계량 행은 평가 표본이 아니다. 측정 오차 계산에만 쓴다.
    primary = [sample for sample in samples if sample.repeat_index == 0]
    repeats: dict[str, list[float]] = {}
    served: dict[str, float] = {}
    for sample in samples:
        repeats.setdefault(sample.sample_id, []).append(sample.leftover_edible_mass_g)
        served.setdefault(sample.sample_id, sample.served_edible_mass_g)

    gate = evaluate_gate(primary, rejections)

    print(f"STATUS: {gate['status']}")
    print(f"\n매니페스트 {args.manifest}")
    print(f"  전체 행 {len(samples) + len(rejections)}  ·  유효 {len(primary)}  ·  거절 {len(rejections)}")
    print(f"  음식 범주 {len({s.food_group for s in primary})}")
    print(f"  촬영 회차 {len({s.capture_session_id for s in primary})}")
    high = [s for s in primary if s.waste_rate >= HIGH_WASTE_THRESHOLD]
    print(f"  고잔반(>= {HIGH_WASTE_THRESHOLD}) {len(high)}")
    print(f"  구간 분포 {gate['binCounts']}")

    print("\n=== 관문 ===")
    for name, check in gate["checks"].items():
        mark = "통과" if check["passed"] else "**미달**"
        print(f"  {name:24} {check['actual']:>4} / 기준 {check['required']:<4} {mark}")

    if rejections:
        print("\n=== 거절 사유 ===")
        for reason, count in Counter(item["reason"] for item in rejections).items():
            print(f"  {reason:28} {count}건")
        for item in rejections[:5]:
            print(f"    {item['sampleId']}: {item['reason']} {item['detail']}")

    repeat_entries = []
    for sample_id, values in repeats.items():
        if len(values) >= 2:
            repeat_entries.append(repeatability(values, served[sample_id]))
    summary = None
    if repeat_entries:
        summary = summarize_repeatability(repeat_entries)
        print(f"\n=== 반복 계량 ({summary['samplesRepeated']}개 표본) ===")
        print(
            f"  표준편차 평균 {summary['meanStdG']:.2f} g "
            f"({summary['meanStdWasteRate']:.4f} 잔반률)"
        )
        print(
            f"  표준편차 최대 {summary['maxStdG']:.2f} g "
            f"({summary['maxStdWasteRate']:.4f} 잔반률)"
        )
        print(f"  {summary['interpretation']}")
    else:
        print("\n반복 계량 기록이 없습니다. 최소 5개 표본에서 3회씩 재세요.")

    if not gate["passed"]:
        print(
            "\n관문을 통과하지 못했습니다. **모델 평가를 하지 않습니다.**\n"
            f"  필요: 표본 {MIN_VALID_SAMPLES} / 범주 {MIN_FOOD_GROUPS} / "
            f"고잔반 {MIN_HIGH_WASTE}"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "status": gate["status"],
                    "validSamples": len(primary),
                    "foodGroups": sorted({s.food_group for s in primary}),
                    "captureSessions": sorted({s.capture_session_id for s in primary}),
                    "highWasteSamples": len(high),
                    "binCounts": gate["binCounts"],
                    "rejections": Counter(
                        item["reason"] for item in rejections
                    ),
                    "gate": gate["checks"],
                    "repeatability": summary,
                    "note": (
                        "연출된 잔반 상태다. 사람이 먹고 남긴 표본이 아니다."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
