"""사건 회복 표 — baseline이 놓친 통과를 selected가 살렸는가.

집계 수치는 "몇 건이 좋아졌다"까지만 말한다. 어떤 사건이, 어떤 원인이었고,
살아났는지 죽었는지는 사건 단위로 남겨야 나중에 확인할 수 있다.

**개인정보가 없다.** 남는 것은 시퀀스 이름, 프레임 번호, 그 시퀀스 안에서만
유효한 정답 track 번호, 그리고 연속성 수치뿐이다.

    .venv-cctv\\Scripts\\python.exe research\\cctv_counting\\scripts\\a6_recovery.py ^
      research\\cctv_counting\\experiments\\a6_continuity\\final.json ^
      --out research\\cctv_counting\\experiments\\a6_continuity\\final_event_recovery.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def key(event: dict) -> tuple[int, str, int]:
    return (event["frameIndex"], event["direction"], event["gtTrackId"])


def build(payload: dict, baseline: str, selected: str) -> list[dict]:
    rows = []
    for row in payload["rows"]:
        base = {
            key(e): e
            for e in next(
                r for r in row["entryRows"] if r["policy"] == baseline
            )["events"]
        }
        sel = {
            key(e): e
            for e in next(
                r for r in row["entryRows"] if r["policy"] == selected
            )["events"]
        }
        for identifier, before in sorted(base.items()):
            after = sel[identifier]
            rows.append(
                {
                    "sequence": row["sequence"],
                    "frameIndex": identifier[0],
                    "direction": identifier[1],
                    "gtTrackId": identifier[2],
                    "baselineMatched": before["matched"],
                    "selectedMatched": after["matched"],
                    "baselineCause": before.get("cause"),
                    "selectedCause": after.get("cause"),
                    "baselineDominantShare": before["dominantTrackShare"],
                    "selectedDominantShare": after["dominantTrackShare"],
                    "baselineDetectionCoverage": before["detectionCoverage"],
                    "selectedDetectionCoverage": after["detectionCoverage"],
                    "outcome": (
                        "RECOVERED"
                        if not before["matched"] and after["matched"]
                        else "LOST"
                        if before["matched"] and not after["matched"]
                        else "BOTH_HIT"
                        if before["matched"]
                        else "BOTH_MISS"
                    ),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--baseline", default="P0_baseline")
    parser.add_argument("--selected", default="P2_motion")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    rows = build(payload, args.baseline, args.selected)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    recovered_by_cause: dict[str, list[int]] = {}
    for row in rows:
        if row["baselineMatched"]:
            continue
        cause = row["baselineCause"] or "UNKNOWN"
        hit, total = recovered_by_cause.setdefault(cause, [0, 0])
        recovered_by_cause[cause] = [
            hit + (1 if row["selectedMatched"] else 0),
            total + 1,
        ]

    print(f"split {payload['split']}   정답 사건 {len(rows)}건")
    for name in ("BOTH_HIT", "RECOVERED", "LOST", "BOTH_MISS"):
        print(f"  {name:10} {counts.get(name, 0)}")
    print("\nbaseline 실패 원인별 회복률")
    for cause, (hit, total) in sorted(recovered_by_cause.items()):
        print(f"  {cause:20} {hit}/{total}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "split": payload["split"],
                    "baseline": args.baseline,
                    "selected": args.selected,
                    "outcomes": counts,
                    "recoveredByBaselineCause": {
                        cause: {"recovered": hit, "total": total}
                        for cause, (hit, total) in sorted(recovered_by_cause.items())
                    },
                    "events": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\n결과: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
