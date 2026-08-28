"""H5 판정 — 통과 누락은 detector recall보다 창 연속성과 더 붙어 있는가.

**왜 단순 비교로는 부족한가.** 놓친 사건의 창 연속성이 낮다는 것만으로는
H5를 지지할 수 없다. 검출이 없으면 추적도 없으므로 두 값은 자동으로 같이
떨어진다. 그러면 "검출 문제"를 "연속성 문제"라고 다시 부른 것에 불과하다.

그래서 **검출이 충분했던 사건만** 따로 본다.

    detection_coverage >= 0.5 인 사건들
        -> detector는 그 사람을 창의 절반 이상에서 봤다
        -> 여기서도 맞힘/놓침이 갈린다면, 그 차이는 검출이 아니다

이 비교가 H5의 핵심 근거다.

    .venv-cctv\\Scripts\\python.exe research\\cctv_counting\\scripts\\a6_h5.py ^
      research\\cctv_counting\\experiments\\a6_continuity\\analysis_shop.json ^
      research\\cctv_counting\\experiments\\a6_continuity\\analysis_lobby.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: 사건 창에서 이만큼 이상 검출되었으면 "detector는 봤다"로 본다.
#: attribution.COVERAGE_FLOOR와 같은 값을 쓴다. 결과를 보기 전에 고정했다.
DETECTION_FLOOR = 0.5


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fmt(value, digits: int = 3) -> str:
    return "  N/A" if value is None else f"{value:.{digits}f}"


def collect(paths: list[Path], policy: str) -> list[dict]:
    events = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            entry = next(r for r in row["entryRows"] if r["policy"] == policy)
            for event in entry["events"]:
                if event["windowFrames"] == 0:
                    continue
                events.append({**event, "sequence": row["sequence"], "split": payload["split"]})
    return events


def summary_block(events: list[dict]) -> dict:
    """맞힘/놓침 두 무리의 평균을 그대로 남긴다. 보고서 숫자의 출처다."""
    out = {}
    for name, group in (
        ("matched", [e for e in events if e["matched"]]),
        ("missed", [e for e in events if not e["matched"]]),
    ):
        out[name] = {
            "n": len(group),
            "detectionCoverage": mean([e["detectionCoverage"] for e in group]),
            "trackCoverage": mean([e["trackCoverage"] for e in group]),
            "dominantTrackShare": mean([e["dominantTrackShare"] for e in group]),
            "singleTrackShare": (
                sum(1 for e in group if e["distinctTrackIds"] == 1) / len(group)
                if group
                else None
            ),
        }
    return out


def report(events: list[dict], label: str) -> list[str]:
    matched = [e for e in events if e["matched"]]
    missed = [e for e in events if not e["matched"]]
    lines = [f"### {label}  (사건 {len(events)}건: 맞힘 {len(matched)}, 놓침 {len(missed)})", ""]
    lines.append("| 집합 | n | 검출 coverage | track coverage | dominant share |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, group in (("맞힘", matched), ("놓침", missed)):
        lines.append(
            f"| {name} | {len(group)} | "
            f"{fmt(mean([e['detectionCoverage'] for e in group]))} | "
            f"{fmt(mean([e['trackCoverage'] for e in group]))} | "
            f"{fmt(mean([e['dominantTrackShare'] for e in group]))} |"
        )

    enough = [e for e in events if (e["detectionCoverage"] or 0.0) >= DETECTION_FLOOR]
    m_enough = [e for e in enough if e["matched"]]
    x_enough = [e for e in enough if not e["matched"]]
    lines += [
        "",
        f"**검출이 충분했던 사건만** (detection coverage >= {DETECTION_FLOOR}): "
        f"{len(enough)}건",
        "",
        "| 집합 | n | 검출 coverage | dominant share | 단일 track 비율 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, group in (("맞힘", m_enough), ("놓침", x_enough)):
        single = (
            sum(1 for e in group if e["distinctTrackIds"] == 1) / len(group)
            if group
            else None
        )
        lines.append(
            f"| {name} | {len(group)} | "
            f"{fmt(mean([e['detectionCoverage'] for e in group]))} | "
            f"{fmt(mean([e['dominantTrackShare'] for e in group]))} | "
            f"{fmt(single)} |"
        )

    gap_detection = None
    gap_share = None
    if m_enough and x_enough:
        gap_detection = mean([e["detectionCoverage"] for e in m_enough]) - mean(
            [e["detectionCoverage"] for e in x_enough]
        )
        gap_share = mean([e["dominantTrackShare"] for e in m_enough]) - mean(
            [e["dominantTrackShare"] for e in x_enough]
        )
    lines += [
        "",
        f"검출이 충분한 사건 안에서의 차이:  검출 coverage {fmt(gap_detection)}  /  "
        f"창 연속성 {fmt(gap_share)}",
        "",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--policy", default="P0_baseline")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    events = collect(args.results, args.policy)
    lines = [f"# H5 — policy {args.policy}", ""]
    lines += report(events, "ANALYSIS 전체")
    for split in sorted({e["split"] for e in events}):
        lines += report([e for e in events if e["split"] == split], split)

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")

    if args.json_out:
        enough = [
            e for e in events if (e["detectionCoverage"] or 0.0) >= DETECTION_FLOOR
        ]
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "policy": args.policy,
                    "detectionFloor": DETECTION_FLOOR,
                    "sources": [str(p) for p in args.results],
                    "all": summary_block(events),
                    "detectionSufficient": summary_block(enough),
                    "note": (
                        "detectionSufficient는 창 검출 coverage가 floor 이상인 "
                        "사건만 모은 것이다. 여기서도 맞힘/놓침이 갈리면 그 "
                        "차이는 검출이 아니다."
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
