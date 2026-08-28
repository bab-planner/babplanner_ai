"""A6 결과 집계.

시퀀스별 행을 정책별로 묶어 보고서에 그대로 들어가는 표를 만든다.

**macro와 micro를 둘 다 낸다.** macro는 시퀀스 평균이라 짧은 시퀀스도 같은
무게를 갖고, micro는 사건/프레임을 전부 모아 계산한다. 하나만 쓰면 긴
시퀀스 하나가 결론을 끌고 갈 수 있다.

    .venv-cctv\\Scripts\\python.exe research\\cctv_counting\\scripts\\a6_aggregate.py ^
      research\\cctv_counting\\experiments\\a6_continuity\\development.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def mean(values: list[float]) -> float | None:
    usable = [v for v in values if v is not None]
    return sum(usable) / len(usable) if usable else None


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "  N/A"
    return f"{value:.{digits}f}"


def occupancy_table(payload: dict) -> list[dict]:
    rows = []
    for threshold in payload["occupancyThresholds"]:
        macro, recalls, fps_frame, biases = [], [], [], []
        gt_total = gt_matched = 0
        for row in payload["rows"]:
            entry = next(
                r for r in row["occupancyRows"] if r["threshold"] == threshold
            )
            macro.append(entry["occupancy"]["mae"])
            biases.append(entry["occupancy"]["bias"])
            recalls.append(entry["detection"]["recall"])
            fps_frame.append(entry["detection"]["falsePositivesPerFrame"])
            gt_total += entry["detection"]["gtTotal"]
            gt_matched += entry["detection"]["gtMatched"]
        rows.append(
            {
                "threshold": threshold,
                "macroMae": mean(macro),
                "bias": mean(biases),
                "macroRecall": mean(recalls),
                "microRecall": gt_matched / gt_total if gt_total else None,
                "fpPerFrame": mean(fps_frame),
            }
        )
    return rows


def entry_table(payload: dict) -> list[dict]:
    names = list(payload["policies"])
    rows = []
    for name in names:
        totals = {
            d: {"actual": 0, "predicted": 0, "matched": 0, "fp": 0}
            for d in ("IN", "OUT")
        }
        count_errors: list[float] = []
        fragments: list[float] = []
        gaps = recovered = switches = 0
        gt_tracks = covered_tracks = total_fragments = 0
        tracker_ms: list[float] = []
        causes: dict[str, int] = {}
        matched_windows: list[float] = []
        missed_windows: list[float] = []
        matched_detection: list[float] = []
        missed_detection: list[float] = []
        for row in payload["rows"]:
            entry = next(r for r in row["entryRows"] if r["policy"] == name)
            errors = []
            for direction in ("IN", "OUT"):
                block = entry["crossing"][direction]
                totals[direction]["actual"] += block["actual"]
                totals[direction]["predicted"] += block["predicted"]
                totals[direction]["matched"] += block["matched"]
                totals[direction]["fp"] += block["falsePositives"]
                errors.append(block["countError"])
            count_errors.append(sum(errors) / 2)
            continuity = entry["continuity"]
            gt_tracks += continuity["gtTracks"]
            covered_tracks += continuity["coveredTracks"]
            total_fragments += continuity["totalFragments"]
            switches += continuity["idSwitches"]
            gaps += continuity["gaps"]
            recovered += continuity["recoveredGaps"]
            if continuity["fragmentsPerGtTrack"] is not None:
                fragments.append(continuity["fragmentsPerGtTrack"])
            if entry["trackerMsPerFrame"] is not None:
                tracker_ms.append(entry["trackerMsPerFrame"])
            for cause, value in entry["missedCauses"].items():
                causes[cause] = causes.get(cause, 0) + value
            for event in entry["events"]:
                share = event["dominantTrackShare"]
                detection = event["detectionCoverage"]
                if event["matched"]:
                    if share is not None:
                        matched_windows.append(share)
                    if detection is not None:
                        matched_detection.append(detection)
                else:
                    if share is not None:
                        missed_windows.append(share)
                    if detection is not None:
                        missed_detection.append(detection)

        precision_parts = []
        recall_parts = []
        for direction in ("IN", "OUT"):
            block = totals[direction]
            predicted = block["matched"] + block["fp"]
            precision_parts.append(
                block["matched"] / predicted if predicted else None
            )
            recall_parts.append(
                block["matched"] / block["actual"] if block["actual"] else None
            )
        precision = mean(precision_parts)
        recall = mean(recall_parts)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else None
        )
        rows.append(
            {
                "policy": name,
                "countErrorMae": mean(count_errors),
                "in": totals["IN"],
                "out": totals["OUT"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fragmentsPerGtTrack": (
                    total_fragments / covered_tracks if covered_tracks else None
                ),
                "idSwitches": switches,
                "gaps": gaps,
                "recoveredGaps": recovered,
                "gapRecoveryRate": recovered / gaps if gaps else None,
                "trackerMsPerFrame": mean(tracker_ms),
                "missedCauses": causes,
                "matchedDominantShare": mean(matched_windows),
                "missedDominantShare": mean(missed_windows),
                "matchedDetectionCoverage": mean(matched_detection),
                "missedDetectionCoverage": mean(missed_detection),
                "matchedEvents": len(matched_windows),
                "missedEvents": len(missed_windows),
            }
        )
    return rows


def render(payload: dict) -> str:
    lines = [f"split: {payload['split']}   시퀀스 {len(payload['rows'])}편", ""]
    lines.append("## occupancy 갈래 (추적 없음)")
    lines.append(
        "| T_occ | macro MAE | bias | macro recall | micro recall | FP/frame |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in occupancy_table(payload):
        lines.append(
            f"| {row['threshold']} | {fmt(row['macroMae'])} | "
            f"{fmt(row['bias'])} | {fmt(row['macroRecall'])} | "
            f"{fmt(row['microRecall'])} | {fmt(row['fpPerFrame'])} |"
        )

    lines += ["", "## entry 갈래 (통과 계수)"]
    lines.append(
        "| policy | count err | IN p/a | OUT p/a | precision | recall | F1 | "
        "frag/track | IDSW | gap rec | ms/frame |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in entry_table(payload):
        lines.append(
            f"| {row['policy']} | {fmt(row['countErrorMae'], 2)} | "
            f"{row['in']['predicted']}/{row['in']['actual']} | "
            f"{row['out']['predicted']}/{row['out']['actual']} | "
            f"{fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['f1'])} | "
            f"{fmt(row['fragmentsPerGtTrack'], 2)} | {row['idSwitches']} | "
            f"{fmt(row['gapRecoveryRate'])} | {fmt(row['trackerMsPerFrame'], 3)} |"
        )

    lines += ["", "## 놓친 통과 사건의 원인 (자동 1차 분류)"]
    causes = sorted(
        {c for row in entry_table(payload) for c in row["missedCauses"]}
    )
    lines.append("| policy | " + " | ".join(causes) + " |")
    lines.append("|---|" + "---:|" * len(causes))
    for row in entry_table(payload):
        lines.append(
            f"| {row['policy']} | "
            + " | ".join(str(row["missedCauses"].get(c, 0)) for c in causes)
            + " |"
        )

    lines += ["", "## H5 — 통과 창의 연속성 (맞힌 사건 vs 놓친 사건)"]
    lines.append(
        "| policy | 맞힘 n | 맞힘 dominant share | 놓침 n | 놓침 dominant share | "
        "맞힘 검출 coverage | 놓침 검출 coverage |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in entry_table(payload):
        lines.append(
            f"| {row['policy']} | {row['matchedEvents']} | "
            f"{fmt(row['matchedDominantShare'])} | {row['missedEvents']} | "
            f"{fmt(row['missedDominantShare'])} | "
            f"{fmt(row['matchedDetectionCoverage'])} | "
            f"{fmt(row['missedDetectionCoverage'])} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    summaries = []
    for path in args.results:
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(render(payload))
        print()
        summaries.append(
            {
                "split": payload["split"],
                "source": str(path),
                "sequences": len(payload["rows"]),
                "occupancy": occupancy_table(payload),
                "entry": entry_table(payload),
            }
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"요약: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
