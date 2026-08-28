"""A5 결과를 표로 내보낸다 (§47).

JSON은 기계가 읽기 좋고 CSV는 사람이 표계산기에서 열어 보기 좋다. 둘 다
남긴다. **여기서 새로 계산하는 값은 없다.** 이미 기록된 수치를 옮기기만
한다. 표를 만드는 김에 값을 조금 다르게 계산하면 두 파일이 어긋나고,
나중에 어느 쪽이 맞는지 알 수 없게 된다.

원본 프레임이나 사람 이미지는 내보내지 않는다. 집계 수치만이다.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  {path.name}  ({len(rows)} 행)")


def metrics_rows(payload: dict) -> list[list]:
    rows = []
    for r in payload["rows"]:
        d, o = r["detection"], r["occupancy"]
        rows.append([
            r["candidate"], r["sequence"], r["scoredFrames"],
            d["gtTotal"], d["gtMatched"],
            round(d["recall"], 4) if d["recall"] is not None else "",
            round(d["precision"], 4) if d["precision"] is not None else "",
            round(d["falsePositivesPerFrame"], 4)
            if d["falsePositivesPerFrame"] is not None else "",
            round(o["mae"], 4) if o["mae"] is not None else "",
            round(o["rmse"], 4) if o["rmse"] is not None else "",
            round(o["bias"], 4) if o["bias"] is not None else "",
            r["crossing"]["IN"]["actual"], r["crossing"]["IN"]["predicted"],
            r["crossing"]["IN"]["matched"],
            r["crossing"]["OUT"]["actual"], r["crossing"]["OUT"]["predicted"],
            r["crossing"]["OUT"]["matched"],
        ])
    return rows


METRICS_HEADER = [
    "candidate", "sequence", "scored_frames", "gt_boxes", "gt_matched",
    "det_recall", "det_precision", "fp_per_frame",
    "occ_mae", "occ_rmse", "occ_bias",
    "in_actual", "in_predicted", "in_matched",
    "out_actual", "out_predicted", "out_matched",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args(argv)
    d = args.dir

    for name, source in (("development_metrics.csv", "development.json"),
                         ("final_test_metrics.csv", "final_test.json")):
        path = d / source
        if not path.exists():
            print(f"  건너뜀 (없음): {source}")
            continue
        write(d / name, METRICS_HEADER,
              metrics_rows(json.loads(path.read_text(encoding="utf-8"))))

    # 크기별 recall. 표본 수를 함께 적는다. 표본 없는 칸은 빈칸이지 0이 아니다.
    size_rows = []
    for split, source in (("DEV", "development.json"), ("FINAL", "final_test.json")):
        path = d / source
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        totals: dict[tuple[str, str], list[int]] = {}
        for r in payload["rows"]:
            for bin_name, v in r["detection"]["bySize"].items():
                key = (r["candidate"], bin_name)
                acc = totals.setdefault(key, [0, 0])
                acc[0] += v["total"]
                acc[1] += v["matched"]
        for (candidate, bin_name), (total, matched) in sorted(totals.items()):
            size_rows.append([
                split, candidate, bin_name, total, matched,
                round(matched / total, 4) if total else "",
            ])
    if size_rows:
        write(d / "size_recall.csv",
              ["split", "candidate", "size_bin", "gt_boxes", "matched", "recall"],
              size_rows)

    # 통과 사건별 크기와 회복 여부.
    path = d / "crossing_size.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for seq in payload["sequences"]:
            chain = seq.get("chain")
            if not chain:
                continue
            for row in chain["rows"]:
                rows.append([
                    seq["sequence"], row["trackId"], row["frame"], row["direction"],
                    row["normalizedHeight"],
                    round(row["baseTrackRecall"], 4)
                    if row["baseTrackRecall"] is not None else "",
                    round(row["contrastTrackRecall"], 4)
                    if row["contrastTrackRecall"] is not None else "",
                    int(row["eventRecovered"]),
                ])
        write(d / "crossing_size_analysis.csv",
              ["sequence", "track_id", "frame", "direction", "normalized_height",
               "baseline_track_recall", "contrast_track_recall", "event_recovered"],
              rows)
    else:
        print("  건너뜀 (없음): crossing_size.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
