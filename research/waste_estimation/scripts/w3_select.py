"""W3 2단계 — 신뢰 신호 비교와 정책 동결.

**여기서 쓰는 데이터는 DEVELOPMENT OOF뿐이다.** W2 TEST는 이미 결과를 본
집합이므로 방법이나 임계값을 고르는 데 쓰지 않는다.

선택 규칙을 결과 보기 전에 고정한다.

    Primary   AURC (위험-coverage 곡선 아래 면적, 낮을수록 좋다)
    Tie       차이가 0.002 이내이면 더 단순한 쪽
              (단일 신호 < 결합 신호, 그리고 k가 작은 쪽)

AURC를 고른 이유는 **coverage 지점을 임의로 고르지 않아도 되기** 때문이다.
"80%에서의 위험"을 기준으로 삼으면 왜 80%인지를 설명해야 하고, 그 숫자는
운영 정책이지 연구가 정할 값이 아니다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w3_select.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SERVICE = Path(__file__).resolve().parents[3] / "src"
if str(_SERVICE) not in sys.path:
    sys.path.insert(0, str(_SERVICE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from w2_prepare import LEVEL_EDGES, LEVEL_NAMES  # noqa: E402
from w3_oof import ENSEMBLE_MEMBERS, ENSEMBLE_SEED, KNN_CANDIDATES  # noqa: E402

from waste_estimation.reliability import (  # noqa: E402
    aurc,
    combine,
    oracle_rejection_risk,
    random_rejection_risk,
    risk_at_coverage_points,
    spearman,
)

OUT = Path("research/waste_estimation/experiments/w3_selective")

#: 보고할 coverage 지점. **결과를 보기 전에 고정했다.**
COVERAGE_POINTS = (1.0, 0.9, 0.8, 0.7, 0.5)

#: 선택 규칙. 위 docstring과 같은 값이며 동결 기록에 그대로 실린다.
SELECTION_PRIMARY = "AURC (낮을수록 좋다)"
SELECTION_TIE_MARGIN = 0.002

#: 무작위 제거 대조의 반복 수.
RANDOM_TRIALS = 400
RANDOM_SEED = 20260829


def load_oof(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    rows = load_oof(args.out / "oof_predictions.csv")
    errors = np.array([float(r["absError"]) for r in rows])
    targets = np.array([float(r["target"]) for r in rows])
    levels = [r["wasteLevel"] for r in rows]

    candidates: dict[str, np.ndarray] = {}
    for k in KNN_CANDIDATES:
        candidates[f"C0_knn{k}"] = np.array([float(r[f"C0_knn{k}"]) for r in rows])
    candidates["C1_disagreement"] = np.array(
        [float(r["C1_disagreement"]) for r in rows]
    )
    candidates["C2_extremeness"] = np.array([float(r["C2_extremeness"]) for r in rows])

    # --- 상관 ---------------------------------------------------------
    print("=== 신뢰 점수와 절대 오차의 관계 (DEVELOPMENT OOF, n=%d) ===" % len(rows))
    correlation_rows = []
    for name, scores in candidates.items():
        rho = spearman(scores, errors)
        correlation_rows.append({"signal": name, "spearman": rho, "aurc": aurc(errors, scores)})
        print(f"  {name:20} Spearman {rho:+.3f}   AURC {aurc(errors, scores):.4f}")

    # 결합은 가장 좋은 C0 하나와 C1을 쓴다. 후보를 전부 섞지 않는다.
    best_c0 = min(
        (row for row in correlation_rows if row["signal"].startswith("C0")),
        key=lambda row: row["aurc"],
    )["signal"]
    combined = combine(candidates[best_c0], candidates["C1_disagreement"])
    candidates["combined"] = combined
    correlation_rows.append(
        {
            "signal": "combined",
            "spearman": spearman(combined, errors),
            "aurc": aurc(errors, combined),
        }
    )
    print(
        f"  {'combined':20} Spearman {spearman(combined, errors):+.3f}   "
        f"AURC {aurc(errors, combined):.4f}   ({best_c0} + C1)"
    )

    # --- 사분위 ------------------------------------------------------
    quartile_rows = []
    for name, scores in candidates.items():
        order = scores.argsort(kind="stable")
        chunks = np.array_split(order, 4)
        for index, chunk in enumerate(chunks):
            quartile_rows.append(
                {
                    "signal": name,
                    "quartile": index + 1,
                    "count": len(chunk),
                    "meanAbsError": float(errors[chunk].mean()),
                    "medianAbsError": float(np.median(errors[chunk])),
                    "meanTarget": float(targets[chunk].mean()),
                }
            )

    print("\n=== 사분위별 오차 (1 = 가장 믿을 만함) ===")
    for name in candidates:
        values = [r for r in quartile_rows if r["signal"] == name]
        summary = "  ".join(f"Q{r['quartile']} {r['meanAbsError']:.3f}" for r in values)
        print(f"  {name:20} {summary}")

    # --- 위험-coverage ------------------------------------------------
    coverage_rows = []
    for name, scores in candidates.items():
        risks = risk_at_coverage_points(errors, scores, COVERAGE_POINTS)
        for coverage, risk in risks.items():
            coverage_rows.append(
                {
                    "signal": name,
                    "coverage": coverage,
                    "risk": risk,
                    "accepted": int(round(coverage * len(errors))),
                }
            )
    for coverage in COVERAGE_POINTS:
        coverage_rows.append(
            {
                "signal": "random_baseline",
                "coverage": coverage,
                "risk": random_rejection_risk(
                    errors, coverage, trials=RANDOM_TRIALS, seed=RANDOM_SEED
                ),
                "accepted": int(round(coverage * len(errors))),
            }
        )
        coverage_rows.append(
            {
                "signal": "oracle_upper_bound",
                "coverage": coverage,
                "risk": oracle_rejection_risk(errors, coverage),
                "accepted": int(round(coverage * len(errors))),
            }
        )

    print("\n=== coverage별 승인 표본 MAE ===")
    header = "  " + " ".join(f"{c:>7.0%}" for c in COVERAGE_POINTS)
    print(f"  {'signal':22}{header}")
    for name in list(candidates) + ["random_baseline", "oracle_upper_bound"]:
        cells = []
        for coverage in COVERAGE_POINTS:
            value = next(
                r["risk"]
                for r in coverage_rows
                if r["signal"] == name and r["coverage"] == coverage
            )
            cells.append("    N/A" if value is None else f"{value:7.4f}")
        print(f"  {name:22}" + " ".join(cells))

    # --- 고잔반 하위집단 ------------------------------------------------
    high_index = np.array([i for i, level in enumerate(levels) if level == "high"])
    subgroup_rows = []
    if len(high_index):
        print(f"\n=== 고잔반 하위집단 (n={len(high_index)}) ===")
        for name, scores in candidates.items():
            order = scores.argsort(kind="stable")
            for coverage in COVERAGE_POINTS:
                keep = int(round(coverage * len(errors)))
                accepted = set(order[:keep].tolist())
                kept_high = [i for i in high_index if i in accepted]
                subgroup_rows.append(
                    {
                        "signal": name,
                        "coverage": coverage,
                        "highAccepted": len(kept_high),
                        "highTotal": len(high_index),
                        "highRejectionRate": 1 - len(kept_high) / len(high_index),
                        "highRisk": (
                            float(errors[kept_high].mean()) if kept_high else None
                        ),
                    }
                )
        for name in candidates:
            values = [r for r in subgroup_rows if r["signal"] == name and r["coverage"] == 0.8]
            row = values[0]
            risk = "N/A" if row["highRisk"] is None else f"{row['highRisk']:.3f}"
            print(
                f"  {name:20} 80% coverage에서 고잔반 {row['highAccepted']}/{row['highTotal']} 승인, "
                f"거절률 {row['highRejectionRate']:.0%}, 승인 MAE {risk}"
            )

    # --- 수준별 내부 상관 ------------------------------------------------
    within_rows = []
    for index, level in enumerate(LEVEL_NAMES):
        mask = np.array([i for i, value in enumerate(levels) if value == level])
        if len(mask) < 10:
            within_rows.append({"level": level, "count": len(mask), "spearman": None})
            continue
        for name, scores in candidates.items():
            within_rows.append(
                {
                    "level": level,
                    "signal": name,
                    "count": len(mask),
                    "spearman": spearman(scores[mask], errors[mask]),
                }
            )

    # --- 수준 통제 선택 평가 -------------------------------------------
    # RQ-W3-3의 핵심. 전체에서 20%를 거절하면 고잔반 표본이 더 많이 빠지고
    # 그것만으로 MAE가 내려간다. 그건 "어려운 표본을 골랐다"는 뜻이 아니다.
    # 그래서 **각 잔반 수준 안에서** 같은 비율을 거절해 본다. 구성이 바뀌지
    # 않으므로 여기서도 MAE가 내려가야 진짜 식별력이다.
    stratified_rows = []
    print("\n=== 수준을 통제한 선택 평가 (각 수준 안에서 같은 비율 거절) ===")
    print(f"  {'signal':22}{header}")
    for name, scores in candidates.items():
        for coverage in COVERAGE_POINTS:
            accepted: list[int] = []
            for level in LEVEL_NAMES:
                mask = np.array([i for i, value in enumerate(levels) if value == level])
                if not len(mask):
                    continue
                order = mask[scores[mask].argsort(kind="stable")]
                accepted.extend(order[: int(round(coverage * len(mask)))].tolist())
            stratified_rows.append(
                {
                    "signal": name,
                    "coverage": coverage,
                    "accepted": len(accepted),
                    "stratifiedRisk": (
                        float(errors[accepted].mean()) if accepted else None
                    ),
                }
            )
    for name in candidates:
        cells = []
        for coverage in COVERAGE_POINTS:
            value = next(
                r["stratifiedRisk"]
                for r in stratified_rows
                if r["signal"] == name and r["coverage"] == coverage
            )
            cells.append("    N/A" if value is None else f"{value:7.4f}")
        print(f"  {name:22}" + " ".join(cells))

    # --- 범주별 --------------------------------------------------------
    # RQ-W3-4. fold를 범주로 나눴으므로 모든 OOF 예측이 이미 "처음 보는
    # 음식"이다. 신호가 특정 범주에서만 통하는지를 본다.
    category_rows = []
    for group in sorted({r["group"] for r in rows}):
        mask = np.array([i for i, r in enumerate(rows) if r["group"] == group])
        category_rows.append(
            {
                "group": group,
                "foodName": rows[mask[0]]["foodName"],
                "count": len(mask),
                "mae": float(errors[mask].mean()),
                "meanNovelty": float(candidates["C0_knn3"][mask].mean()),
                "spearman": (
                    spearman(candidates["C0_knn3"][mask], errors[mask])
                    if len(mask) >= 10
                    else None
                ),
            }
        )
    usable = [r for r in category_rows if r["spearman"] is not None]
    positive = [r for r in usable if r["spearman"] > 0]
    print(
        f"\n범주 {len(category_rows)}개 중 n>=10인 {len(usable)}개에서 "
        f"C0_knn3 상관이 양수인 범주: {len(positive)}개"
    )

    # --- 선택 --------------------------------------------------------
    ranked = sorted(correlation_rows, key=lambda row: row["aurc"])
    best_aurc = ranked[0]["aurc"]
    tied = [row for row in ranked if row["aurc"] - best_aurc <= SELECTION_TIE_MARGIN]

    def simplicity(name: str) -> tuple[int, int]:
        if name == "combined":
            return (1, 0)
        if name.startswith("C0_knn"):
            return (0, int(name.replace("C0_knn", "")))
        return (0, 0)

    chosen = min(tied, key=lambda row: (simplicity(row["signal"]), row["signal"]))
    print(
        f"\n선택: {chosen['signal']}  AURC {chosen['aurc']:.4f}  "
        f"Spearman {chosen['spearman']:+.3f}   (동률 후보 {len(tied)}개)"
    )

    # --- 기록 --------------------------------------------------------
    def write_csv(name: str, fieldnames: list[str], data: list[dict]) -> None:
        with (args.out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    write_csv("reliability_scores.csv", ["signal", "spearman", "aurc"], correlation_rows)
    write_csv(
        "reliability_quartiles.csv",
        ["signal", "quartile", "count", "meanAbsError", "medianAbsError", "meanTarget"],
        quartile_rows,
    )
    write_csv(
        "coverage_risk.csv", ["signal", "coverage", "risk", "accepted"], coverage_rows
    )
    if subgroup_rows:
        write_csv(
            "high_waste_selective.csv",
            ["signal", "coverage", "highAccepted", "highTotal", "highRejectionRate", "highRisk"],
            subgroup_rows,
        )
    write_csv("within_level_correlation.csv", ["level", "signal", "count", "spearman"], within_rows)
    write_csv(
        "stratified_coverage_risk.csv",
        ["signal", "coverage", "accepted", "stratifiedRisk"],
        stratified_rows,
    )
    write_csv(
        "category_reliability.csv",
        ["group", "foodName", "count", "mae", "meanNovelty", "spearman"],
        category_rows,
    )

    # 운영에서 쓸 임계값은 **DEVELOPMENT 점수 분포의 분위수**로 고정한다.
    # 나중 집합에서 분위수를 다시 잡으면 그 집합을 본 것이 된다.
    selected_scores = candidates[chosen["signal"]]
    thresholds = {
        f"{coverage:.2f}": float(np.quantile(selected_scores, coverage))
        for coverage in COVERAGE_POINTS
    }

    policy = {
        "stage": "W3",
        "frozenAt": "2026-08-29",
        "purpose": (
            "DEVELOPMENT OOF에서 신뢰 신호와 임계 정책을 고정한다. 이후 W2 "
            "TEST(이제 ANALYSIS)에서 이 정책을 바꾸지 않는다."
        ),
        "baseEstimator": json.loads(
            (args.out / "metadata.json").read_text(encoding="utf-8")
        )["baseEstimator"],
        "developmentOnly": True,
        "candidates": [row["signal"] for row in correlation_rows],
        "knnCandidates": list(KNN_CANDIDATES),
        "ensembleMembers": ENSEMBLE_MEMBERS,
        "ensembleSeed": ENSEMBLE_SEED,
        "combinationRule": (
            "가장 좋은 C0 하나와 C1을 rank-normalize 후 동일 가중 평균. "
            "메타 모델을 학습하지 않는다."
        ),
        "selectionPrimary": SELECTION_PRIMARY,
        "selectionTieMargin": SELECTION_TIE_MARGIN,
        "selectionTieRule": "단일 신호 < 결합 신호, 그리고 k가 작은 쪽",
        "coveragePoints": list(COVERAGE_POINTS),
        "thresholds": thresholds,
        "thresholdSource": (
            "DEVELOPMENT OOF 점수의 분위수. 이후 집합에서 다시 잡지 않는다."
        ),
        "selectedSignal": chosen["signal"],
        "selectedAurc": chosen["aurc"],
        "selectedSpearman": chosen["spearman"],
        "scoreDirection": "높을수록 덜 믿을 만함 (reliability_score)",
        "approvedRule": "reliability_score <= threshold -> APPROVED",
        "rejectedRule": "reliability_score > threshold -> REJECTED (추론 거절)",
        "calibrated": False,
        "calibrationNote": (
            "이 점수는 보정된 정확도 확률이 아니다. '0.82'가 '82% 확률로 "
            "맞다'는 뜻이 아니다."
        ),
        "forbidden": [
            "W2 TEST(ANALYSIS)를 보고 신호/임계값/결합 규칙 변경",
            "정답이나 잔차를 신뢰 신호 입력으로 사용",
            "coverage 없이 위험만 보고",
            "oracle을 실행 가능한 방법으로 표시",
            "production fixture 교체",
        ],
    }
    payload = json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    record = dict(policy)
    record["sha256OfRecordWithoutHash"] = digest
    (args.out / "freeze.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "selected_policy.yaml").write_text(
        "# DEVELOPMENT OOF에서 고른 신뢰 정책. ANALYSIS를 보기 전에 정했다.\n"
        f"signal: {chosen['signal']}\n"
        f"aurc: {chosen['aurc']:.6f}\n"
        f"spearman: {chosen['spearman']:+.6f}\n"
        f"score_direction: higher_is_less_reliable\n"
        f"calibrated: false\n",
        encoding="utf-8",
    )
    print(f"\n동결: {args.out / 'freeze.json'}\nsha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
