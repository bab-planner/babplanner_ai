"""W3 3단계 — 동결한 정책을 W2 분석 집합에 적용한다.

**이건 테스트가 아니다.** W2 TEST 116건은 W2에서 이미 결과를 본 집합이다.
한 번 본 집합은 다시 시험지가 되지 않는다. 여기서 하는 일은 "DEVELOPMENT에서
정한 정책이 이미 아는 집합에서 어떻게 보이는가"를 기록하는 사후 분석이다.
새로운 일반화 근거가 아니다.

정책은 `freeze.json`에서 읽는다. 이 스크립트는 신호도, 임계값도, 결합 규칙도
고르지 않는다. 고를 수 있는 코드 경로 자체를 두지 않는다.

예측은 W2가 남긴 `test_predictions.csv`의 `B2_after` 열을 그대로 쓴다.
다시 학습하면 미세한 차이가 생길 수 있고, 그 차이는 이 단계가 만들어도 되는
것이 아니다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w3_analysis.py
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
_SERVICE = Path(__file__).resolve().parents[3] / "src"
if str(_SERVICE) not in sys.path:
    sys.path.insert(0, str(_SERVICE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from w2_experiment import load_features, split_masks  # noqa: E402
from w2_prepare import LEVEL_EDGES, LEVEL_NAMES  # noqa: E402
from w3_select import COVERAGE_POINTS, RANDOM_SEED, RANDOM_TRIALS  # noqa: E402

from waste_estimation.metrics import to_class_index  # noqa: E402
from waste_estimation.reliability import (  # noqa: E402
    aurc,
    knn_novelty,
    oracle_rejection_risk,
    random_rejection_risk,
    risk_at_coverage_points,
    spearman,
)

W2 = Path("research/waste_estimation/experiments/w2_external")
OUT = Path("research/waste_estimation/experiments/w3_selective")

#: 이 단계가 만들어 내는 숫자에 붙는 이름. FINAL도 GENERALIZATION도 아니다.
SPLIT_LABEL = "W2_ANALYSIS"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w2", type=Path, default=W2)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    policy = json.loads((args.out / "freeze.json").read_text(encoding="utf-8"))
    signal = policy["selectedSignal"]
    if not signal.startswith("C0_knn"):
        raise SystemExit(
            f"이 스크립트는 동결된 신호가 kNN novelty일 때만 쓸 수 있다: {signal}"
        )
    k = int(signal.replace("C0_knn", ""))
    backbone = policy["baseEstimator"]["backbone"]
    print(f"동결 정책: {signal} (k={k}), 추정기 {backbone}")
    print(f"동결 해시: {policy['sha256OfRecordWithoutHash']}")

    manifest = json.loads((args.w2 / "split_manifest.json").read_text(encoding="utf-8"))
    data = load_features(backbone)
    dev_index, test_index = split_masks(data["row_ids"], manifest)

    # 참조는 DEVELOPMENT 임베딩 전체다. 분석 집합 표본끼리는 서로를 보지 않는다.
    reference = data["after"][dev_index]
    novelty = knn_novelty(data["after"][test_index], reference, k=k)

    predictions = read_csv(args.w2 / "test_predictions.csv")
    by_id = {row["sampleId"]: row for row in predictions}
    test_ids = [data["row_ids"][i] for i in test_index]
    missing = [sample for sample in test_ids if str(sample) not in by_id]
    if missing:
        raise SystemExit(f"W2 예측에 없는 표본 {len(missing)}건: {missing[:5]}")

    targets = np.array([float(by_id[str(s)]["target"]) for s in test_ids])
    predicted = np.array([float(by_id[str(s)]["B2_after"]) for s in test_ids])
    errors = np.abs(targets - predicted)
    levels = [LEVEL_NAMES[to_class_index(float(t), LEVEL_EDGES)] for t in targets]
    groups = [by_id[str(s)]["group"] for s in test_ids]
    names = [by_id[str(s)]["foodName"] for s in test_ids]

    print(f"\n{SPLIT_LABEL} {len(test_ids)}건 / {len(set(groups))}범주")
    print(f"전체 MAE {errors.mean():.4f}   Spearman(신호, 오차) {spearman(novelty, errors):+.3f}")
    print(f"AURC {aurc(errors, novelty):.4f}")

    # --- 동결 임계값을 그대로 적용 --------------------------------------
    # DEVELOPMENT 분위수를 그대로 쓰므로 실현 coverage는 80%가 아닐 수 있다.
    # 그 차이 자체가 분포가 옮겨갔다는 신호다. 여기서 임계값을 다시 잡으면
    # 분석 집합을 보고 정책을 고친 것이 된다.
    frozen_rows = []
    print("\n=== 동결 임계값 적용 (분위수를 다시 잡지 않는다) ===")
    for point in COVERAGE_POINTS:
        threshold = float(policy["thresholds"][f"{point:.2f}"])
        accepted = novelty <= threshold
        realized = float(accepted.mean())
        risk = float(errors[accepted].mean()) if accepted.any() else None
        frozen_rows.append(
            {
                "split": SPLIT_LABEL,
                "plannedCoverage": point,
                "threshold": threshold,
                "realizedCoverage": realized,
                "accepted": int(accepted.sum()),
                "risk": risk,
            }
        )
        shown = "N/A" if risk is None else f"{risk:.4f}"
        print(
            f"  목표 {point:>5.0%}  임계 {threshold:.4f}  "
            f"실현 {realized:>6.1%} ({int(accepted.sum())}건)  승인 MAE {shown}"
        )

    # --- 참고용 분위수 곡선 ---------------------------------------------
    # 위 표와 목적이 다르다. 여기서는 DEVELOPMENT 곡선과 모양을 견주기 위해
    # 분석 집합 안에서 순위로 자른다. 운영 정책이 아니라 비교용이다.
    curve_rows = []
    risks = risk_at_coverage_points(errors, novelty, COVERAGE_POINTS)
    for coverage, risk in risks.items():
        curve_rows.append(
            {"split": SPLIT_LABEL, "signal": signal, "coverage": coverage, "risk": risk}
        )
    for coverage in COVERAGE_POINTS:
        curve_rows.append(
            {
                "split": SPLIT_LABEL,
                "signal": "random_baseline",
                "coverage": coverage,
                "risk": random_rejection_risk(
                    errors, coverage, trials=RANDOM_TRIALS, seed=RANDOM_SEED
                ),
            }
        )
        curve_rows.append(
            {
                "split": SPLIT_LABEL,
                "signal": "oracle_upper_bound",
                "coverage": coverage,
                "risk": oracle_rejection_risk(errors, coverage),
            }
        )
    print("\n=== 순위 기준 곡선 (DEVELOPMENT와 모양 비교용) ===")
    header = " ".join(f"{c:>7.0%}" for c in COVERAGE_POINTS)
    print(f"  {'signal':22}{header}")
    for name in (signal, "random_baseline", "oracle_upper_bound"):
        cells = []
        for coverage in COVERAGE_POINTS:
            value = next(
                r["risk"]
                for r in curve_rows
                if r["signal"] == name and r["coverage"] == coverage
            )
            cells.append("    N/A" if value is None else f"{value:7.4f}")
        print(f"  {name:22}" + " ".join(cells))

    # --- 수준 통제 -------------------------------------------------------
    for point in (0.8, 0.5):
        accepted_index: list[int] = []
        for level in LEVEL_NAMES:
            mask = np.array([i for i, value in enumerate(levels) if value == level])
            if not len(mask):
                continue
            order = mask[novelty[mask].argsort(kind="stable")]
            accepted_index.extend(order[: int(round(point * len(mask)))].tolist())
        if accepted_index:
            curve_rows.append(
                {
                    "split": SPLIT_LABEL,
                    "signal": f"{signal}_level_controlled",
                    "coverage": point,
                    "risk": float(errors[accepted_index].mean()),
                }
            )

    # --- 실패 분석 -------------------------------------------------------
    # 가장 중요한 칸: 신호가 "믿을 만하다"고 본 표본 중 오차가 큰 것들.
    # 선택 예측이 조용히 통과시키는 오류가 여기 있다.
    threshold_80 = float(policy["thresholds"]["0.80"])
    failure_rows = []
    for index, sample in enumerate(test_ids):
        approved = bool(novelty[index] <= threshold_80)
        failure_rows.append(
            {
                "sampleId": sample,
                "group": groups[index],
                "foodName": names[index],
                "target": float(targets[index]),
                "prediction": float(predicted[index]),
                "absError": float(errors[index]),
                "wasteLevel": levels[index],
                "reliabilityScore": float(novelty[index]),
                "decisionAt80": "APPROVED" if approved else "REJECTED",
                "caughtLargeError": (not approved) and float(errors[index]) >= 0.30,
                "missedLargeError": approved and float(errors[index]) >= 0.30,
            }
        )
    caught = [r for r in failure_rows if r["caughtLargeError"]]
    missed = [r for r in failure_rows if r["missedLargeError"]]
    large = [r for r in failure_rows if r["absError"] >= 0.30]
    print(f"\n=== 큰 오차(>=0.30) {len(large)}건의 처리 (동결 80% 임계) ===")
    print(f"  거절해서 걸러낸 것 {len(caught)}건, 승인해서 놓친 것 {len(missed)}건")
    for row in sorted(missed, key=lambda r: -r["absError"])[:5]:
        print(
            f"    놓침  {row['foodName']:<22} 정답 {row['target']:.2f} "
            f"예측 {row['prediction']:.2f} 오차 {row['absError']:.2f} "
            f"신뢰점수 {row['reliabilityScore']:.3f}"
        )

    # --- 기록 -----------------------------------------------------------
    def write_csv(name: str, fieldnames: list[str], data: list[dict]) -> None:
        with (args.out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    write_csv(
        "analysis_metrics.csv",
        ["split", "plannedCoverage", "threshold", "realizedCoverage", "accepted", "risk"],
        frozen_rows,
    )
    write_csv(
        "analysis_coverage_risk.csv", ["split", "signal", "coverage", "risk"], curve_rows
    )
    write_csv(
        "analysis_failure_cases.csv",
        [
            "sampleId",
            "group",
            "foodName",
            "target",
            "prediction",
            "absError",
            "wasteLevel",
            "reliabilityScore",
            "decisionAt80",
            "caughtLargeError",
            "missedLargeError",
        ],
        failure_rows,
    )

    (args.out / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "stage": "W3",
                "split": SPLIT_LABEL,
                "isFinalTest": False,
                "isGeneralizationTest": False,
                "note": (
                    "W2에서 이미 결과를 본 집합이다. 여기 숫자는 사후 분석이며 "
                    "새로운 일반화 근거가 아니다."
                ),
                "policyHash": policy["sha256OfRecordWithoutHash"],
                "selectedSignal": signal,
                "sampleCount": len(test_ids),
                "groupCount": len(set(groups)),
                "mae": float(errors.mean()),
                "spearman": spearman(novelty, errors),
                "aurc": aurc(errors, novelty),
                "largeErrorCount": len(large),
                "largeErrorCaught": len(caught),
                "largeErrorMissed": len(missed),
                "predictionSource": "w2_external/test_predictions.csv B2_after",
                "refit": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n기록: {args.out / 'analysis_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
