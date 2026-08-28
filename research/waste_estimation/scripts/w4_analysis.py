"""W4 3단계 — 동결한 것을 W2 분석 집합에 정확히 한 번 적용한다.

> **이건 테스트가 아니다.** W2 TEST 116건은 W2에서 결과를 이미 본 집합이고,
> W3에서 `W2_ANALYSIS`로 재분류했다. 여기 숫자는 POST-HOC ANALYSIS다.
> FINAL TEST도, pristine validation도, 일반화 근거도 아니다.

이 스크립트는 아무것도 고르지 않는다. 후보도, 가중치도, 분류기 임계값도,
신뢰 정규화 방식도 전부 앞 단계 기록에서 읽는다.

세 후보를 모두 적용하는 이유는 선택을 다시 하기 위해서가 아니라 —
선택은 `metadata.json`에 이미 봉해져 있다 — **DEVELOPMENT에서 본 맞교환이
여기서도 같은 모양인지** 보기 위해서다. 어느 숫자를 보든 선택은 바뀌지 않는다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w4_analysis.py
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
from w2_prepare import ORDINAL_EDGES  # noqa: E402
from w4_experiment import classifier_scores, inner_oof_probability, tail_scores  # noqa: E402

from waste_estimation.baselines import LogisticRegression, Ridge  # noqa: E402
from waste_estimation.reliability import knn_novelty, spearman  # noqa: E402
from waste_estimation.tail import (  # noqa: E402
    catastrophic_underestimation,
    coverage_drift,
    inverse_sqrt_bin_weights,
    is_high_waste,
    normalized_novelty,
    realized_coverage,
)

W2 = Path("research/waste_estimation/experiments/w2_external")
OUT = Path("research/waste_estimation/experiments/w4_tail")

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
    development = json.loads((args.out / "metadata.json").read_text(encoding="utf-8"))
    decision = json.loads(
        (args.out / "reliability_decision.json").read_text(encoding="utf-8")
    )
    selected = development["selected"]
    reliability_method = (
        "C0_NORMALIZED" if decision["adoptNormalized"] else "C0_RAW"
    )
    print(f"동결 해시: {policy['sha256OfRecordWithoutHash']}")
    print(f"선택된 추정기: {selected}   신뢰 방식: {reliability_method}")
    print("여기서는 아무것도 고르지 않는다.")

    manifest = json.loads((args.w2 / "split_manifest.json").read_text(encoding="utf-8"))
    data = load_features(policy["representation"]["backbone"])
    dev_index, test_index = split_masks(data["row_ids"], manifest)
    dev_features, dev_targets = data["after"][dev_index], data["targets"][dev_index]
    dev_groups = [data["groups"][i] for i in dev_index]
    features = data["after"][test_index]
    targets = data["targets"][test_index]
    groups = [data["groups"][i] for i in test_index]
    names = [data["names"][i] for i in test_index]
    row_ids = [data["row_ids"][i] for i in test_index]
    labels = is_high_waste(targets).astype(float)
    print(
        f"\n{SPLIT_LABEL} {len(row_ids)}건 / {len(set(groups))}범주 · "
        f"고잔반 {int(labels.sum())}건"
    )

    alpha = float(policy["ridgeAlpha"])
    clf_alpha = float(policy["classifierAlpha"])
    dev_labels = is_high_waste(dev_targets).astype(float)

    predictions: dict[str, "np.ndarray"] = {}
    predictions["R0_STANDARD"] = (
        Ridge(alpha=alpha).fit(dev_features, dev_targets).predict(features)
    )
    predictions["R1_BALANCED"] = (
        Ridge(alpha=alpha)
        .fit(
            dev_features,
            dev_targets,
            sample_weight=inverse_sqrt_bin_weights(dev_targets, ORDINAL_EDGES),
        )
        .predict(features)
    )
    p_dev = inner_oof_probability(
        dev_features,
        dev_labels,
        dev_groups,
        int(policy["innerFolds"]),
        int(policy["innerSeed"]),
        clf_alpha,
    )
    classifier = LogisticRegression(alpha=clf_alpha, class_weight="balanced").fit(
        dev_features, dev_labels
    )
    p_analysis = classifier.predict_proba(features)
    predictions["R2_HIGH_AUX"] = (
        Ridge(alpha=alpha)
        .fit(np.hstack([dev_features, p_dev[:, None]]), dev_targets)
        .predict(np.hstack([features, p_analysis[:, None]]))
    )

    print("\n=== POST-HOC ANALYSIS (선택은 이미 끝났다) ===")
    metric_rows = []
    for name in ("R0_STANDARD", "R1_BALANCED", "R2_HIGH_AUX"):
        scores = tail_scores(targets, predictions[name])
        scores["candidate"] = name
        scores["split"] = SPLIT_LABEL
        scores["isSelected"] = name == selected
        metric_rows.append(scores)
        print(
            f"  {name:14} 전체 MAE {scores['mae']:.4f}  RMSE {scores['rmse']:.4f}  "
            f"Bias {scores['bias']:+.4f}  고잔반 MAE {scores['highMae']:.4f}  "
            f"고잔반 Bias {scores['highBias']:+.4f}  "
            f"치명 {scores['catastrophicCount']}/{scores['highCount']} "
            f"({scores['catastrophicRate']:.1%})"
        )

    clf = classifier_scores(labels, p_analysis)
    print(
        f"\n고잔반 분류기: recall {clf['recall']:.3f}  precision {clf['precision']:.3f}  "
        f"F1 {clf['f1']:.3f}  AUROC {clf['auroc']:.3f}"
    )

    # --- W2 대표 오류의 회복 여부 -----------------------------------------
    # 이 표본들 때문에 모델을 다시 바꾸지 않는다. 기록만 한다.
    w2_failures = read_csv(args.w2 / "failure_cases.csv")[:3]
    by_id = {row_ids[i]: i for i in range(len(row_ids))}
    recovery_rows = []
    print("\n=== W2 대표 오류 3건 (모델을 바꾸지 않는다. 기록만) ===")
    for failure in w2_failures:
        sample = failure.get("sampleId")
        if sample not in by_id:
            continue
        index = by_id[sample]
        entry = {
            "sampleId": sample,
            "foodName": names[index],
            "target": float(targets[index]),
        }
        for name in predictions:
            entry[name] = float(predictions[name][index])
            entry[f"{name}_absError"] = abs(
                float(predictions[name][index]) - float(targets[index])
            )
        recovery_rows.append(entry)
        print(
            f"  {names[index]:<26} 정답 {entry['target']:.2f}  "
            f"R0 {entry['R0_STANDARD']:.2f}  R1 {entry['R1_BALANCED']:.2f}  "
            f"R2 {entry['R2_HIGH_AUX']:.2f}"
        )

    # --- 치명적 과소추정의 이동 -------------------------------------------
    base = catastrophic_underestimation(predictions["R0_STANDARD"], targets)
    balanced = catastrophic_underestimation(predictions["R1_BALANCED"], targets)
    improved = np.flatnonzero(base & ~balanced)
    remaining = np.flatnonzero(base & balanced)
    introduced = np.flatnonzero(~base & balanced)
    print(
        f"\n=== 치명적 과소추정의 이동 (R0 -> R1, 서술일 뿐 선택 아님) ===\n"
        f"  회복 {len(improved)}건 · 그대로 {len(remaining)}건 · 새로 생김 {len(introduced)}건"
    )

    failure_rows = []
    for index in range(len(row_ids)):
        failure_rows.append(
            {
                "sampleId": row_ids[index],
                "group": groups[index],
                "foodName": names[index],
                "target": float(targets[index]),
                "isHighWaste": int(labels[index]),
                "pHigh": float(p_analysis[index]),
                "R0_STANDARD": float(predictions["R0_STANDARD"][index]),
                "R1_BALANCED": float(predictions["R1_BALANCED"][index]),
                "R2_HIGH_AUX": float(predictions["R2_HIGH_AUX"][index]),
                "R0_catastrophic": bool(base[index]),
                "R1_catastrophic": bool(balanced[index]),
                "movement": (
                    "recovered"
                    if base[index] and not balanced[index]
                    else "introduced"
                    if balanced[index] and not base[index]
                    else "remaining"
                    if base[index]
                    else "none"
                ),
            }
        )

    # --- 신뢰 임계값 이식 --------------------------------------------------
    # W3이 실패한 자리를 그대로 다시 밟는다. 참조는 DEVELOPMENT 전체(398건),
    # 임계값은 DEVELOPMENT OOF 점수(참조 ~318건)에서 잡은 값이다.
    calibration = read_csv(args.out / "development_reliability.csv")
    scores = {
        "C0_RAW": knn_novelty(
            features, dev_features, k=int(policy["reliability"]["knnK"])
        ),
        "C0_NORMALIZED": normalized_novelty(
            features, dev_features, k=int(policy["reliability"]["knnK"])
        ),
    }
    errors = np.abs(targets - predictions[selected])
    transport_rows = []
    print(
        f"\n=== 신뢰 임계값 이식 (참조 {len(dev_features)}건, "
        f"보정은 OOF 참조 ~318건) ==="
    )
    header = " ".join(f"{c:>7.0%}" for c in policy["reliability"]["coverageTargets"])
    print(f"  {'method':16}{header}   평균 drift")
    for method, values in scores.items():
        calibration_scores = np.array([float(row[method]) for row in calibration])
        cells, drifts = [], []
        for coverage in policy["reliability"]["coverageTargets"]:
            coverage = float(coverage)
            threshold = float(np.quantile(calibration_scores, coverage))
            realized = realized_coverage(values, threshold)
            accepted = values <= threshold
            drift = coverage_drift(coverage, realized)
            cells.append(f"{realized:7.1%}")
            drifts.append(drift)
            transport_rows.append(
                {
                    "split": SPLIT_LABEL,
                    "method": method,
                    "targetCoverage": coverage,
                    "threshold": threshold,
                    "realizedCoverage": realized,
                    "coverageDrift": drift,
                    "accepted": int(accepted.sum()),
                    "acceptedMae": (
                        float(errors[accepted].mean()) if accepted.any() else None
                    ),
                }
            )
        print(f"  {method:16}" + " ".join(cells) + f"   {float(np.mean(drifts)):.3f}")

    high = labels.astype(bool)
    tail_rows = []
    for method, values in scores.items():
        tail_rows.append(
            {
                "split": SPLIT_LABEL,
                "method": method,
                "scope": "all",
                "count": len(targets),
                "spearman": spearman(values, errors),
            }
        )
        tail_rows.append(
            {
                "split": SPLIT_LABEL,
                "method": method,
                "scope": "high_waste_only",
                "count": int(high.sum()),
                "spearman": spearman(values[high], errors[high]),
            }
        )
    print("\n=== 신뢰 점수와 오차의 상관 ===")
    for row in tail_rows:
        print(
            f"  {row['method']:16} {row['scope']:16} n={row['count']:>3}  "
            f"Spearman {row['spearman']:+.3f}"
        )

    # --- 기록 -------------------------------------------------------------
    def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
        with (args.out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    write_csv(
        "analysis_metrics.csv",
        [
            "split",
            "candidate",
            "isSelected",
            "count",
            "mae",
            "rmse",
            "bias",
            "lowMae",
            "mediumMae",
            "highMae",
            "highBias",
            "highCount",
            "catastrophicCount",
            "catastrophicRate",
            "p90AbsError",
        ],
        metric_rows,
    )
    write_csv(
        "analysis_reliability_transport.csv",
        [
            "split",
            "method",
            "targetCoverage",
            "threshold",
            "realizedCoverage",
            "coverageDrift",
            "accepted",
            "acceptedMae",
        ],
        transport_rows,
    )
    write_csv(
        "analysis_tail_reliability.csv",
        ["split", "method", "scope", "count", "spearman"],
        tail_rows,
    )
    write_csv(
        "failure_cases.csv",
        [
            "sampleId",
            "group",
            "foodName",
            "target",
            "isHighWaste",
            "pHigh",
            "R0_STANDARD",
            "R1_BALANCED",
            "R2_HIGH_AUX",
            "R0_catastrophic",
            "R1_catastrophic",
            "movement",
        ],
        failure_rows,
    )
    if recovery_rows:
        write_csv(
            "w2_error_recovery.csv", list(recovery_rows[0]), recovery_rows
        )

    (args.out / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "stage": "W4",
                "split": SPLIT_LABEL,
                "isFinalTest": False,
                "isGeneralizationTest": False,
                "isPristineValidation": False,
                "resultName": "POST-HOC ANALYSIS",
                "note": (
                    "W2에서 이미 결과를 본 집합이다. 선택은 DEVELOPMENT에서 "
                    "끝났고 여기서 바뀌지 않는다."
                ),
                "policyHash": policy["sha256OfRecordWithoutHash"],
                "selected": selected,
                "selectionMadeOn": "DEVELOPMENT nested group CV",
                "reliabilityMethod": reliability_method,
                "applications": 1,
                "sampleCount": len(row_ids),
                "groupCount": len(set(groups)),
                "highWasteCount": int(labels.sum()),
                "classifier": clf,
                "catastrophicMovement": {
                    "recovered": len(improved),
                    "remaining": len(remaining),
                    "introduced": len(introduced),
                },
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
