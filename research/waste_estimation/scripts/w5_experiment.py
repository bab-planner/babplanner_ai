"""W5 1단계 — 출력 구조 세 가지를 같은 표현 위에서 비교한다.

**DEVELOPMENT 398건만 쓴다.** W2_ANALYSIS는 여기서 열지 않는다.

바깥 fold는 W4와 **같은 분할**이다. 후보마다 새로 만들지 않는다.

표현도 정규화도 그대로다. 바뀌는 것은 "어떤 형태로 답을 내는가" 하나뿐이라,
차이가 나면 그 출처가 출력 구조밖에 없다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w5_experiment.py
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

from w2_experiment import group_folds, load_features, split_masks  # noqa: E402
from w4_experiment import classifier_scores, tail_scores  # noqa: E402

from waste_estimation.baselines import LogisticRegression, Ridge, clip  # noqa: E402
from waste_estimation.structured import (  # noqa: E402
    CUMULATIVE_THRESHOLDS,
    bin_probabilities,
    bin_representatives,
    cumulative_bin_index,
    expected_value,
    prediction_quantiles,
    soft_mixture,
)
from waste_estimation.tail import is_high_waste  # noqa: E402

W2 = Path("research/waste_estimation/experiments/w2_external")
OUT = Path("research/waste_estimation/experiments/w5_structured")

NAMES = (
    "S0_STANDARD_RIDGE",
    "S1_CUMULATIVE_ORDINAL_EXPECTATION",
    "S2_SOFT_HIGH_LOW_MIXTURE",
)


def fit_predict_all(
    train_features,
    train_targets,
    query_features,
    alpha: float,
    clf_alpha: float,
    thresholds: tuple[float, ...],
    min_expert: int,
):
    """세 후보를 **같은 학습 표본**으로 맞추고 같은 질의에 답하게 한다.

    평가 표본의 정답은 어디에도 들어가지 않는다 — 구간 대표값도, 전문가 분할도,
    게이트도 전부 학습 정답만 본다.
    """
    import numpy as np

    out: dict[str, "np.ndarray"] = {}
    diagnostics: dict[str, object] = {}

    # S0 — 점 회귀 그대로
    out["S0_STANDARD_RIDGE"] = (
        Ridge(alpha=alpha).fit(train_features, train_targets).predict(query_features)
    )

    # S1 — 임계값마다 이진 분류기 -> 누적 확률 -> 구간 확률 -> 기대값
    columns = []
    threshold_scores = []
    for t in thresholds:
        labels = (train_targets >= t).astype(float)
        if len(np.unique(labels)) < 2:
            # 학습 fold에 한 쪽만 있다. 기저 비율을 그대로 쓴다.
            columns.append(np.full(len(query_features), float(labels.mean())))
            threshold_scores.append({"threshold": t, "degenerate": True})
            continue
        model = LogisticRegression(alpha=clf_alpha, class_weight="balanced").fit(
            train_features, labels
        )
        columns.append(model.predict_proba(query_features))
        threshold_scores.append({"threshold": t, "degenerate": False})
    cumulative = np.stack(columns, axis=1)
    probabilities = bin_probabilities(cumulative)
    representatives = bin_representatives(train_targets, thresholds)
    out["S1_CUMULATIVE_ORDINAL_EXPECTATION"] = expected_value(
        probabilities, representatives
    )
    diagnostics["cumulative"] = cumulative
    diagnostics["repairedViolations"] = int(
        (np.diff(cumulative, axis=1) > 1e-9).sum()
    )
    diagnostics["representatives"] = representatives.tolist()
    diagnostics["thresholdModels"] = threshold_scores

    # S2 — 게이트 + 두 전문가의 부드러운 혼합
    high_mask = is_high_waste(train_targets)
    low_count, high_count = int((~high_mask).sum()), int(high_mask.sum())
    diagnostics["lowExpertCount"] = low_count
    diagnostics["highExpertCount"] = high_count
    if high_count < min_expert or low_count < min_expert:
        diagnostics["mixtureFailed"] = True
        out["S2_SOFT_HIGH_LOW_MIXTURE"] = out["S0_STANDARD_RIDGE"]
    else:
        diagnostics["mixtureFailed"] = False
        gate = LogisticRegression(alpha=clf_alpha, class_weight="balanced").fit(
            train_features, high_mask.astype(float)
        )
        gate_score = gate.predict_proba(query_features)
        low_expert = Ridge(alpha=alpha).fit(
            train_features[~high_mask], train_targets[~high_mask]
        )
        high_expert = Ridge(alpha=alpha).fit(
            train_features[high_mask], train_targets[high_mask]
        )
        out["S2_SOFT_HIGH_LOW_MIXTURE"] = soft_mixture(
            gate_score,
            low_expert.predict(query_features),
            high_expert.predict(query_features),
        )
        diagnostics["gateScore"] = gate_score
    return out, diagnostics


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w2", type=Path, default=W2)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    policy = json.loads((args.out / "freeze.json").read_text(encoding="utf-8"))
    alpha = float(policy["ridgeAlpha"])
    clf_alpha = float(policy["classifierAlpha"])
    thresholds = tuple(float(v) for v in policy["cumulativeThresholds"])
    min_expert = int(policy["candidates"][2]["minExpertSamples"])
    print(f"동결 해시: {policy['sha256OfRecordWithoutHash']}")
    print(f"주 지표: {policy['predictionPolicy']['primary']}")

    manifest = json.loads((args.w2 / "split_manifest.json").read_text(encoding="utf-8"))
    data = load_features(policy["representation"]["backbone"])
    dev_index, test_index = split_masks(data["row_ids"], manifest)
    features = data["after"][dev_index]
    targets = data["targets"][dev_index]
    groups = [data["groups"][i] for i in dev_index]
    row_ids = [data["row_ids"][i] for i in dev_index]
    high = is_high_waste(targets)

    print(f"\nDEVELOPMENT {len(row_ids)}건 / {len(set(groups))}범주")
    print(f"고잔반 {int(high.sum())}건 · 구간 분포 "
          f"{np.bincount(cumulative_bin_index(targets, thresholds), minlength=5).tolist()}")
    print(f"ANALYSIS는 여기서 열지 않는다 ({len(test_index)}건 보류)")

    buckets = group_folds(groups, int(policy["outerFolds"]), int(policy["outerSeed"]))
    raw = {name: np.zeros(len(targets), dtype=float) for name in NAMES}
    gate_of = np.zeros(len(targets), dtype=float)
    fold_of = np.zeros(len(targets), dtype=int)
    fold_rows: list[dict] = []
    gate_rows: list[dict] = []
    diagnostic_rows: list[dict] = []

    for fold_id, fold in enumerate(buckets):
        if not fold:
            continue
        held = np.array(fold)
        mask = np.ones(len(targets), dtype=bool)
        mask[held] = False
        train = np.flatnonzero(mask)
        fold_of[held] = fold_id
        if {groups[i] for i in fold} & {groups[i] for i in train}:
            raise SystemExit(f"바깥 fold {fold_id} 범주 누수")

        predictions, diagnostics = fit_predict_all(
            features[train],
            targets[train],
            features[held],
            alpha,
            clf_alpha,
            thresholds,
            min_expert,
        )
        for name in NAMES:
            raw[name][held] = predictions[name]
        if "gateScore" in diagnostics:
            gate_of[held] = diagnostics["gateScore"]
            scores = classifier_scores(high[held].astype(float), diagnostics["gateScore"])
            scores["fold"] = fold_id
            gate_rows.append(scores)

        diagnostic_rows.append(
            {
                "fold": fold_id,
                "trainCount": len(train),
                "lowExpertCount": diagnostics["lowExpertCount"],
                "highExpertCount": diagnostics["highExpertCount"],
                "mixtureFailed": diagnostics["mixtureFailed"],
                "monotonicViolations": diagnostics["repairedViolations"],
                "representatives": ";".join(
                    f"{value:.4f}" for value in diagnostics["representatives"]
                ),
            }
        )

        # **주 지표는 자른 예측이다.** 후보마다 범위 정책이 다르면 구조가
        # 아니라 범위를 비교하게 된다.
        for name in NAMES:
            for policy_name, values in (
                ("bounded", clip(predictions[name], 0.0, 1.0)),
                ("raw", predictions[name]),
            ):
                row = tail_scores(targets[held], values)
                row["candidate"] = name
                row["fold"] = fold_id
                row["output"] = policy_name
                fold_rows.append(row)

    print("\n=== 바깥 fold 평균 · BOUNDED (주 지표) ===")
    summary_rows = []
    for output in ("bounded", "raw"):
        for name in NAMES:
            rows = [
                row
                for row in fold_rows
                if row["candidate"] == name and row["output"] == output
            ]

            def stat(key: str) -> tuple[float | None, float | None]:
                values = [row[key] for row in rows if row[key] is not None]
                if not values:
                    return None, None
                return float(np.mean(values)), float(np.std(values))

            entry = {"candidate": name, "output": output}
            for key in (
                "mae",
                "rmse",
                "bias",
                "lowMae",
                "mediumMae",
                "highMae",
                "highBias",
                "p90AbsError",
            ):
                mean, deviation = stat(key)
                entry[key] = mean
                entry[f"{key}Std"] = deviation
            macro, _ = stat("catastrophicRate")
            entry["catastrophicRateMacro"] = macro
            entry["catastrophicCount"] = sum(row["catastrophicCount"] for row in rows)
            entry["highCount"] = sum(row["highCount"] for row in rows)
            entry["catastrophicRatePooled"] = (
                entry["catastrophicCount"] / entry["highCount"]
                if entry["highCount"]
                else None
            )
            summary_rows.append(entry)
            if output == "bounded":
                print(
                    f"  {name:36} MAE {entry['mae']:.4f}  고잔반 {entry['highMae']:.4f}  "
                    f"고잔반 Bias {entry['highBias']:+.4f}  "
                    f"치명 {entry['catastrophicCount']}/{entry['highCount']} "
                    f"({entry['catastrophicRatePooled']:.1%})  "
                    f"low {entry['lowMae']:.4f}  med {entry['mediumMae']:.4f}"
                )

    bounded = {row["candidate"]: row for row in summary_rows if row["output"] == "bounded"}

    print("\n=== S0 대비 fold 단위 개선 ===")
    stability_rows = []
    base = {
        row["fold"]: row
        for row in fold_rows
        if row["candidate"] == "S0_STANDARD_RIDGE" and row["output"] == "bounded"
    }
    for name in NAMES[1:]:
        tail_better = catastrophic_better = 0
        compared = 0
        for row in fold_rows:
            if row["candidate"] != name or row["output"] != "bounded":
                continue
            reference = base[row["fold"]]
            if row["highMae"] is None or reference["highMae"] is None:
                continue
            compared += 1
            tail_better += row["highMae"] < reference["highMae"]
            catastrophic_better += row["catastrophicCount"] < reference["catastrophicCount"]
        stability_rows.append(
            {
                "candidate": name,
                "foldsCompared": compared,
                "highMaeImproved": tail_better,
                "catastrophicImproved": catastrophic_better,
            }
        )
        print(
            f"  {name:36} 고잔반 MAE {tail_better}/{compared} fold  "
            f"치명 건수 {catastrophic_better}/{compared} fold"
        )

    if gate_rows:
        print("\n=== S2 게이트 (바깥 fold 평균) ===")
        for key in ("recall", "precision", "f1", "auroc"):
            values = [row[key] for row in gate_rows if row[key] is not None]
            print(f"  {key:10} {np.mean(values):.3f}")

    # --- 예측 분포 진단 -------------------------------------------------
    distribution_rows = [
        dict({"candidate": "GROUND_TRUTH"}, **prediction_quantiles(targets))
    ]
    for name in NAMES:
        distribution_rows.append(
            dict(
                {"candidate": name},
                **prediction_quantiles(clip(raw[name], 0.0, 1.0)),
            )
        )
    print("\n=== 예측 분포 (자른 값) ===")
    for row in distribution_rows:
        print(
            f"  {row['candidate']:36} min {row['min']:.3f}  q25 {row['q25']:.3f}  "
            f"med {row['median']:.3f}  q75 {row['q75']:.3f}  max {row['max']:.3f}  "
            f"sd {row['std']:.3f}"
        )

    # --- 선택 ------------------------------------------------------------
    reference_mae = bounded["S0_STANDARD_RIDGE"]["mae"]
    guard = float(policy["overallGuardMargin"])
    passed = [
        bounded[name] for name in NAMES if bounded[name]["mae"] <= reference_mae + guard
    ]
    print(
        f"\nGate 1 (전체 MAE <= {reference_mae + guard:.4f}) 통과: "
        f"{[row['candidate'] for row in passed]}"
    )
    best = min(row["catastrophicRatePooled"] for row in passed)
    tie = float(policy["catastrophicTieMargin"])
    tied = [row for row in passed if row["catastrophicRatePooled"] - best <= tie]
    if len(tied) > 1:
        lowest_tail = min(row["highMae"] for row in tied)
        tied = [row for row in tied if row["highMae"] <= lowest_tail + 1e-12]
    if len(tied) > 1:
        lowest_overall = min(row["mae"] for row in tied)
        tied = [row for row in tied if row["mae"] <= lowest_overall + 1e-12]
    chosen = min(tied, key=lambda row: NAMES.index(row["candidate"]))
    print(
        f"선택: {chosen['candidate']}  치명 {chosen['catastrophicRatePooled']:.1%}  "
        f"고잔반 MAE {chosen['highMae']:.4f}  전체 MAE {chosen['mae']:.4f}"
    )

    # --- 기록 ------------------------------------------------------------
    def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
        with (args.out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    write_csv(
        "fold_metrics.csv",
        [
            "candidate",
            "output",
            "fold",
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
        fold_rows,
    )
    write_csv(
        "candidate_metrics.csv",
        ["candidate", "output"]
        + [key for key in summary_rows[0] if key not in ("candidate", "output")],
        summary_rows,
    )
    write_csv(
        "fold_stability.csv",
        ["candidate", "foldsCompared", "highMaeImproved", "catastrophicImproved"],
        stability_rows,
    )
    write_csv(
        "structure_diagnostics.csv",
        [
            "fold",
            "trainCount",
            "lowExpertCount",
            "highExpertCount",
            "mixtureFailed",
            "monotonicViolations",
            "representatives",
        ],
        diagnostic_rows,
    )
    if gate_rows:
        write_csv(
            "gate_metrics.csv",
            ["fold", "positives", "recall", "precision", "f1", "auroc"],
            gate_rows,
        )
    write_csv(
        "prediction_distribution.csv",
        ["candidate", "min", "q25", "median", "q75", "max", "mean", "std"],
        distribution_rows,
    )

    with (args.out / "development_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sampleId", "group", "fold", "target", "isHighWaste", "gateScore"]
            + [f"{name}_raw" for name in NAMES]
            + [f"{name}_bounded" for name in NAMES]
        )
        for i in range(len(row_ids)):
            writer.writerow(
                [
                    row_ids[i],
                    groups[i],
                    int(fold_of[i]),
                    float(targets[i]),
                    int(high[i]),
                    float(gate_of[i]),
                ]
                + [float(raw[name][i]) for name in NAMES]
                + [float(clip(raw[name], 0.0, 1.0)[i]) for name in NAMES]
            )

    (args.out / "selected_config.yaml").write_text(
        "# DEVELOPMENT group CV에서 고른 출력 구조. ANALYSIS를 보기 전에 정했다.\n"
        f"candidate: {chosen['candidate']}\n"
        f"prediction_policy: bounded\n"
        f"overall_mae: {chosen['mae']:.6f}\n"
        f"high_waste_mae: {chosen['highMae']:.6f}\n"
        f"high_waste_bias: {chosen['highBias']:.6f}\n"
        f"catastrophic_rate_pooled: {chosen['catastrophicRatePooled']:.6f}\n"
        f"catastrophic_rate_macro: {chosen['catastrophicRateMacro']:.6f}\n"
        f"ridge_alpha: {alpha}\n"
        "selection: gate1 bounded overall MAE guard, gate2 lowest pooled catastrophic rate\n",
        encoding="utf-8",
    )
    (args.out / "metadata.json").write_text(
        json.dumps(
            {
                "stage": "W5",
                "policyHash": policy["sha256OfRecordWithoutHash"],
                "developmentCount": len(row_ids),
                "developmentGroups": len(set(groups)),
                "highWasteCount": int(high.sum()),
                "outerFolds": int(policy["outerFolds"]),
                "outerGroupLeakage": 0,
                "analysisOpened": False,
                "predictionPolicy": "bounded",
                "selected": chosen["candidate"],
                "selectionReferenceMae": reference_mae,
                "gate1Threshold": reference_mae + guard,
                "gate1Passed": [row["candidate"] for row in passed],
                "hyperparameterSearch": False,
                "representationIdenticalAcrossCandidates": True,
                "monotonicViolationsTotal": sum(
                    row["monotonicViolations"] for row in diagnostic_rows
                ),
                "mixtureFailedFolds": sum(
                    1 for row in diagnostic_rows if row["mixtureFailed"]
                ),
                "note": (
                    "구간 대표값·전문가 분할·게이트 모두 학습 fold 정답만 봤다. "
                    "평가 fold 라벨은 어디에도 들어가지 않는다."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n기록: {args.out / 'candidate_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
