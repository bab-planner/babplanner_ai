"""W4 1단계 — 중첩 group CV로 꼬리 인지 후보를 비교한다.

**DEVELOPMENT 398건만 쓴다.** W2_ANALYSIS는 여기서 열지 않는다.

바깥 fold는 W2/W3과 같은 분할이다. 다르게 잡으면 이전 숫자와 나란히 놓을 수
없고, 개선처럼 보이는 것이 분할 차이일 수 있다.

안쪽 fold는 **R2의 p_high를 만드는 데만** 쓴다. 여기가 누수가 나기 가장
쉬운 자리다 — 학습 표본의 p_high를 그 표본의 라벨로 학습한 분류기에서 뽑으면,
회귀기는 정답을 반쯤 들여다보게 된다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w4_experiment.py
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
from w2_prepare import LEVEL_EDGES, LEVEL_NAMES, ORDINAL_EDGES  # noqa: E402

from waste_estimation.baselines import LogisticRegression, Ridge  # noqa: E402
from waste_estimation.metrics import unbounded_bias, unbounded_mae  # noqa: E402
from waste_estimation.metrics import unbounded_rmse  # noqa: E402
from waste_estimation.tail import (  # noqa: E402
    catastrophic_underestimation,
    inverse_sqrt_bin_weights,
    is_high_waste,
    percentile_abs_error,
)

W2 = Path("research/waste_estimation/experiments/w2_external")
OUT = Path("research/waste_estimation/experiments/w4_tail")


def auroc(labels, scores) -> float | None:
    """순위 기반 AUROC. 동점은 평균 순위로."""
    import numpy as np

    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())
    if positive == 0 or negative == 0:
        return None
    order = s.argsort(kind="stable")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    unique, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    for index, count in enumerate(counts):
        if count > 1:
            mask = inverse == index
            ranks[mask] = ranks[mask].mean()
    return float((ranks[y == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def classifier_scores(labels, probability, threshold: float = 0.5) -> dict:
    """고잔반 분류기 지표. **주 지표는 recall이다.**

    제품이 무서워하는 것은 고잔반을 놓치는 쪽이다. precision이 낮으면 사람이
    헛되이 한 번 더 보지만, recall이 낮으면 남은 음식이 그대로 지나간다.
    """
    import numpy as np

    y = np.asarray(labels, dtype=int)
    predicted = (np.asarray(probability, dtype=float) >= threshold).astype(int)
    true_positive = int(((y == 1) & (predicted == 1)).sum())
    false_positive = int(((y == 0) & (predicted == 1)).sum())
    false_negative = int(((y == 1) & (predicted == 0)).sum())
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else None
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall
        else 0.0
        if recall is not None
        else None
    )
    return {
        "positives": int((y == 1).sum()),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "auroc": auroc(y, probability),
    }


def tail_scores(targets, predictions) -> dict:
    """꼬리를 따로 재는 지표 묶음.

    전체 MAE만 보면 고잔반 89건의 실패가 저잔반 203건에 묻힌다. W2에서 실제로
    그렇게 묻혀 있었다.
    """
    import numpy as np

    actual = np.asarray(targets, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    high = is_high_waste(actual)
    # 수준 구분을 `is_high_waste`와 **같은 경계**로 맞춘다. W2/W3의 LEVEL_EDGES는
    # `to_class_index` 규약상 0.8을 아래 칸에 넣어서 high가 89건이었다. W4의
    # 동결 규칙은 `>= 0.8`이라 90건이다. 정확히 0.8인 표본 1건 차이이며,
    # 한 표에서 두 규칙을 섞으면 그 1건이 medium과 high에 동시에 세어진다.
    levels = np.where(
        high, "high", np.where(actual <= LEVEL_EDGES[0], "low", "medium")
    )
    scores = {
        "count": len(actual),
        "mae": unbounded_mae(actual.tolist(), predicted.tolist()),
        "rmse": unbounded_rmse(actual.tolist(), predicted.tolist()),
        "bias": unbounded_bias(actual.tolist(), predicted.tolist()),
        "p90AbsError": percentile_abs_error(predicted, actual, 90.0),
        "highCount": int(high.sum()),
    }
    for level in LEVEL_NAMES:
        mask = levels == level
        scores[f"{level}Mae"] = (
            unbounded_mae(actual[mask].tolist(), predicted[mask].tolist())
            if mask.any()
            else None
        )
    if high.any():
        scores["highMae"] = unbounded_mae(
            actual[high].tolist(), predicted[high].tolist()
        )
        scores["highBias"] = unbounded_bias(
            actual[high].tolist(), predicted[high].tolist()
        )
        catastrophic = catastrophic_underestimation(predicted, actual)
        scores["catastrophicCount"] = int(catastrophic.sum())
        scores["catastrophicRate"] = float(catastrophic.sum()) / int(high.sum())
    else:
        scores.update(
            {
                "highMae": None,
                "highBias": None,
                "catastrophicCount": 0,
                "catastrophicRate": None,
            }
        )
    return scores


def inner_oof_probability(features, labels, groups, folds: int, seed: int, alpha: float):
    """학습 fold 안에서 group OOF로 p_high를 만든다.

    **이 함수가 없으면 R2는 누수 모델이다.** 자기 라벨로 학습한 분류기의
    확률을 자기 특징에 붙이면, 회귀기는 임베딩이 아니라 그 확률에 실린 정답을
    읽는다. 그러면 개발 성능만 좋아지고 새 표본에서 무너진다.
    """
    import numpy as np

    probability = np.zeros(len(labels), dtype=float)
    buckets = group_folds(groups, folds, seed)
    for fold in buckets:
        if not fold:
            continue
        held = np.array(fold)
        mask = np.ones(len(labels), dtype=bool)
        mask[held] = False
        inner_train = np.flatnonzero(mask)
        if len(np.unique(labels[inner_train])) < 2:
            # 이 안쪽 fold에는 한 클래스만 있다. 기저 비율로 채운다.
            probability[held] = float(labels[inner_train].mean())
            continue
        model = LogisticRegression(alpha=alpha, class_weight="balanced").fit(
            features[inner_train], labels[inner_train]
        )
        probability[held] = model.predict_proba(features[held])
    return probability


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w2", type=Path, default=W2)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    policy = json.loads((args.out / "freeze.json").read_text(encoding="utf-8"))
    alpha = float(policy["ridgeAlpha"])
    clf_alpha = float(policy["classifierAlpha"])
    outer_folds = int(policy["outerFolds"])
    outer_seed = int(policy["outerSeed"])
    inner_folds = int(policy["innerFolds"])
    inner_seed = int(policy["innerSeed"])
    print(f"동결 해시: {policy['sha256OfRecordWithoutHash']}")

    manifest = json.loads((args.w2 / "split_manifest.json").read_text(encoding="utf-8"))
    data = load_features(policy["representation"]["backbone"])
    dev_index, test_index = split_masks(data["row_ids"], manifest)
    features = data["after"][dev_index]
    targets = data["targets"][dev_index]
    groups = [data["groups"][i] for i in dev_index]
    row_ids = [data["row_ids"][i] for i in dev_index]
    labels = is_high_waste(targets).astype(float)

    print(f"DEVELOPMENT {len(row_ids)}건 / {len(set(groups))}범주")
    print(f"고잔반 {int(labels.sum())}건 ({labels.mean():.1%})")
    print(f"ANALYSIS는 여기서 열지 않는다 ({len(test_index)}건 보류)")

    buckets = group_folds(groups, outer_folds, outer_seed)
    names = ["R0_STANDARD", "R1_BALANCED", "R2_HIGH_AUX"]
    predictions = {name: np.zeros(len(targets), dtype=float) for name in names}
    p_high_oof = np.zeros(len(targets), dtype=float)
    fold_of = np.zeros(len(targets), dtype=int)
    fold_rows: list[dict] = []
    classifier_rows: list[dict] = []

    for fold_id, fold in enumerate(buckets):
        if not fold:
            continue
        held = np.array(fold)
        mask = np.ones(len(targets), dtype=bool)
        mask[held] = False
        train = np.flatnonzero(mask)
        fold_of[held] = fold_id

        held_groups = {groups[i] for i in fold}
        train_groups = {groups[i] for i in train}
        if held_groups & train_groups:
            raise SystemExit(f"바깥 fold {fold_id} 범주 누수: {held_groups & train_groups}")

        # R0 — 표준
        r0 = Ridge(alpha=alpha).fit(features[train], targets[train])
        predictions["R0_STANDARD"][held] = r0.predict(features[held])

        # R1 — 구간 균형 가중치. **빈도는 학습 fold 정답에서만 센다.**
        weights = inverse_sqrt_bin_weights(targets[train], ORDINAL_EDGES)
        r1 = Ridge(alpha=alpha).fit(
            features[train], targets[train], sample_weight=weights
        )
        predictions["R1_BALANCED"][held] = r1.predict(features[held])

        # R2 — 보조 분류기. 학습 표본은 안쪽 OOF, 평가 표본은 학습 전체 적합.
        train_groups_list = [groups[i] for i in train]
        p_train = inner_oof_probability(
            features[train],
            labels[train],
            train_groups_list,
            inner_folds,
            inner_seed + fold_id,
            clf_alpha,
        )
        classifier = LogisticRegression(alpha=clf_alpha, class_weight="balanced").fit(
            features[train], labels[train]
        )
        p_held = classifier.predict_proba(features[held])
        p_high_oof[held] = p_held
        r2 = Ridge(alpha=alpha).fit(
            np.hstack([features[train], p_train[:, None]]), targets[train]
        )
        predictions["R2_HIGH_AUX"][held] = r2.predict(
            np.hstack([features[held], p_held[:, None]])
        )

        scores = classifier_scores(labels[held], p_held)
        scores["fold"] = fold_id
        classifier_rows.append(scores)

        for name in names:
            row = tail_scores(targets[held], predictions[name][held])
            row["candidate"] = name
            row["fold"] = fold_id
            fold_rows.append(row)

    print("\n=== 바깥 fold 평균 (5-fold, 괄호는 표준편차) ===")
    summary_rows = []
    for name in names:
        rows = [row for row in fold_rows if row["candidate"] == name]

        def stat(key: str) -> tuple[float | None, float | None]:
            values = [row[key] for row in rows if row[key] is not None]
            if not values:
                return None, None
            return float(np.mean(values)), float(np.std(values))

        entry = {"candidate": name}
        for key in (
            "mae",
            "rmse",
            "bias",
            "lowMae",
            "mediumMae",
            "highMae",
            "highBias",
            "catastrophicRate",
            "p90AbsError",
        ):
            mean, deviation = stat(key)
            entry[key] = mean
            entry[f"{key}Std"] = deviation
        entry["catastrophicCount"] = sum(row["catastrophicCount"] for row in rows)
        entry["highCount"] = sum(row["highCount"] for row in rows)
        summary_rows.append(entry)
        print(
            f"  {name:14} 전체 MAE {entry['mae']:.4f}({entry['maeStd']:.3f})  "
            f"고잔반 MAE {entry['highMae']:.4f}({entry['highMaeStd']:.3f})  "
            f"고잔반 Bias {entry['highBias']:+.4f}  "
            f"치명 {entry['catastrophicRate']:.1%}  "
            f"low {entry['lowMae']:.4f}  med {entry['mediumMae']:.4f}"
        )

    # fold 단위 안정성. 한 fold만 크게 좋아진 것을 개선이라 부르지 않는다.
    print("\n=== R0 대비 고잔반 MAE가 개선된 fold 수 ===")
    stability_rows = []
    base = {row["fold"]: row for row in fold_rows if row["candidate"] == "R0_STANDARD"}
    for name in names[1:]:
        improved = 0
        deltas = []
        for row in fold_rows:
            if row["candidate"] != name or row["highMae"] is None:
                continue
            reference = base[row["fold"]]["highMae"]
            if reference is None:
                continue
            deltas.append(row["highMae"] - reference)
            if row["highMae"] < reference:
                improved += 1
        stability_rows.append(
            {
                "candidate": name,
                "foldsImproved": improved,
                "foldsCompared": len(deltas),
                "meanDelta": float(np.mean(deltas)),
                "worstDelta": float(np.max(deltas)),
            }
        )
        print(
            f"  {name:14} {improved}/{len(deltas)} fold   "
            f"평균 변화 {np.mean(deltas):+.4f}  최악 {np.max(deltas):+.4f}"
        )

    print("\n=== 고잔반 보조 분류기 (바깥 fold 평균) ===")
    for key in ("recall", "precision", "f1", "auroc"):
        values = [row[key] for row in classifier_rows if row[key] is not None]
        print(f"  {key:10} {np.mean(values):.3f}" if values else f"  {key:10} N/A")

    # --- 선택 ---------------------------------------------------------
    by_name = {row["candidate"]: row for row in summary_rows}
    reference_mae = by_name["R0_STANDARD"]["mae"]
    guard = float(policy["overallGuardMargin"])
    passed = [
        row for row in summary_rows if row["mae"] <= reference_mae + guard
    ]
    print(f"\nGate 1 (전체 MAE <= {reference_mae + guard:.4f}) 통과: "
          f"{[row['candidate'] for row in passed]}")

    best_tail = min(row["highMae"] for row in passed)
    tie_margin = float(policy["tailTieMargin"])
    tied = [row for row in passed if row["highMae"] - best_tail <= tie_margin]
    if len(tied) > 1:
        lowest = min(row["catastrophicRate"] for row in tied)
        tied = [row for row in tied if row["catastrophicRate"] <= lowest + 1e-12]
    chosen = min(tied, key=lambda row: names.index(row["candidate"]))
    print(
        f"선택: {chosen['candidate']}  고잔반 MAE {chosen['highMae']:.4f}  "
        f"전체 MAE {chosen['mae']:.4f}  (동률 후보 {len(tied)}개)"
    )

    # --- 기록 ---------------------------------------------------------
    def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
        with (args.out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    fold_fields = [
        "candidate",
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
    ]
    write_csv("outer_fold_metrics.csv", fold_fields, fold_rows)
    write_csv(
        "candidate_summary.csv",
        ["candidate"]
        + [
            key
            for key in summary_rows[0]
            if key != "candidate"
        ],
        summary_rows,
    )
    write_csv(
        "high_waste_metrics.csv",
        ["fold", "positives", "recall", "precision", "f1", "auroc"],
        classifier_rows,
    )
    write_csv(
        "fold_stability.csv",
        ["candidate", "foldsImproved", "foldsCompared", "meanDelta", "worstDelta"],
        stability_rows,
    )

    with (args.out / "development_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sampleId", "group", "fold", "target", "isHighWaste", "pHigh"] + names
        )
        for i in range(len(row_ids)):
            writer.writerow(
                [
                    row_ids[i],
                    groups[i],
                    int(fold_of[i]),
                    float(targets[i]),
                    int(labels[i]),
                    float(p_high_oof[i]),
                ]
                + [float(predictions[name][i]) for name in names]
            )

    (args.out / "selected_config.yaml").write_text(
        "# DEVELOPMENT 중첩 group CV에서 고른 추정기. ANALYSIS를 보기 전에 정했다.\n"
        f"candidate: {chosen['candidate']}\n"
        f"overall_mae: {chosen['mae']:.6f}\n"
        f"high_waste_mae: {chosen['highMae']:.6f}\n"
        f"high_waste_bias: {chosen['highBias']:.6f}\n"
        f"catastrophic_rate: {chosen['catastrophicRate']:.6f}\n"
        f"ridge_alpha: {alpha}\n"
        "selection: gate1 overall MAE guard, gate2 lowest high-waste MAE\n",
        encoding="utf-8",
    )
    (args.out / "metadata.json").write_text(
        json.dumps(
            {
                "stage": "W4",
                "policyHash": policy["sha256OfRecordWithoutHash"],
                "developmentCount": len(row_ids),
                "developmentGroups": len(set(groups)),
                "highWasteCount": int(labels.sum()),
                "outerFolds": outer_folds,
                "innerFolds": inner_folds,
                "outerGroupLeakage": 0,
                "analysisOpened": False,
                "selected": chosen["candidate"],
                "selectionReferenceMae": reference_mae,
                "gate1Threshold": reference_mae + guard,
                "gate1Passed": [row["candidate"] for row in passed],
                "hyperparameterSearch": False,
                "note": (
                    "R2의 학습 표본 p_high는 안쪽 group OOF로 만들었다. "
                    "자기 라벨로 학습한 분류기의 확률을 자기 특징에 붙이지 않는다."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n기록: {args.out / 'candidate_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
