"""W3 1단계 — DEVELOPMENT의 out-of-fold 예측과 신뢰 점수.

**왜 OOF인가.** 자기 정답을 본 모델이 자기 예측을 만들면 오차가 실제보다
작게 나온다. 신뢰 신호가 그 오차와 얼마나 붙어 있는지를 재는 것이 목적인데,
오차 자체가 낙관적이면 관계도 낙관적이다.

그래서 표본마다 **그 표본의 음식 범주가 학습 fold에 없는 모델**이 예측을
만든다. 범주 단위로 나누므로 "처음 보는 음식"에 대한 예측이 된다.

**추정기를 다시 고르지 않는다.** W2가 DEVELOPMENT에서 고른 after-only 설정
(resnet18 / alpha)을 `development_selection.json`에서 읽어 그대로 쓴다.
W3의 주제는 추정기가 아니라 그 위에 얹는 신뢰 신호다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w3_oof.py
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
from w2_prepare import CV_FOLDS, LEVEL_EDGES, LEVEL_NAMES, SPLIT_SEED  # noqa: E402

from waste_estimation.baselines import Ridge  # noqa: E402
from waste_estimation.metrics import to_class_index  # noqa: E402
from waste_estimation.reliability import (  # noqa: E402
    bootstrap_disagreement,
    combine,
    knn_novelty,
    prediction_extremeness,
)

W2 = Path("research/waste_estimation/experiments/w2_external")
OUT = Path("research/waste_estimation/experiments/w3_selective")

#: C0의 이웃 수. 소수 후보만 본다. 대규모 sweep을 하지 않는다.
KNN_CANDIDATES = (3, 5, 10)

#: C1의 부트스트랩 모델 수. CV fold 수와 맞춘다.
ENSEMBLE_MEMBERS = 5
ENSEMBLE_SEED = 20260829


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w2", type=Path, default=W2)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    manifest = json.loads((args.w2 / "split_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (args.w2 / "development_selection.json").read_text(encoding="utf-8")
    )["after"]
    backbone = selection["backbone"]
    alpha = float(selection["alpha"])
    print(f"W2가 고른 추정기: {backbone}, alpha={alpha}  (그대로 쓴다)")

    data = load_features(backbone)
    dev_index, _ = split_masks(data["row_ids"], manifest)
    features = data["after"][dev_index]
    targets = data["targets"][dev_index]
    groups = [data["groups"][i] for i in dev_index]
    names = [data["names"][i] for i in dev_index]
    row_ids = [data["row_ids"][i] for i in dev_index]
    print(f"DEVELOPMENT {len(row_ids)}건 / {len(set(groups))}범주")

    buckets = group_folds(groups, CV_FOLDS, SPLIT_SEED)
    predictions = np.zeros(len(targets), dtype=float)
    novelty = {k: np.zeros(len(targets), dtype=float) for k in KNN_CANDIDATES}
    disagreement = np.zeros(len(targets), dtype=float)
    fold_of = np.zeros(len(targets), dtype=int)

    for fold_id, fold in enumerate(buckets):
        if not fold:
            continue
        held = np.array(fold)
        train_mask = np.ones(len(targets), dtype=bool)
        train_mask[held] = False
        train_index = np.flatnonzero(train_mask)
        fold_of[held] = fold_id

        model = Ridge(alpha=alpha).fit(features[train_index], targets[train_index])
        predictions[held] = model.predict(features[held])

        # C0. 참조는 **학습 fold의 임베딩만**이다. 자기 자신도, 같은 범주의
        # 다른 표본도 참조에 들어가지 않는다(범주 단위로 나눴으므로).
        for k in KNN_CANDIDATES:
            novelty[k][held] = knn_novelty(
                features[held], features[train_index], k=k
            )

        # C1. 같은 frozen 임베딩 위에서 학습 표본만 부트스트랩한다.
        def fit_predict(sample_index, _held=held):
            fitted = Ridge(alpha=alpha).fit(features[sample_index], targets[sample_index])
            return fitted.predict(features[_held])

        disagreement[held] = bootstrap_disagreement(
            fit_predict,
            train_index,
            features[held],
            members=ENSEMBLE_MEMBERS,
            seed=ENSEMBLE_SEED + fold_id,
        )

    # 누수 확인. fold마다 학습/평가 범주가 겹치면 여기서 멈춘다.
    for fold_id, fold in enumerate(buckets):
        if not fold:
            continue
        held_groups = {groups[i] for i in fold}
        train_groups = {groups[i] for i in range(len(groups)) if i not in set(fold)}
        overlap = held_groups & train_groups
        if overlap:
            raise SystemExit(f"fold {fold_id} 범주 누수: {sorted(overlap)}")
    print("fold 간 범주 누수: 없음")

    # C2. 학습 평균은 DEVELOPMENT 전체 평균을 쓴다. 추론 시점에 상수다.
    development_mean = float(targets.mean())
    extremeness = prediction_extremeness(predictions, development_mean)

    errors = np.abs(targets - predictions)
    print(f"OOF MAE {errors.mean():.4f}   (참고: W2 TEST after-only 0.1526)")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "oof_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sampleId",
                "group",
                "foodName",
                "fold",
                "target",
                "prediction",
                "absError",
                "wasteLevel",
            ]
            + [f"C0_knn{k}" for k in KNN_CANDIDATES]
            + ["C1_disagreement", "C2_extremeness"]
        )
        for i in range(len(row_ids)):
            writer.writerow(
                [
                    row_ids[i],
                    groups[i],
                    names[i],
                    int(fold_of[i]),
                    float(targets[i]),
                    float(predictions[i]),
                    float(errors[i]),
                    LEVEL_NAMES[to_class_index(float(targets[i]), LEVEL_EDGES)],
                ]
                + [float(novelty[k][i]) for k in KNN_CANDIDATES]
                + [float(disagreement[i]), float(extremeness[i])]
            )

    (args.out / "metadata.json").write_text(
        json.dumps(
            {
                "stage": "W3",
                "developmentCount": len(row_ids),
                "developmentGroups": len(set(groups)),
                "folds": CV_FOLDS,
                "foldSeed": SPLIT_SEED,
                "baseEstimator": {
                    "backbone": backbone,
                    "alpha": alpha,
                    "source": "W2 development_selection.json (after-only)",
                    "retuned": False,
                },
                "oofMae": float(errors.mean()),
                "developmentMean": development_mean,
                "knnCandidates": list(KNN_CANDIDATES),
                "ensembleMembers": ENSEMBLE_MEMBERS,
                "ensembleSeed": ENSEMBLE_SEED,
                "groupLeakage": 0,
                "note": (
                    "신뢰 점수는 전부 추론 시점 계산이다. 정답이나 잔차가 "
                    "입력으로 들어가지 않는다."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n기록: {args.out / 'oof_predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
