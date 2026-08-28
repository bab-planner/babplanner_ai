"""W2 실험 — after-only 대 before+after.

두 단계로 나뉜다.

    DEVELOPMENT   그룹 교차검증으로 backbone·입력·alpha를 고른다
    TEST          고른 것 하나로 **한 번만** 채점한다

`--stage development` 로 먼저 돌리고, 선택이 끝난 뒤에 `--stage test` 를
돌린다. 두 단계를 한 번에 돌리지 않는 이유는 사람이 중간에 TEST 결과를 보고
후보를 바꾸는 일을 구조적으로 막기 위해서다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w2_experiment.py --stage development
    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w2_experiment.py --stage test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SERVICE = Path(__file__).resolve().parents[3] / "src"
if str(_SERVICE) not in sys.path:
    sys.path.insert(0, str(_SERVICE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from w2_prepare import (  # noqa: E402
    ALPHA_GRID,
    CV_FOLDS,
    LEVEL_EDGES,
    LEVEL_NAMES,
    ORDINAL_EDGES,
    SPLIT_SEED,
)

from waste_estimation.baselines import GlobalMean, Ridge, clip  # noqa: E402
from waste_estimation.features import paired_features  # noqa: E402
from waste_estimation.metrics import (  # noqa: E402
    class_index_mae,
    macro_f1,
    quadratic_weighted_kappa,
    to_class_index,
    unbounded_bias,
    unbounded_mae,
    unbounded_median_ae,
    unbounded_rmse,
)

CACHE = Path("research/waste_estimation/data/features")
OUT = Path("research/waste_estimation/experiments/w2_external")

#: 비교하는 입력 갈래. after만 운영 API와 같은 입력이다.
ARMS = ("after", "concat", "concat_diff")
BACKBONES = ("mobilenet_v3_small", "resnet18")


@dataclass(frozen=True)
class Scores:
    mae: float
    rmse: float
    median_ae: float
    bias: float
    count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "medianAe": self.median_ae,
            "bias": self.bias,
            "count": self.count,
        }


def score(actual, predicted) -> Scores:
    truth = [float(v) for v in actual]
    guess = [float(v) for v in predicted]
    # 예측은 0~1을 벗어날 수 있다. 지표 전에 자르지 않는다.
    return Scores(
        mae=unbounded_mae(truth, guess),
        rmse=unbounded_rmse(truth, guess),
        median_ae=unbounded_median_ae(truth, guess),
        bias=unbounded_bias(truth, guess),
        count=len(truth),
    )


def load_features(name: str):
    import numpy as np

    data = np.load(CACHE / f"{name}.npz", allow_pickle=False)
    return {
        "row_ids": [str(v) for v in data["row_ids"]],
        "groups": [str(v) for v in data["groups"]],
        "names": [str(v) for v in data["names"]],
        "targets": data["targets"],
        "before": data["before"],
        "after": data["after"],
    }


def split_masks(row_ids: list[str], manifest: dict) -> tuple[list[int], list[int]]:
    development = set(manifest["development"]["sampleIds"])
    test = set(manifest["test"]["sampleIds"])
    dev_index = [i for i, rid in enumerate(row_ids) if rid in development]
    test_index = [i for i, rid in enumerate(row_ids) if rid in test]
    if len(dev_index) + len(test_index) != len(row_ids):
        raise SystemExit("분할 매니페스트가 특징 캐시와 맞지 않습니다.")
    return dev_index, test_index


def group_folds(groups: list[str], folds: int, seed: int) -> list[list[int]]:
    """그룹 단위 폴드. 같은 음식 범주가 두 폴드에 걸치지 않는다."""
    import random

    unique = sorted(set(groups))
    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    assignment = {group: index % folds for index, group in enumerate(shuffled)}
    buckets: list[list[int]] = [[] for _ in range(folds)]
    for index, group in enumerate(groups):
        buckets[assignment[group]].append(index)
    return buckets


def cross_validate(features, targets, groups, alpha: float) -> Scores:
    """DEVELOPMENT 안에서만 도는 그룹 교차검증."""
    import numpy as np

    buckets = group_folds(groups, CV_FOLDS, SPLIT_SEED)
    predictions = np.zeros(len(targets), dtype=float)
    for fold in buckets:
        if not fold:
            continue
        mask = np.ones(len(targets), dtype=bool)
        mask[fold] = False
        model = Ridge(alpha=alpha).fit(features[mask], targets[mask])
        predictions[fold] = model.predict(features[fold])
    return score(targets, predictions)


def development(out: Path) -> int:
    import numpy as np

    manifest = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    rows: list[dict] = []

    simple = load_features("simple")
    dev_index, _ = split_masks(simple["row_ids"], manifest)
    dev_targets = simple["targets"][dev_index]
    dev_groups = [simple["groups"][i] for i in dev_index]

    # B0. 전역 평균. 교차검증으로 재야 공정하다.
    buckets = group_folds(dev_groups, CV_FOLDS, SPLIT_SEED)
    mean_predictions = np.zeros(len(dev_targets), dtype=float)
    for fold in buckets:
        if not fold:
            continue
        mask = np.ones(len(dev_targets), dtype=bool)
        mask[fold] = False
        model = GlobalMean().fit(None, dev_targets[mask])
        mean_predictions[fold] = model.predict(range(len(fold)))
    b0 = score(dev_targets, mean_predictions)
    rows.append({"model": "B0_global_mean", "backbone": "-", "arm": "-", "alpha": "-", **b0.as_dict()})
    print(f"B0 global mean        MAE {b0.mae:.4f}  bias {b0.bias:+.4f}")

    # B1. 단순 이미지 통계.
    for arm in ARMS:
        features = paired_features(simple["before"], simple["after"], arm)[dev_index]
        for alpha in ALPHA_GRID:
            result = cross_validate(features, dev_targets, dev_groups, alpha)
            rows.append(
                {
                    "model": "B1_simple",
                    "backbone": "-",
                    "arm": arm,
                    "alpha": alpha,
                    **result.as_dict(),
                }
            )
        best = min(
            (r for r in rows if r["model"] == "B1_simple" and r["arm"] == arm),
            key=lambda r: r["mae"],
        )
        print(f"B1 simple {arm:12} MAE {best['mae']:.4f}  (alpha {best['alpha']})")

    # B2. frozen backbone.
    for backbone in BACKBONES:
        data = load_features(backbone)
        for arm in ARMS:
            features = paired_features(data["before"], data["after"], arm)[dev_index]
            for alpha in ALPHA_GRID:
                result = cross_validate(features, dev_targets, dev_groups, alpha)
                rows.append(
                    {
                        "model": "B2_frozen",
                        "backbone": backbone,
                        "arm": arm,
                        "alpha": alpha,
                        **result.as_dict(),
                    }
                )
            best = min(
                (
                    r
                    for r in rows
                    if r["model"] == "B2_frozen"
                    and r["backbone"] == backbone
                    and r["arm"] == arm
                ),
                key=lambda r: r["mae"],
            )
            print(
                f"B2 {backbone:20} {arm:12} MAE {best['mae']:.4f}  "
                f"(alpha {best['alpha']})"
            )

    with (out / "development_cv.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "backbone", "arm", "alpha", "mae", "rmse", "medianAe", "bias", "count"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # 동결한 선택 규칙을 적용한다. after-only와 짝 갈래를 따로 고른다.
    selection = {}
    for arm in ARMS:
        candidates = [r for r in rows if r["model"] == "B2_frozen" and r["arm"] == arm]
        best_mae = min(r["mae"] for r in candidates)
        tied = [r for r in candidates if r["mae"] - best_mae <= 0.005]
        chosen = min(tied, key=lambda r: (abs(r["bias"]), r["backbone"], r["alpha"]))
        selection[arm] = {
            "backbone": chosen["backbone"],
            "alpha": chosen["alpha"],
            "cvMae": chosen["mae"],
            "cvBias": chosen["bias"],
            "tiedCandidates": len(tied),
        }
        print(
            f"\n선택 [{arm}] {chosen['backbone']} alpha={chosen['alpha']} "
            f"CV MAE {chosen['mae']:.4f} bias {chosen['bias']:+.4f}"
        )

    (out / "selected_config.yaml").write_text(
        "# DEVELOPMENT 그룹 교차검증으로 고른 설정. TEST를 보기 전에 정했다.\n"
        + "\n".join(
            f"{arm}:\n"
            f"  backbone: {value['backbone']}\n"
            f"  alpha: {value['alpha']}\n"
            f"  cv_mae: {value['cvMae']:.6f}\n"
            f"  cv_bias: {value['cvBias']:+.6f}\n"
            for arm, value in selection.items()
        ),
        encoding="utf-8",
    )
    (out / "development_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n기록: {out / 'development_cv.csv'}, {out / 'selected_config.yaml'}")
    return 0


def test(out: Path) -> int:
    import numpy as np

    manifest = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads((out / "development_selection.json").read_text(encoding="utf-8"))

    simple = load_features("simple")
    dev_index, test_index = split_masks(simple["row_ids"], manifest)
    dev_targets = simple["targets"][dev_index]
    test_targets = simple["targets"][test_index]
    test_names = [simple["names"][i] for i in test_index]
    test_groups = [simple["groups"][i] for i in test_index]
    test_ids = [simple["row_ids"][i] for i in test_index]

    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}

    # B0
    b0_model = GlobalMean().fit(None, dev_targets)
    b0_pred = b0_model.predict(test_targets)
    predictions["B0_global_mean"] = b0_pred
    rows.append(
        {"model": "B0_global_mean", "backbone": "-", "arm": "-", "output": "raw", **score(test_targets, b0_pred).as_dict()}
    )

    # B1 (after-only만. 단순 통계는 참고용 바닥이다)
    b1_features = paired_features(simple["before"], simple["after"], "after")
    b1_model = Ridge(alpha=ALPHA_GRID[0]).fit(b1_features[dev_index], dev_targets)
    b1_pred = b1_model.predict(b1_features[test_index])
    predictions["B1_simple_after"] = b1_pred
    rows.append(
        {"model": "B1_simple", "backbone": "-", "arm": "after", "output": "raw", **score(test_targets, b1_pred).as_dict()}
    )
    rows.append(
        {"model": "B1_simple", "backbone": "-", "arm": "after", "output": "clipped", **score(test_targets, clip(b1_pred)).as_dict()}
    )

    # B2
    for arm, config in selection.items():
        data = load_features(config["backbone"])
        features = paired_features(data["before"], data["after"], arm)
        model = Ridge(alpha=float(config["alpha"])).fit(features[dev_index], dev_targets)
        raw = model.predict(features[test_index])
        predictions[f"B2_{arm}"] = raw
        rows.append(
            {"model": "B2_frozen", "backbone": config["backbone"], "arm": arm, "output": "raw", **score(test_targets, raw).as_dict()}
        )
        rows.append(
            {"model": "B2_frozen", "backbone": config["backbone"], "arm": arm, "output": "clipped", **score(test_targets, clip(raw)).as_dict()}
        )

    with (out / "regression_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "backbone", "arm", "output", "mae", "rmse", "medianAe", "bias", "count"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("=== TEST (한 번만) ===")
    for row in rows:
        print(
            f"  {row['model']:16} {row['backbone']:20} {row['arm']:12} "
            f"{row['output']:8} MAE {row['mae']:.4f}  bias {row['bias']:+.4f}"
        )

    # --- 정보 격차 ---------------------------------------------------
    gap_rows = []
    after_mae = next(r["mae"] for r in rows if r["model"] == "B2_frozen" and r["arm"] == "after" and r["output"] == "raw")
    for arm in ("concat", "concat_diff"):
        pair_mae = next(r["mae"] for r in rows if r["model"] == "B2_frozen" and r["arm"] == arm and r["output"] == "raw")
        gap_rows.append(
            {
                "scope": "TEST_all",
                "pairArm": arm,
                "afterOnlyMae": after_mae,
                "pairMae": pair_mae,
                "deltaMae": after_mae - pair_mae,
            }
        )

    # 그룹(음식 범주)별로도 본다. 한 분할의 우연인지 보려면 필요하다.
    for arm in ("concat", "concat_diff"):
        for group in sorted(set(test_groups)):
            mask = [i for i, g in enumerate(test_groups) if g == group]
            if len(mask) < 3:
                continue
            a = unbounded_mae(
                [float(test_targets[i]) for i in mask],
                [float(predictions["B2_after"][i]) for i in mask],
            )
            p = unbounded_mae(
                [float(test_targets[i]) for i in mask],
                [float(predictions[f"B2_{arm}"][i]) for i in mask],
            )
            gap_rows.append(
                {
                    "scope": group,
                    "pairArm": arm,
                    "afterOnlyMae": a,
                    "pairMae": p,
                    "deltaMae": a - p,
                }
            )

    with (out / "information_gap.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "pairArm", "afterOnlyMae", "pairMae", "deltaMae"])
        writer.writeheader()
        writer.writerows(gap_rows)

    # --- 통제된 정보 격차 -------------------------------------------
    #
    # 위의 표는 갈래마다 DEVELOPMENT에서 고른 backbone이 달라서 정보량과
    # backbone이 섞여 있다. 정보 격차를 재려면 **같은 backbone·같은 전처리**
    # 위에서 입력만 바꿔야 한다. alpha는 각 (backbone, arm)에 대해
    # DEVELOPMENT에서 고른 값을 그대로 쓴다 — TEST를 보고 고르지 않는다.
    dev_cv = {}
    with (out / "development_cv.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != "B2_frozen":
                continue
            key = (row["backbone"], row["arm"])
            mae = float(row["mae"])
            if key not in dev_cv or mae < dev_cv[key][1]:
                dev_cv[key] = (float(row["alpha"]), mae)

    matched_rows = []
    matched_predictions: dict[tuple[str, str], "np.ndarray"] = {}
    for backbone in BACKBONES:
        data = load_features(backbone)
        for arm in ARMS:
            alpha, cv_mae = dev_cv[(backbone, arm)]
            features = paired_features(data["before"], data["after"], arm)
            model = Ridge(alpha=alpha).fit(features[dev_index], dev_targets)
            raw = model.predict(features[test_index])
            matched_predictions[(backbone, arm)] = raw
            matched_rows.append(
                {
                    "backbone": backbone,
                    "arm": arm,
                    "alpha": alpha,
                    "devCvMae": cv_mae,
                    "testMae": unbounded_mae(
                        [float(v) for v in test_targets], [float(v) for v in raw]
                    ),
                }
            )

    gap_matched = []
    for backbone in BACKBONES:
        after_row = next(
            r for r in matched_rows if r["backbone"] == backbone and r["arm"] == "after"
        )
        for arm in ("concat", "concat_diff"):
            pair_row = next(
                r for r in matched_rows if r["backbone"] == backbone and r["arm"] == arm
            )
            gap_matched.append(
                {
                    "backbone": backbone,
                    "pairArm": arm,
                    "devDeltaMae": after_row["devCvMae"] - pair_row["devCvMae"],
                    "testDeltaMae": after_row["testMae"] - pair_row["testMae"],
                    "afterTestMae": after_row["testMae"],
                    "pairTestMae": pair_row["testMae"],
                }
            )

    with (out / "information_gap_matched.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["backbone", "pairArm", "devDeltaMae", "testDeltaMae", "afterTestMae", "pairTestMae"],
        )
        writer.writeheader()
        writer.writerows(gap_matched)

    print("")
    print("=== 같은 backbone 위에서의 정보 격차 (양수면 짝이 낫다) ===")
    for row in gap_matched:
        print(
            f"  {row['backbone']:20} {row['pairArm']:12} "
            f"DEV dMAE {row['devDeltaMae']:+.4f}   TEST dMAE {row['testDeltaMae']:+.4f}"
        )

    print("\n=== 정보 격차 (dMAE = after-only - pair) ===")
    for row in gap_rows:
        if row["scope"] == "TEST_all":
            print(f"  {row['pairArm']:12} dMAE {row['deltaMae']:+.4f}")
    per_group = [r for r in gap_rows if r["scope"] != "TEST_all" and r["pairArm"] == "concat_diff"]
    improved = sum(1 for r in per_group if r["deltaMae"] > 0)
    print(f"  범주별(concat_diff): {improved}/{len(per_group)} 범주에서 개선")

    # --- 순서형 ------------------------------------------------------
    ordinal_rows = []
    classes = len(ORDINAL_EDGES) + 1
    true_classes = [to_class_index(float(v), ORDINAL_EDGES) for v in test_targets]

    for arm, config in selection.items():
        data = load_features(config["backbone"])
        features = paired_features(data["before"], data["after"], arm)
        # T2. 같은 특징·같은 회귀기, 목표만 구간 번호로 바꾼다.
        dev_classes = np.array(
            [to_class_index(float(v), ORDINAL_EDGES) for v in dev_targets], dtype=float
        )
        model = Ridge(alpha=float(config["alpha"])).fit(features[dev_index], dev_classes)
        raw = model.predict(features[test_index])
        predicted_classes = [int(min(max(round(float(v)), 0), classes - 1)) for v in raw]
        ordinal_rows.append(
            {
                "source": "T2_ordinal_target",
                "arm": arm,
                "classIndexMae": class_index_mae(true_classes, predicted_classes),
                "macroF1": macro_f1(true_classes, predicted_classes, classes),
                "qwk": quadratic_weighted_kappa(true_classes, predicted_classes, classes),
            }
        )
        # T3의 연속 예측을 같은 구간으로 옮겨 같은 자로 비교한다.
        binned = [to_class_index(float(v), ORDINAL_EDGES) for v in clip(predictions[f"B2_{arm}"])]
        ordinal_rows.append(
            {
                "source": "T3_binned",
                "arm": arm,
                "classIndexMae": class_index_mae(true_classes, binned),
                "macroF1": macro_f1(true_classes, binned, classes),
                "qwk": quadratic_weighted_kappa(true_classes, binned, classes),
            }
        )

    with (out / "ordinal_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "arm", "classIndexMae", "macroF1", "qwk"])
        writer.writeheader()
        writer.writerows(ordinal_rows)

    print("\n=== 순서형 ===")
    for row in ordinal_rows:
        qwk = row["qwk"]
        print(
            f"  {row['source']:18} {row['arm']:12} class MAE {row['classIndexMae']:.3f}  "
            f"macroF1 {row['macroF1']:.3f}  QWK {qwk:.3f}" if qwk is not None else
            f"  {row['source']:18} {row['arm']:12} class MAE {row['classIndexMae']:.3f}  macroF1 {row['macroF1']:.3f}  QWK N/A"
        )

    # --- 잔반 수준별 오차 ---------------------------------------------
    level_rows = []
    for model_name in ("B0_global_mean", "B2_after", "B2_concat_diff"):
        if model_name not in predictions:
            continue
        for index, name in enumerate(LEVEL_NAMES):
            mask = [
                i
                for i, value in enumerate(test_targets)
                if to_class_index(float(value), LEVEL_EDGES) == index
            ]
            if not mask:
                continue
            level_rows.append(
                {
                    "model": model_name,
                    "level": name,
                    "count": len(mask),
                    "mae": unbounded_mae(
                        [float(test_targets[i]) for i in mask],
                        [float(predictions[model_name][i]) for i in mask],
                    ),
                    "meanTarget": float(np.mean([test_targets[i] for i in mask])),
                    "meanPrediction": float(np.mean([predictions[model_name][i] for i in mask])),
                }
            )

    with (out / "error_by_level.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "level", "count", "mae", "meanTarget", "meanPrediction"])
        writer.writeheader()
        writer.writerows(level_rows)

    print("\n=== 잔반 수준별 오차 ===")
    for row in level_rows:
        print(
            f"  {row['model']:16} {row['level']:8} n={row['count']:3}  "
            f"MAE {row['mae']:.3f}  정답평균 {row['meanTarget']:.3f}  예측평균 {row['meanPrediction']:.3f}"
        )

    # --- 실패 사례 (이미지 없이 기록) ----------------------------------
    errors = [
        {
            "sampleId": test_ids[i],
            "foodName": test_names[i],
            "group": test_groups[i],
            "target": float(test_targets[i]),
            "prediction": float(predictions["B2_after"][i]),
            "absError": abs(float(test_targets[i]) - float(predictions["B2_after"][i])),
        }
        for i in range(len(test_ids))
    ]
    errors.sort(key=lambda row: -row["absError"])
    with (out / "failure_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sampleId", "foodName", "group", "target", "prediction", "absError"])
        writer.writeheader()
        writer.writerows(errors[:20])

    print("\n=== 가장 큰 오차 5건 (after-only) ===")
    for row in errors[:5]:
        print(
            f"  {row['sampleId']:>4} {row['foodName'][:22]:22} 정답 {row['target']:.3f} "
            f"예측 {row['prediction']:+.3f} 오차 {row['absError']:.3f}"
        )

    # --- 예측 기록 (W3 selective prediction 준비) -----------------------
    with (out / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sampleId", "group", "foodName", "target", *sorted(predictions)])
        for i in range(len(test_ids)):
            writer.writerow(
                [test_ids[i], test_groups[i], test_names[i], float(test_targets[i])]
                + [float(predictions[key][i]) for key in sorted(predictions)]
            )

    print(f"\n기록: {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["development", "test"], required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    return development(args.out) if args.stage == "development" else test(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
