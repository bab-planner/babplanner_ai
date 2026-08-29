"""W5 2단계 — 동결한 구조를 W2 분석 집합에 정확히 한 번 적용한다.

> **이건 테스트가 아니다.** W2 TEST 116건은 W2에서 결과를 이미 본 집합이고
> W3에서 `W2_ANALYSIS`로 재분류했다. 여기 숫자는 POST-HOC FROZEN ANALYSIS다.
> FINAL TEST도, pristine validation도, 일반화 근거도 아니다.

이 스크립트는 아무것도 고르지 않는다. 후보도, 임계값도, 구조도, 예측 정책도
전부 앞 단계 기록에서 읽는다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w5_analysis.py
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
from w4_experiment import classifier_scores, tail_scores  # noqa: E402
from w5_experiment import NAMES, fit_predict_all  # noqa: E402

from waste_estimation.baselines import clip  # noqa: E402
from waste_estimation.structured import prediction_quantiles  # noqa: E402
from waste_estimation.tail import catastrophic_underestimation, is_high_waste  # noqa: E402

W2 = Path("research/waste_estimation/experiments/w2_external")
OUT = Path("research/waste_estimation/experiments/w5_structured")

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
    selected = development["selected"]
    print(f"동결 해시: {policy['sha256OfRecordWithoutHash']}")
    print(f"선택된 구조: {selected}   예측 정책: {development['predictionPolicy']}")
    print("여기서는 아무것도 고르지 않는다.")

    manifest = json.loads((args.w2 / "split_manifest.json").read_text(encoding="utf-8"))
    data = load_features(policy["representation"]["backbone"])
    dev_index, test_index = split_masks(data["row_ids"], manifest)
    dev_features, dev_targets = data["after"][dev_index], data["targets"][dev_index]
    features = data["after"][test_index]
    targets = data["targets"][test_index]
    groups = [data["groups"][i] for i in test_index]
    names = [data["names"][i] for i in test_index]
    row_ids = [data["row_ids"][i] for i in test_index]
    high = is_high_waste(targets)
    print(
        f"\n{SPLIT_LABEL} {len(row_ids)}건 / {len(set(groups))}범주 · "
        f"고잔반 {int(high.sum())}건"
    )

    # DEVELOPMENT 전체로 한 번 맞추고 분석 집합에 한 번 적용한다.
    raw, diagnostics = fit_predict_all(
        dev_features,
        dev_targets,
        features,
        float(policy["ridgeAlpha"]),
        float(policy["classifierAlpha"]),
        tuple(float(v) for v in policy["cumulativeThresholds"]),
        int(policy["candidates"][2]["minExpertSamples"]),
    )
    bounded = {name: clip(raw[name], 0.0, 1.0) for name in NAMES}

    print("\n=== POST-HOC FROZEN ANALYSIS · BOUNDED (선택은 이미 끝났다) ===")
    metric_rows = []
    for name in NAMES:
        for output, values in (("bounded", bounded[name]), ("raw", raw[name])):
            scores = tail_scores(targets, values)
            scores["candidate"] = name
            scores["output"] = output
            scores["split"] = SPLIT_LABEL
            scores["isSelected"] = name == selected
            scores["catastrophicRatePooled"] = scores["catastrophicRate"]
            metric_rows.append(scores)
        row = next(
            r for r in metric_rows if r["candidate"] == name and r["output"] == "bounded"
        )
        print(
            f"  {name:36} MAE {row['mae']:.4f}  RMSE {row['rmse']:.4f}  "
            f"Bias {row['bias']:+.4f}  고잔반 {row['highMae']:.4f}  "
            f"고잔반 Bias {row['highBias']:+.4f}  "
            f"치명 {row['catastrophicCount']}/{row['highCount']} "
            f"({row['catastrophicRate']:.1%})  low {row['lowMae']:.4f}"
        )

    gate = None
    if "gateScore" in diagnostics:
        gate = classifier_scores(high.astype(float), diagnostics["gateScore"])
        print(
            f"\n게이트: recall {gate['recall']:.3f}  precision {gate['precision']:.3f}  "
            f"F1 {gate['f1']:.3f}  AUROC {gate['auroc']:.3f}"
        )
    print(
        f"전문가 학습 표본: 저잔반 {diagnostics['lowExpertCount']} / "
        f"고잔반 {diagnostics['highExpertCount']}  "
        f"단조 위반 보정 {diagnostics['repairedViolations']}건"
    )

    # --- 치명적 오류의 회복과 새 악화 -------------------------------------
    base = catastrophic_underestimation(bounded["S0_STANDARD_RIDGE"], targets)
    chosen_flag = catastrophic_underestimation(bounded[selected], targets)
    recovered = np.flatnonzero(base & ~chosen_flag)
    remaining = np.flatnonzero(base & chosen_flag)
    introduced = np.flatnonzero(~base & chosen_flag)
    print(
        f"\n=== 치명적 과소추정 (S0 -> {selected}) ===\n"
        f"  RECOVERED {len(recovered)}건 · 그대로 {len(remaining)}건 · "
        f"NEW_CATASTROPHIC_REGRESSION {len(introduced)}건"
    )

    recovery_rows = []
    for index in range(len(row_ids)):
        recovery_rows.append(
            {
                "sampleId": row_ids[index],
                "group": groups[index],
                "foodName": names[index],
                "target": float(targets[index]),
                "isHighWaste": int(high[index]),
                "gateScore": (
                    float(diagnostics["gateScore"][index])
                    if "gateScore" in diagnostics
                    else None
                ),
                "S0_bounded": float(bounded["S0_STANDARD_RIDGE"][index]),
                "selected_bounded": float(bounded[selected][index]),
                "S0_absError": abs(
                    float(bounded["S0_STANDARD_RIDGE"][index]) - float(targets[index])
                ),
                "selected_absError": abs(
                    float(bounded[selected][index]) - float(targets[index])
                ),
                "S0_catastrophic": bool(base[index]),
                "selected_catastrophic": bool(chosen_flag[index]),
                "movement": (
                    "RECOVERED"
                    if base[index] and not chosen_flag[index]
                    else "NEW_CATASTROPHIC_REGRESSION"
                    if chosen_flag[index] and not base[index]
                    else "remaining"
                    if base[index]
                    else "none"
                ),
            }
        )

    # --- W2 대표 오류 3건 -------------------------------------------------
    # 이 표본들 때문에 모델을 다시 바꾸지 않는다. 기록만 한다.
    by_id = {row_ids[i]: i for i in range(len(row_ids))}
    print("\n=== W2 대표 오류 3건 (모델을 바꾸지 않는다. 기록만) ===")
    largest_rows = []
    for failure in read_csv(args.w2 / "failure_cases.csv")[:3]:
        sample = failure.get("sampleId")
        if sample not in by_id:
            continue
        index = by_id[sample]
        entry = {
            "sampleId": sample,
            "foodName": names[index],
            "target": float(targets[index]),
            "S0_prediction": float(bounded["S0_STANDARD_RIDGE"][index]),
            "selected_prediction": float(bounded[selected][index]),
            "S0_absError": abs(
                float(bounded["S0_STANDARD_RIDGE"][index]) - float(targets[index])
            ),
            "selected_absError": abs(
                float(bounded[selected][index]) - float(targets[index])
            ),
            "movement": recovery_rows[index]["movement"],
        }
        largest_rows.append(entry)
        print(
            f"  {entry['foodName']:<26} 정답 {entry['target']:.2f}  "
            f"S0 {entry['S0_prediction']:.2f} (오차 {entry['S0_absError']:.2f})  "
            f"-> 선택 {entry['selected_prediction']:.2f} "
            f"(오차 {entry['selected_absError']:.2f})  {entry['movement']}"
        )

    # --- 방향성 보정 -------------------------------------------------------
    # 고잔반에서 평균 수축이 실제로 줄었는지. 평균끼리 비교해야 보인다.
    level = np.where(
        high, "high", np.where(targets <= 0.2, "low", "medium")
    )
    calibration_rows = []
    print("\n=== 수준별 평균: 정답 대 예측 ===")
    for value in ("low", "medium", "high"):
        mask = level == value
        if not mask.any():
            continue
        entry = {
            "level": value,
            "count": int(mask.sum()),
            "meanTarget": float(targets[mask].mean()),
        }
        for name in NAMES:
            entry[f"{name}_meanPrediction"] = float(bounded[name][mask].mean())
        calibration_rows.append(entry)
        print(
            f"  {value:7} n={entry['count']:>3}  정답 {entry['meanTarget']:.3f}  "
            + "  ".join(
                f"{name.split('_')[0]} {entry[f'{name}_meanPrediction']:.3f}"
                for name in NAMES
            )
        )

    distribution_rows = [
        dict({"split": SPLIT_LABEL, "candidate": "GROUND_TRUTH"}, **prediction_quantiles(targets))
    ]
    for name in NAMES:
        distribution_rows.append(
            dict(
                {"split": SPLIT_LABEL, "candidate": name},
                **prediction_quantiles(bounded[name]),
            )
        )
    print("\n=== 예측 분포 (자른 값) ===")
    for row in distribution_rows:
        print(
            f"  {row['candidate']:36} min {row['min']:.3f}  q25 {row['q25']:.3f}  "
            f"med {row['median']:.3f}  q75 {row['q75']:.3f}  max {row['max']:.3f}  "
            f"sd {row['std']:.3f}"
        )

    # --- 기록 ---------------------------------------------------------------
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
            "output",
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
            "catastrophicRatePooled",
            "p90AbsError",
        ],
        metric_rows,
    )
    write_csv(
        "catastrophic_recovery.csv",
        [
            "sampleId",
            "group",
            "foodName",
            "target",
            "isHighWaste",
            "gateScore",
            "S0_bounded",
            "selected_bounded",
            "S0_absError",
            "selected_absError",
            "S0_catastrophic",
            "selected_catastrophic",
            "movement",
        ],
        recovery_rows,
    )
    if largest_rows:
        write_csv("w2_largest_failures.csv", list(largest_rows[0]), largest_rows)
    write_csv(
        "directional_calibration.csv",
        ["level", "count", "meanTarget"]
        + [f"{name}_meanPrediction" for name in NAMES],
        calibration_rows,
    )
    write_csv(
        "analysis_prediction_distribution.csv",
        ["split", "candidate", "min", "q25", "median", "q75", "max", "mean", "std"],
        distribution_rows,
    )

    (args.out / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "stage": "W5",
                "split": SPLIT_LABEL,
                "isFinalTest": False,
                "isGeneralizationTest": False,
                "isPristineValidation": False,
                "resultName": "POST-HOC FROZEN ANALYSIS",
                "note": (
                    "W2에서 이미 결과를 본 집합이다. 선택은 DEVELOPMENT에서 "
                    "끝났고 여기서 바뀌지 않는다."
                ),
                "policyHash": policy["sha256OfRecordWithoutHash"],
                "selected": selected,
                "selectionMadeOn": "DEVELOPMENT group CV",
                "predictionPolicy": "bounded",
                "applications": 1,
                "sampleCount": len(row_ids),
                "groupCount": len(set(groups)),
                "highWasteCount": int(high.sum()),
                "gate": gate,
                "lowExpertCount": diagnostics["lowExpertCount"],
                "highExpertCount": diagnostics["highExpertCount"],
                "monotonicViolationsRepaired": diagnostics["repairedViolations"],
                "binRepresentatives": diagnostics["representatives"],
                "catastrophicMovement": {
                    "recovered": len(recovered),
                    "remaining": len(remaining),
                    "newRegression": len(introduced),
                },
                "freshTest": policy["freshTest"],
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
