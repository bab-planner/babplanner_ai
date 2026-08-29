"""W6 — 동결한 W5 모델을 통제된 연출 파일럿에서 평가한다.

**모델을 LeFood DEVELOPMENT 398건으로만 맞춘다.** 파일럿 라벨은 학습에
들어가지 않는다. 임계값도, 게이트도, 전문가 분할도 파일럿을 보고 바꾸지
않는다 — 그 코드 경로 자체를 두지 않았다.

전역 평균 baseline도 **LeFood DEVELOPMENT 평균**이다. 파일럿 라벨로 평균을
내면 그 순간 평가 누수다.

파일럿이 관문을 통과하지 못하면 **아무 숫자도 내지 않고 멈춘다.**

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w6_evaluate.py ^
      --manifest research\\waste_estimation\\data\\private\\w6_pilot\\manifest.csv ^
      --images research\\waste_estimation\\data\\private\\w6_pilot\\images
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

from waste_estimation.pilot import (  # noqa: E402
    PilotError,
    evaluate_gate,
    load_manifest,
    repeatability,
    summarize_repeatability,
)

W2 = Path("research/waste_estimation/experiments/w2_external")
W5 = Path("research/waste_estimation/experiments/w5_structured")
OUT = Path("research/waste_estimation/experiments/w6_pilot")

BLOCKED = "BLOCKED_PENDING_CONTROLLED_DATA"
SPLIT_LABEL = "CONTROLLED_STAGED_PILOT"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--w2", type=Path, default=W2)
    parser.add_argument("--w5", type=Path, default=W5)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    if not args.manifest.exists():
        print(f"STATUS: {BLOCKED}")
        print(f"매니페스트가 없습니다: {args.manifest}")
        print("수집 절차: docs/waste_controlled_pilot_runbook.md")
        return 0

    try:
        samples, rejections = load_manifest(args.manifest, args.images)
    except PilotError as error:
        print(f"STATUS: PILOT_INSUFFICIENT — {error}")
        return 1

    primary = [sample for sample in samples if sample.repeat_index == 0]
    gate = evaluate_gate(primary, rejections)
    if not gate["passed"]:
        # **여기서 멈춘다.** 부족한 표본으로 낸 숫자는 반드시 인용된다.
        print(f"STATUS: {gate['status']}")
        for name, check in gate["checks"].items():
            if not check["passed"]:
                print(f"  {name}: {check['actual']} / 기준 {check['required']}")
        print("\n표본이 관문을 넘지 못했습니다. 모델을 평가하지 않습니다.")
        return 2

    # 여기서부터는 무거운 의존성이 필요하다.
    import numpy as np

    from w2_experiment import load_features, split_masks
    from w4_experiment import classifier_scores, tail_scores
    from w5_experiment import NAMES, fit_predict_all

    from waste_estimation.baselines import clip
    from waste_estimation.features import FrozenBackbone
    from waste_estimation.structured import prediction_quantiles
    from waste_estimation.tail import catastrophic_underestimation, is_high_waste

    policy = json.loads((args.w5 / "freeze.json").read_text(encoding="utf-8"))
    development = json.loads((args.w5 / "metadata.json").read_text(encoding="utf-8"))
    selected = development["selected"]
    print(f"동결 해시: {policy['sha256OfRecordWithoutHash']}")
    print(f"선택된 구조: {selected}  (여기서 바꾸지 않는다)")

    # --- 학습은 LeFood DEVELOPMENT만 -----------------------------------
    manifest = json.loads((args.w2 / "split_manifest.json").read_text(encoding="utf-8"))
    data = load_features(policy["representation"]["backbone"])
    dev_index, _ = split_masks(data["row_ids"], manifest)
    dev_features, dev_targets = data["after"][dev_index], data["targets"][dev_index]
    print(f"학습 표본: LeFood DEVELOPMENT {len(dev_index)}건 (파일럿 라벨 미사용)")

    # --- 파일럿 이미지를 같은 전처리로 임베딩 ---------------------------
    # W2와 **같은 backbone·같은 전처리**다. 여기서 다시 정하지 않는다 —
    # 전처리가 어긋나면 특징이 망가지고, 그 손상이 domain gap으로 읽힌다.
    backbone = FrozenBackbone(name=policy["representation"]["backbone"]).load()
    paths = [args.images / sample.image_file for sample in primary]
    features = backbone.embed(paths)
    print(f"임베딩: {backbone.name} / {backbone.weights_name} (frozen)")
    targets = np.array([sample.waste_rate for sample in primary])
    groups = [sample.food_group for sample in primary]
    high = is_high_waste(targets)
    print(
        f"\n{SPLIT_LABEL} {len(primary)}건 / {len(set(groups))}범주 · "
        f"고잔반 {int(high.sum())}건"
    )

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
    # B0 — **LeFood DEVELOPMENT 평균**이다. 파일럿 평균을 쓰면 누수다.
    bounded["B0_GLOBAL_MEAN"] = np.full(len(targets), float(dev_targets.mean()))

    print("\n=== 통제 파일럿 결과 (bounded) ===")
    metric_rows = []
    for name in ("B0_GLOBAL_MEAN",) + NAMES:
        scores = tail_scores(targets, bounded[name])
        scores["candidate"] = name
        scores["split"] = SPLIT_LABEL
        scores["isSelected"] = name == selected
        metric_rows.append(scores)
        catastrophic = (
            f"{scores['catastrophicCount']}/{scores['highCount']} "
            f"({scores['catastrophicRate']:.1%})"
            if scores["catastrophicRate"] is not None
            else "N/A"
        )
        print(
            f"  {name:36} MAE {scores['mae']:.4f}  RMSE {scores['rmse']:.4f}  "
            f"Bias {scores['bias']:+.4f}  고잔반 {scores['highMae']:.4f}  "
            f"치명 {catastrophic}"
        )

    gate_scores = None
    if "gateScore" in diagnostics:
        gate_scores = classifier_scores(high.astype(float), diagnostics["gateScore"])
        print(
            f"\n게이트: recall {gate_scores['recall']:.3f}  "
            f"precision {gate_scores['precision']:.3f}  F1 {gate_scores['f1']:.3f}"
        )

    # --- 범주별 -----------------------------------------------------------
    category_rows = []
    for group in sorted(set(groups)):
        mask = np.array([index for index, g in enumerate(groups) if g == group])
        category_rows.append(
            {
                "foodGroup": group,
                "count": len(mask),
                "meanTarget": float(targets[mask].mean()),
                "S0_mae": float(
                    np.abs(bounded["S0_STANDARD_RIDGE"][mask] - targets[mask]).mean()
                ),
                "selected_mae": float(
                    np.abs(bounded[selected][mask] - targets[mask]).mean()
                ),
                "selected_bias": float(
                    (bounded[selected][mask] - targets[mask]).mean()
                ),
            }
        )

    # --- domain gap (서술용) ------------------------------------------------
    analysis = {
        row["candidate"]: row
        for row in csv.DictReader(
            (args.w5 / "analysis_metrics.csv").read_text(encoding="utf-8").splitlines()
        )
        if row["output"] == "bounded"
    }
    local = {row["candidate"]: row for row in metric_rows}
    gap_rows = []
    for name in NAMES:
        gap_rows.append(
            {
                "candidate": name,
                "lefoodAnalysisMae": float(analysis[name]["mae"]),
                "localPilotMae": float(local[name]["mae"]),
                "deltaMae": float(local[name]["mae"]) - float(analysis[name]["mae"]),
                "lefoodAnalysisHighMae": float(analysis[name]["highMae"]),
                "localPilotHighMae": float(local[name]["highMae"]),
            }
        )
    print("\n=== domain gap (서술용. 같은 모집단이 아니므로 동등 비교가 아니다) ===")
    for row in gap_rows:
        print(
            f"  {row['candidate']:36} LeFood 분석 {row['lefoodAnalysisMae']:.4f}  "
            f"통제 파일럿 {row['localPilotMae']:.4f}  Δ {row['deltaMae']:+.4f}"
        )

    # --- 측정 오차 바닥 ------------------------------------------------------
    repeats: dict[str, list[float]] = {}
    served: dict[str, float] = {}
    for sample in samples:
        repeats.setdefault(sample.sample_id, []).append(sample.leftover_edible_mass_g)
        served.setdefault(sample.sample_id, sample.served_edible_mass_g)
    entries = [
        repeatability(values, served[key])
        for key, values in repeats.items()
        if len(values) >= 2
    ]
    floor = summarize_repeatability(entries) if entries else None
    if floor:
        print(
            f"\n측정 오차 바닥: 잔반률 표준편차 평균 {floor['meanStdWasteRate']:.4f} "
            f"(최대 {floor['maxStdWasteRate']:.4f})"
        )
        print(f"  {floor['interpretation']}")

    distribution_rows = [
        dict({"candidate": "GROUND_TRUTH"}, **prediction_quantiles(targets))
    ]
    for name in ("B0_GLOBAL_MEAN",) + NAMES:
        distribution_rows.append(
            dict({"candidate": name}, **prediction_quantiles(bounded[name]))
        )

    failure_rows = []
    base = catastrophic_underestimation(bounded["S0_STANDARD_RIDGE"], targets)
    chosen_flag = catastrophic_underestimation(bounded[selected], targets)
    for index, sample in enumerate(primary):
        failure_rows.append(
            {
                "sampleId": sample.sample_id,
                "foodGroup": sample.food_group,
                "captureSessionId": sample.capture_session_id,
                "target": float(targets[index]),
                "isHighWaste": int(high[index]),
                "gateScore": (
                    float(diagnostics["gateScore"][index])
                    if "gateScore" in diagnostics
                    else None
                ),
                "S0_bounded": float(bounded["S0_STANDARD_RIDGE"][index]),
                "selected_bounded": float(bounded[selected][index]),
                "selected_absError": abs(
                    float(bounded[selected][index]) - float(targets[index])
                ),
                "S0_catastrophic": bool(base[index]),
                "selected_catastrophic": bool(chosen_flag[index]),
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)

    def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
        with (args.out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    write_csv(
        "pilot_metrics.csv",
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
        "domain_gap.csv",
        [
            "candidate",
            "lefoodAnalysisMae",
            "localPilotMae",
            "deltaMae",
            "lefoodAnalysisHighMae",
            "localPilotHighMae",
        ],
        gap_rows,
    )
    write_csv(
        "category_metrics.csv",
        ["foodGroup", "count", "meanTarget", "S0_mae", "selected_mae", "selected_bias"],
        category_rows,
    )
    write_csv(
        "pilot_prediction_distribution.csv",
        ["candidate", "min", "q25", "median", "q75", "max", "mean", "std"],
        distribution_rows,
    )
    write_csv(
        "pilot_failure_cases.csv",
        [
            "sampleId",
            "foodGroup",
            "captureSessionId",
            "target",
            "isHighWaste",
            "gateScore",
            "S0_bounded",
            "selected_bounded",
            "selected_absError",
            "S0_catastrophic",
            "selected_catastrophic",
        ],
        failure_rows,
    )

    (args.out / "metadata.json").write_text(
        json.dumps(
            {
                "stage": "W6",
                "split": SPLIT_LABEL,
                "sampleNature": "staged leftover conditions",
                "isNaturalMealData": False,
                "isFinalTest": False,
                "note": (
                    "사람이 먹고 남긴 표본이 아니다. 촬영 조건 전이만 잰다. "
                    "실제 이용자의 식습관에 대해서는 아무것도 말하지 못한다."
                ),
                "w5PolicyHash": policy["sha256OfRecordWithoutHash"],
                "w5SelectedConfigHash": digest(args.w5 / "selected_config.yaml"),
                "lefoodSplitManifestHash": digest(args.w2 / "split_manifest.json"),
                "pilotManifestHash": digest(args.manifest),
                "trainingSource": "LeFood DEVELOPMENT only",
                "pilotUsedForFitting": False,
                "pilotUsedForSelection": False,
                "pilotUsedForThresholds": False,
                "globalMeanSource": "LeFood DEVELOPMENT mean",
                "selected": selected,
                "sampleCount": len(primary),
                "groupCount": len(set(groups)),
                "highWasteCount": int(high.sum()),
                "gate": gate_scores,
                "measurementFloor": floor,
                "acceptanceGate": gate["checks"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n기록: {args.out / 'pilot_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
