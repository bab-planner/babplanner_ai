"""W4 2단계 — 신뢰 임계값이 옮겨지는가.

W3이 남긴 결함이 하나 있었다.

    보정: OOF 참조 ~318건으로 잡은 절대 kNN 거리 임계값
    적용: 참조 398건
    결과: 목표 50% coverage가 62.9%로 실현

원인은 방법 안에 있다. **kNN 거리는 참조가 커지면 저절로 짧아진다.** 이웃이
많아지니 가장 가까운 셋도 더 가까워진다. 절대 거리에 임계값을 걸어 두면
참조가 바뀔 때마다 실제 거절 비율이 달라진다.

여기서 두 가지를 나란히 잰다.

    C0_RAW          raw 코사인 거리 (W3 그대로. 지우지 않는다)
    C0_NORMALIZED   참조 내부 leave-one-out 거리 분포에 대한 백분위

주 지표는 **coverage drift** = |실현 − 목표| 다. W3에서 이걸 재지 않아서
문제를 늦게 알았다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w4_reliability.py
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

from waste_estimation.reliability import knn_novelty, spearman  # noqa: E402
from waste_estimation.tail import (  # noqa: E402
    coverage_drift,
    is_high_waste,
    normalized_novelty,
    realized_coverage,
)

W2 = Path("research/waste_estimation/experiments/w2_external")
OUT = Path("research/waste_estimation/experiments/w4_tail")

#: 참조 축소 비율. 절반으로 줄여 W3이 겪은 크기 변화를 재현한다.
SMALL_REFERENCE_RATIO = 0.5
SMALL_REFERENCE_SEED = 20260830


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w2", type=Path, default=W2)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    policy = json.loads((args.out / "freeze.json").read_text(encoding="utf-8"))
    settings = policy["reliability"]
    k = int(settings["knnK"])
    targets_coverage = [float(value) for value in settings["coverageTargets"]]
    selected = json.loads((args.out / "metadata.json").read_text(encoding="utf-8"))[
        "selected"
    ]
    print(f"동결 해시: {policy['sha256OfRecordWithoutHash']}")
    print(f"선택된 추정기: {selected}  (여기서 다시 고르지 않는다)")

    manifest = json.loads((args.w2 / "split_manifest.json").read_text(encoding="utf-8"))
    data = load_features(policy["representation"]["backbone"])
    dev_index, _ = split_masks(data["row_ids"], manifest)
    features = data["after"][dev_index]
    targets = data["targets"][dev_index]
    groups = [data["groups"][i] for i in dev_index]

    rows = list(
        csv.DictReader(
            (args.out / "development_predictions.csv").read_text(
                encoding="utf-8"
            ).splitlines()
        )
    )
    predictions = np.array([float(row[selected]) for row in rows])
    errors = np.abs(targets - predictions)

    buckets = group_folds(groups, int(policy["outerFolds"]), int(policy["outerSeed"]))
    rng = np.random.default_rng(SMALL_REFERENCE_SEED)

    transport_rows: list[dict] = []
    accepted_rows: list[dict] = []
    raw_scores_full = np.zeros(len(targets), dtype=float)
    normalized_scores_full = np.zeros(len(targets), dtype=float)

    for fold_id, fold in enumerate(buckets):
        if not fold:
            continue
        held = np.array(fold)
        mask = np.ones(len(targets), dtype=bool)
        mask[held] = False
        train = np.flatnonzero(mask)

        # 참조 두 종류. 크기만 다르고 출처는 같다.
        small_size = max(k + 1, int(round(len(train) * SMALL_REFERENCE_RATIO)))
        small = rng.choice(train, size=small_size, replace=False)

        raw_small = knn_novelty(features[held], features[small], k=k)
        raw_full = knn_novelty(features[held], features[train], k=k)
        norm_small = normalized_novelty(features[held], features[small], k=k)
        norm_full = normalized_novelty(features[held], features[train], k=k)
        raw_scores_full[held] = raw_full
        normalized_scores_full[held] = norm_full

        for coverage in targets_coverage:
            # 보정: 작은 참조로 잰 점수의 분위수를 임계값으로 잡는다.
            # 적용: 같은 표본을 큰 참조로 다시 재고 그 임계값을 건다.
            # 방법이 참조 크기에 흔들리지 않으면 실현 coverage가 목표에 붙는다.
            for label, calibration, application in (
                ("C0_RAW", raw_small, raw_full),
                ("C0_NORMALIZED", norm_small, norm_full),
            ):
                threshold = float(np.quantile(calibration, coverage))
                realized = realized_coverage(application, threshold)
                accepted = application <= threshold
                transport_rows.append(
                    {
                        "method": label,
                        "fold": fold_id,
                        "targetCoverage": coverage,
                        "threshold": threshold,
                        "realizedCoverage": realized,
                        "coverageDrift": coverage_drift(coverage, realized),
                        "acceptedMae": (
                            float(errors[held][accepted].mean())
                            if accepted.any()
                            else None
                        ),
                        "referenceSmall": len(small),
                        "referenceFull": len(train),
                    }
                )

    print("\n=== 참조 크기가 바뀔 때의 coverage 이동 (5 fold 평균) ===")
    print(f"  참조 {int(np.mean([r['referenceSmall'] for r in transport_rows]))}건 "
          f"-> {int(np.mean([r['referenceFull'] for r in transport_rows]))}건")
    header = " ".join(f"{c:>7.0%}" for c in targets_coverage)
    print(f"  {'method':16}{header}   평균 drift")
    summary_rows = []
    for label in ("C0_RAW", "C0_NORMALIZED"):
        cells = []
        drifts = []
        for coverage in targets_coverage:
            values = [
                row
                for row in transport_rows
                if row["method"] == label and row["targetCoverage"] == coverage
            ]
            realized = float(np.mean([row["realizedCoverage"] for row in values]))
            cells.append(f"{realized:7.1%}")
            drifts.append(float(np.mean([row["coverageDrift"] for row in values])))
            accepted = [
                row["acceptedMae"] for row in values if row["acceptedMae"] is not None
            ]
            accepted_rows.append(
                {
                    "method": label,
                    "targetCoverage": coverage,
                    "meanRealizedCoverage": realized,
                    "meanCoverageDrift": drifts[-1],
                    "meanAcceptedMae": float(np.mean(accepted)) if accepted else None,
                }
            )
        mean_drift = float(np.mean(drifts))
        summary_rows.append({"method": label, "meanAbsoluteCoverageDrift": mean_drift})
        print(f"  {label:16}" + " ".join(cells) + f"   {mean_drift:.3f}")

    print("\n=== 목표 80%에서 승인 표본 MAE ===")
    for label in ("C0_RAW", "C0_NORMALIZED"):
        row = next(
            r
            for r in accepted_rows
            if r["method"] == label and r["targetCoverage"] == 0.8
        )
        print(
            f"  {label:16} 실현 {row['meanRealizedCoverage']:.1%}  "
            f"승인 MAE {row['meanAcceptedMae']:.4f}"
        )

    # --- 채택 판정 ------------------------------------------------------
    raw_drift = next(r for r in summary_rows if r["method"] == "C0_RAW")[
        "meanAbsoluteCoverageDrift"
    ]
    normalized_drift = next(
        r for r in summary_rows if r["method"] == "C0_NORMALIZED"
    )["meanAbsoluteCoverageDrift"]
    raw_accepted = next(
        r for r in accepted_rows if r["method"] == "C0_RAW" and r["targetCoverage"] == 0.8
    )["meanAcceptedMae"]
    normalized_accepted = next(
        r
        for r in accepted_rows
        if r["method"] == "C0_NORMALIZED" and r["targetCoverage"] == 0.8
    )["meanAcceptedMae"]
    adopt = normalized_drift < raw_drift and normalized_accepted <= raw_accepted + 0.01
    print(
        f"\n채택 판정: {'C0_NORMALIZED' if adopt else 'C0_RAW 유지'}  "
        f"(drift {raw_drift:.3f} -> {normalized_drift:.3f}, "
        f"승인 MAE {raw_accepted:.4f} -> {normalized_accepted:.4f})"
    )

    # --- 꼬리 안에서의 신뢰 신호 ------------------------------------------
    # W3의 결론이 새 추정기에서도 유지되는지. 여기가 §43의 판정 자리다.
    high = is_high_waste(targets)
    tail_rows = []
    for label, scores in (
        ("C0_RAW", raw_scores_full),
        ("C0_NORMALIZED", normalized_scores_full),
    ):
        tail_rows.append(
            {
                "method": label,
                "scope": "all",
                "count": len(targets),
                "spearman": spearman(scores, errors),
            }
        )
        tail_rows.append(
            {
                "method": label,
                "scope": "high_waste_only",
                "count": int(high.sum()),
                "spearman": spearman(scores[high], errors[high]),
            }
        )
    print("\n=== 신뢰 점수와 오차의 상관 ===")
    for row in tail_rows:
        print(
            f"  {row['method']:16} {row['scope']:16} n={row['count']:>3}  "
            f"Spearman {row['spearman']:+.3f}"
        )
    tail_safe = all(
        abs(row["spearman"]) >= 0.2
        for row in tail_rows
        if row["scope"] == "high_waste_only"
    )
    print(
        "\n판정: novelty 기반 거절은 고잔반 안전장치가 "
        + ("된다" if tail_safe else "**아니다**")
    )

    def write_csv(name: str, fieldnames: list[str], data: list[dict]) -> None:
        with (args.out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)

    write_csv(
        "reliability_transport.csv",
        [
            "method",
            "fold",
            "targetCoverage",
            "threshold",
            "realizedCoverage",
            "coverageDrift",
            "acceptedMae",
            "referenceSmall",
            "referenceFull",
        ],
        transport_rows,
    )
    write_csv(
        "reliability_transport_summary.csv",
        [
            "method",
            "targetCoverage",
            "meanRealizedCoverage",
            "meanCoverageDrift",
            "meanAcceptedMae",
        ],
        accepted_rows,
    )
    write_csv(
        "tail_reliability.csv", ["method", "scope", "count", "spearman"], tail_rows
    )
    # 표본별 OOF 점수. 다음 단계가 여기서 임계값을 읽는다 — ANALYSIS 분포에서
    # 다시 잡지 않기 위해서다.
    write_csv(
        "development_reliability.csv",
        ["sampleId", "group", "target", "absError", "C0_RAW", "C0_NORMALIZED"],
        [
            {
                "sampleId": rows[i]["sampleId"],
                "group": rows[i]["group"],
                "target": float(targets[i]),
                "absError": float(errors[i]),
                "C0_RAW": float(raw_scores_full[i]),
                "C0_NORMALIZED": float(normalized_scores_full[i]),
            }
            for i in range(len(rows))
        ],
    )
    (args.out / "reliability_decision.json").write_text(
        json.dumps(
            {
                "stage": "W4",
                "policyHash": policy["sha256OfRecordWithoutHash"],
                "estimator": selected,
                "rawMeanAbsoluteCoverageDrift": raw_drift,
                "normalizedMeanAbsoluteCoverageDrift": normalized_drift,
                "rawAcceptedMaeAt80": raw_accepted,
                "normalizedAcceptedMaeAt80": normalized_accepted,
                "adoptNormalized": bool(adopt),
                "adoptionRule": settings["adoptionRule"],
                "tailSafetyMechanism": bool(tail_safe),
                "tailSafetyNote": (
                    "고잔반 구간 안에서의 상관이 0 근처이면 novelty 거절은 "
                    "고잔반 안전장치가 아니다."
                ),
                "calibrated": False,
                "calibrationNote": settings["calibrationNote"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n기록: {args.out / 'reliability_transport.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
