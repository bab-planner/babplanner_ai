"""X3 — 동결한 W5 모델을 공개 잔반 데이터에서 그대로 돌린다.

**전이 확인(sanity check)이지 성능 측정이 아니다.**

W6와 같은 규칙을 그대로 지킨다.

- 학습은 LeFood DEVELOPMENT만. 이 데이터의 라벨은 학습에 들어가지 않는다.
- backbone·전처리·alpha·임계값 전부 W5 동결값. 여기서 고르지 않는다.
- 전역 평균 baseline도 LeFood DEVELOPMENT 평균이다. 이 데이터 평균을 쓰면
  그 순간 누수다.
- 재학습·fine-tuning·threshold tuning·모델 선택 코드 경로를 두지 않았다.

**유효 표본은 이미지 375장이 아니라 16이다.** 같은 `bonid`의 사진들은 같은
잔반을 재배치해 다시 찍은 것이라 라벨이 같다. 그래서 주 결과는 bonid로
묶은 16쌍이고, 이미지 375장 기준 수치는 참고로만 둔다.

**고잔반(>= 0.8) 표본이 0건이라 critical underestimation은 측정할 수 없다.**
0%로 적지 않는다. N/A로 적는다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\x3_external_inference.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SERVICE = Path(__file__).resolve().parents[3] / "src"
if str(_SERVICE) not in sys.path:
    sys.path.insert(0, str(_SERVICE))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from babplanner_ai.dataset.manifest import read_manifest  # noqa: E402

W2 = Path("research/waste_estimation/experiments/w2_external")
W5 = Path("research/waste_estimation/experiments/w5_structured")
OUT = Path("research/waste_estimation/experiments/x3_external_food_waste")
ROOT = Path("research/waste_estimation/data/external/food_waste")

#: 결과 어디에도 밥플래너 성능으로 읽히지 않게 붙인다.
BENCHMARK = "external-public"
SPLIT_LABEL = "EXTERNAL_PUBLIC_FOOD_WASTE"

#: 고잔반 경계. W4/W5와 같은 값을 쓴다. 여기서 새로 정하지 않는다.
HIGH_WASTE_THRESHOLD = 0.8


def na(value: float | None, fmt: str = "{:.4f}") -> str:
    """없는 값을 0으로 적지 않는다."""
    return "N/A" if value is None else fmt.format(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifest.csv")
    parser.add_argument("--images", type=Path, default=ROOT)
    parser.add_argument("--w2", type=Path, default=W2)
    parser.add_argument("--w5", type=Path, default=W5)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    samples = read_manifest(args.manifest)
    approved = [s for s in samples if s.is_supervised_usable]
    print(f"매니페스트 {len(samples)}행, 정답 사용 가능 {len(approved)}건")

    import numpy as np

    from w2_experiment import load_features, split_masks
    from w4_experiment import tail_scores
    from w5_experiment import NAMES, fit_predict_all

    from waste_estimation.baselines import clip
    from waste_estimation.features import FrozenBackbone

    policy = json.loads((args.w5 / "freeze.json").read_text(encoding="utf-8"))
    development = json.loads((args.w5 / "metadata.json").read_text(encoding="utf-8"))
    selected = development["selected"]
    print(f"동결 해시: {policy['sha256OfRecordWithoutHash']}")
    print(f"선택된 구조: {selected}  (여기서 바꾸지 않는다)")

    # --- 학습은 LeFood DEVELOPMENT만 --------------------------------------
    manifest = json.loads((args.w2 / "split_manifest.json").read_text(encoding="utf-8"))
    data = load_features(policy["representation"]["backbone"])
    dev_index, _ = split_masks(data["row_ids"], manifest)
    dev_features, dev_targets = data["after"][dev_index], data["targets"][dev_index]
    print(f"학습 표본: LeFood DEVELOPMENT {len(dev_index)}건 (외부 라벨 미사용)")

    # --- 외부 이미지를 같은 전처리로 임베딩 --------------------------------
    backbone = FrozenBackbone(name=policy["representation"]["backbone"]).load()
    paths = [args.images / s.image_path for s in approved]
    features = backbone.embed(paths)
    print(f"임베딩: {backbone.name} / {backbone.weights_name} (frozen)")

    targets = np.array([s.waste_rate for s in approved], dtype=float)
    groups = [s.participant_group_id or s.capture_session_id for s in approved]

    raw, _ = fit_predict_all(
        dev_features,
        dev_targets,
        features,
        float(policy["ridgeAlpha"]),
        float(policy["classifierAlpha"]),
        tuple(float(v) for v in policy["cumulativeThresholds"]),
        int(policy["candidates"][2]["minExpertSamples"]),
    )
    bounded = {name: clip(raw[name], 0.0, 1.0) for name in NAMES}
    # LeFood DEVELOPMENT 평균이다. 이 데이터 평균이 아니다.
    bounded["B0_GLOBAL_MEAN"] = np.full(len(targets), float(dev_targets.mean()))

    # --- 그룹 단위로 접는다 -------------------------------------------------
    #
    # 같은 bonid 사진들은 같은 잔반을 다시 찍은 것이라 라벨이 하나다.
    # 이미지 단위로 평균을 내면 사진이 많은 잔반 상태가 그만큼 더 세어진다.
    order = sorted(set(groups))
    index_of = {g: [i for i, gg in enumerate(groups) if gg == g] for g in order}
    group_targets = np.array([targets[index_of[g]][0] for g in order])
    for g in order:
        assert len({round(float(t), 6) for t in targets[index_of[g]]}) == 1, g

    high_count = int((group_targets >= HIGH_WASTE_THRESHOLD).sum())
    print(
        f"\n{SPLIT_LABEL}: 이미지 {len(approved)}장 / **유효 표본 {len(order)}** / "
        f"고잔반(>= {HIGH_WASTE_THRESHOLD}) {high_count}건"
    )
    if high_count == 0:
        print("  고잔반 표본이 0건이다. critical underestimation 은 측정할 수 없다 (N/A).")

    rows = []
    print(f"\n=== 결과 (bounded, {BENCHMARK}) ===")
    print(f"{'candidate':36} {'MAE(16)':>9} {'Bias(16)':>9} {'MAE(375)':>9} {'치명':>6}")
    for name in ("B0_GLOBAL_MEAN",) + NAMES:
        group_preds = np.array([float(bounded[name][index_of[g]].mean()) for g in order])
        g_scores = tail_scores(group_targets, group_preds)
        i_scores = tail_scores(targets, bounded[name])
        rows.append(
            {
                "candidate": name,
                "isSelected": name == selected,
                "benchmark": BENCHMARK,
                "split": SPLIT_LABEL,
                "effectiveSamples": len(order),
                "groupMae": g_scores["mae"],
                "groupRmse": g_scores["rmse"],
                "groupBias": g_scores["bias"],
                "groupP90AbsError": g_scores["p90AbsError"],
                "imageMae": i_scores["mae"],
                "imageBias": i_scores["bias"],
                "highWasteCount": high_count,
                "highMae": g_scores.get("highMae"),
                "catastrophicRate": g_scores.get("catastrophicRate"),
            }
        )
        mark = " *" if name == selected else "  "
        print(
            f"{name:36}{mark}{g_scores['mae']:8.4f} {g_scores['bias']:+9.4f} "
            f"{i_scores['mae']:9.4f} {na(g_scores.get('catastrophicRate'), '{:.1%}'):>6}"
        )

    # --- 선택된 구조의 16쌍 전체 -------------------------------------------
    sel_preds = np.array([float(bounded[selected][index_of[g]].mean()) for g in order])
    pairs = []
    print(f"\n=== {selected} 의 예측 / 정답 16쌍 ===")
    print(f"{'group':16} {'n':>4} {'GT':>8} {'pred':>8} {'error':>9}")
    for g, t, p in sorted(zip(order, group_targets, sel_preds), key=lambda r: r[1]):
        err = float(p - t)
        pairs.append(
            {
                "group": g,
                "images": len(index_of[g]),
                "groundTruth": float(t),
                "prediction": float(p),
                "error": err,
                "absError": abs(err),
            }
        )
        print(f"{g:16} {len(index_of[g]):4} {t:8.4f} {p:8.4f} {err:+9.4f}")

    errs = sorted(abs(row["error"]) for row in pairs)
    signed = [row["error"] for row in pairs]
    print("\n=== 오차 분포 (16쌍) ===")
    print(f"  절대오차  중앙 {statistics.median(errs):.4f}  최대 {errs[-1]:.4f}  최소 {errs[0]:.4f}")
    print(f"  부호오차  평균 {statistics.fmean(signed):+.4f}   (음수 = 과소추정)")
    print(f"  과소추정  {sum(1 for e in signed if e < 0)}/16")

    worst = sorted(pairs, key=lambda r: -r["absError"])[:3]
    print("\n=== 가장 크게 빗나간 3건 ===")
    for row in worst:
        direction = "과소" if row["error"] < 0 else "과대"
        print(
            f"  {row['group']:16} GT {row['groundTruth']:.4f} -> {row['prediction']:.4f} "
            f"({direction} {row['absError']:.4f}, 사진 {row['images']}장)"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(
        json.dumps(
            {
                "benchmark": BENCHMARK,
                "split": SPLIT_LABEL,
                "dataset": "Voxel51/food-waste-dataset",
                "datasetRegistry": "research/waste_estimation/datasets/x1_food_waste_registry.json",
                "frozenPolicySha256": policy["sha256OfRecordWithoutHash"],
                "selected": selected,
                "trainingSource": "LeFood DEVELOPMENT only",
                "globalMeanSource": "LeFood DEVELOPMENT mean",
                "images": len(approved),
                "effectiveSamples": len(order),
                "highWasteSamples": high_count,
                "criticalUnderestimation": None,
                "criticalUnderestimationNote": (
                    "고잔반(>= 0.8) 표본이 0건이라 측정할 수 없다. 0%가 아니라 N/A다."
                ),
                "candidates": rows,
                "pairs": pairs,
                "claimLimit": (
                    "외부 공개 도메인에서의 전이 확인이다. 밥플래너 급식소 성능이 아니며, "
                    "유효 표본 16으로 잰 값이라 순위 비교 이상으로 인용하지 않는다."
                ),
                "interpretationLimits": [
                    "이 결과는 '구조가 전이되지 않는다'와 'frozen resnet18 특징이 이 "
                    "영상에서 분포 밖이다'를 구분하지 못한다. 둘 다 같은 증상을 낸다.",
                    "정답 분포가 0.0618~0.5476 로 잘려 있다. 고잔반 구간이 없는 곳에서 "
                    "고잔반을 놓치지 않으려 위로 미는 구조는 그 밀기가 그대로 오차가 된다.",
                    "촬영 조건이 다르다 — 독일 급식소, 둥근 접시, 다른 카메라.",
                    "이 결과로 W5 선택을 되돌리지 않는다. 그러려면 밥플래너 실제 데이터가 "
                    "있어야 한다.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n{args.out / 'metrics.json'}")
    print("\n이 숫자는 external-public 벤치마크다. 밥플래너 성능이 아니다.")
    print(
        "이 결과는 '구조가 전이되지 않는다'와 'frozen 특징이 이 영상에서 분포 밖이다'를\n"
        "구분하지 못한다. 둘 다 같은 증상을 낸다. W5 선택은 이 결과로 되돌리지 않는다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
