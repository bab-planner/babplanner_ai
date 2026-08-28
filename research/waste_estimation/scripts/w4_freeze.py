"""W4 0단계 — 실험 **전에** 정한 것을 봉한다.

이 파일이 먼저 돌아야 한다. 결과를 본 뒤에 정의를 손보면 그 뒤의 모든 숫자가
자기 자신을 증명하게 된다.

봉하는 것: 후보 3개, 구간 정의, 가중치 식, 고잔반 기준, 치명적 과소추정 기준,
바깥/안쪽 fold, 선택 규칙, 신뢰 정규화 방식, coverage 목표.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w4_freeze.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from w2_prepare import CV_FOLDS, ORDINAL_EDGES, SPLIT_SEED  # noqa: E402

from waste_estimation.tail import (  # noqa: E402
    CATASTROPHIC_MARGIN,
    CATASTROPHIC_NAME,
    HIGH_WASTE_THRESHOLD,
)

OUT = Path("research/waste_estimation/experiments/w4_tail")

#: 바깥 fold는 W2/W3과 **같은 분할**이다. 다르게 잡으면 이전 숫자와 나란히
#: 놓을 수 없고, 개선처럼 보이는 것이 분할 차이일 수 있다.
OUTER_FOLDS = CV_FOLDS
OUTER_SEED = SPLIT_SEED

#: 안쪽 fold는 p_high OOF를 만드는 데만 쓴다. 모델 선택에는 쓰지 않는다.
INNER_FOLDS = 4
INNER_SEED = 20260830

#: ridge alpha는 W2가 고른 값 그대로. W4의 질문은 목적함수와 보조 정보이지
#: 하이퍼파라미터가 아니다.
RIDGE_ALPHA = 100.0

#: 보조 분류기의 정규화. **탐색하지 않는다.** ridge와 같은 값을 쓴다 —
#: 512차원에서 완전 분리를 막을 만큼 크고, 근거는 "같은 축척의 같은 임베딩"이다.
CLASSIFIER_ALPHA = 100.0

#: 신뢰 정규화 비교에 쓸 coverage 목표.
COVERAGE_TARGETS = (0.9, 0.8, 0.7, 0.5)

#: W3이 고른 이웃 수. 다시 고르지 않는다.
KNN_K = 3

SELECTION_RULE = [
    "Gate 1) 전체 MAE(바깥 fold 평균)가 R0보다 +0.02 넘게 나빠지면 탈락",
    "Gate 2) 통과 후보 중 고잔반 MAE가 가장 낮은 모델",
    "Tie) 고잔반 MAE 차이가 0.01 이내이면 치명적 과소추정 비율이 낮은 쪽",
    "Tie2) 그래도 같으면 더 단순한 모델 (R0 < R1 < R2)",
]

CANDIDATES = [
    {
        "name": "R0_STANDARD",
        "description": "W2가 고른 그대로. 표준 ridge, 가중치 없음.",
        "regressor": "Ridge",
        "alpha": RIDGE_ALPHA,
        "sampleWeight": None,
        "extraFeatures": [],
    },
    {
        "name": "R1_BALANCED",
        "description": "구간 빈도의 역제곱근 가중치를 준 ridge.",
        "regressor": "Ridge",
        "alpha": RIDGE_ALPHA,
        "sampleWeight": "inverse_sqrt_bin_frequency",
        "extraFeatures": [],
    },
    {
        "name": "R2_HIGH_AUX",
        "description": (
            "고잔반 여부 보조 분류기의 확률을 특징에 붙인 ridge. "
            "학습 표본의 p_high는 안쪽 group OOF로 만든다."
        ),
        "regressor": "Ridge",
        "alpha": RIDGE_ALPHA,
        "sampleWeight": None,
        "extraFeatures": ["p_high"],
        "classifier": "LogisticRegression",
        "classifierAlpha": CLASSIFIER_ALPHA,
        "classWeight": "balanced",
    },
]


def build() -> dict:
    return {
        "stage": "W4",
        "frozenAt": "2026-08-30",
        "purpose": (
            "DEVELOPMENT 중첩 group CV에서 꼬리 인지 회귀 후보를 비교하고 "
            "신뢰 임계값의 이식성을 잰다. 결과를 보기 전에 정의를 봉한다."
        ),
        "priorStages": {
            "w2Freeze": "662626343482afa9",
            "w3Freeze": (
                "8740ebcafbb3a70561e443640b2e81796f9a377c91355fe33936987a1e3498ab"
            ),
            "w3LimitationCarriedForward": [
                "고잔반 구간 안에서 선택적 예측이 개선을 못 냈다 (0.250 -> 0.250)",
                "큰 오차 19건 중 10건이 승인됐다",
                "raw kNN 절대 임계값이 참조 집합 크기에 따라 이동했다",
            ],
        },
        "representation": {
            "arm": "after",
            "backbone": "resnet18",
            "weights": "IMAGENET1K_V1",
            "frozen": True,
            "changed": False,
            "note": "W2 전처리 그대로. 이번 단계에서 표현을 바꾸지 않는다.",
        },
        "binEdges": list(ORDINAL_EDGES),
        "binNote": "W2의 등간격 5구간 그대로. 결과를 보고 만든 경계가 아니다.",
        "highWasteThreshold": HIGH_WASTE_THRESHOLD,
        "catastrophicName": CATASTROPHIC_NAME,
        "catastrophicRule": (
            f"정답 >= {HIGH_WASTE_THRESHOLD} 이고 정답 − 예측 >= {CATASTROPHIC_MARGIN}"
        ),
        "catastrophicMargin": CATASTROPHIC_MARGIN,
        "weightFormula": "w_b = (1/sqrt(n_b)) / mean(1/sqrt(n_b))",
        "weightCandidates": ["inverse_sqrt_bin_frequency"],
        "weightNote": (
            "후보를 하나로 고정한다. 완전 역빈도는 드문 구간에 지나친 힘을 준다."
        ),
        "weightComputedOn": "training fold labels only",
        "outerFolds": OUTER_FOLDS,
        "outerSeed": OUTER_SEED,
        "innerFolds": INNER_FOLDS,
        "innerSeed": INNER_SEED,
        "groupKey": "food category",
        "nestedNote": (
            "안쪽 fold는 p_high OOF 생성 전용이다. 모델 선택에 쓰지 않는다."
        ),
        "ridgeAlpha": RIDGE_ALPHA,
        "classifierAlpha": CLASSIFIER_ALPHA,
        "hyperparameterSearch": False,
        "candidates": CANDIDATES,
        "selectionRule": SELECTION_RULE,
        "overallGuardMargin": 0.02,
        "tailTieMargin": 0.01,
        "primaryMetrics": [
            "overall MAE",
            "overall RMSE",
            "overall Bias",
            "high-waste MAE",
            "high-waste Bias",
            "catastrophic underestimation rate",
            "p90 absolute error",
            "low MAE",
            "medium MAE",
        ],
        "classifierPrimaryMetric": "recall (high-waste)",
        "classifierSecondaryMetrics": ["precision", "f1", "auroc"],
        "classifierMetricNote": (
            "AUROC만 강조하지 않는다. 제품 위험은 고잔반을 놓치는 쪽이다."
        ),
        "reliability": {
            "knnK": KNN_K,
            "rawMethod": "C0_RAW — raw kNN cosine distance (W3 그대로, 삭제하지 않음)",
            "normalizedMethod": (
                "C0_NORMALIZED — 참조 집합 내부 leave-one-out kNN 거리 분포에 대한 "
                "경험적 백분위"
            ),
            "selfDistanceExcluded": True,
            "coverageTargets": list(COVERAGE_TARGETS),
            "primaryTransportMetric": "mean absolute coverage drift",
            "secondaryTransportMetric": "accepted MAE",
            "adoptionRule": (
                "normalized를 채택하려면 coverage drift가 raw보다 작고 accepted "
                "MAE가 크게 나빠지지 않아야 한다."
            ),
            "calibrated": False,
            "calibrationNote": (
                "백분위 점수는 보정된 정확도 확률이 아니다. 0.90이 '90% 확률로 "
                "틀린다'는 뜻이 아니다."
            ),
        },
        "analysisSplit": {
            "name": "W2_ANALYSIS",
            "size": 116,
            "groups": 10,
            "isFinalTest": False,
            "isGeneralizationTest": False,
            "applications": 1,
            "note": (
                "이미 소비된 집합이다. 동결 후 정확히 한 번 적용하고 POST-HOC "
                "ANALYSIS라고만 부른다."
            ),
        },
        "forbidden": [
            "W2_ANALYSIS에서 모델 선택",
            "W2_ANALYSIS에서 sample weight 선택",
            "W2_ANALYSIS에서 classifier threshold 선택",
            "W2_ANALYSIS에서 reliability normalization 방식 선택",
            "결과를 보고 구간 경계나 고잔반 기준 변경",
            "정답을 추론 시점 특징으로 사용",
            "category ID를 모델 입력으로 사용",
            "사람 관찰자 값을 모델 입력으로 사용",
            "OOF 잔차를 target으로 하는 error predictor (W5 후보)",
            "production fixture 교체 / FastAPI wiring",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    record = build()
    payload = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    sealed = dict(record)
    sealed["sha256OfRecordWithoutHash"] = digest
    path = args.out / "freeze.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("sha256OfRecordWithoutHash") != digest:
            raise SystemExit(
                "이미 봉한 내용과 다릅니다. 결과를 본 뒤 정의를 바꾸려는 것이라면 "
                "멈추세요.\n"
                f"  기존 {existing.get('sha256OfRecordWithoutHash')}\n"
                f"  지금 {digest}"
            )
        print(f"이미 동일하게 봉해져 있습니다: {digest}")
        return 0
    path.write_text(
        json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"동결: {path}")
    print(f"sha256 {digest}")
    print(f"후보 {len(record['candidates'])}개, 바깥 {OUTER_FOLDS} / 안쪽 {INNER_FOLDS} fold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
