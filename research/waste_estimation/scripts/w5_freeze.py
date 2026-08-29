"""W5 0단계 — 실험 **전에** 정한 것을 봉한다.

W4와 같은 규칙이다. 이 파일이 먼저 돌고, 이미 봉한 내용과 달라지면 멈춘다.

이번에 봉하는 것 중 하나가 특히 중요하다. **주 지표를 자른 예측 기준으로
바꾼다.** S1은 구조상 [0,1] 안에서만 답하고, 제품 API도 [0,1]이다. raw
기준으로 재면 S1만 범위 밖 오차가 없어 유리해진다 — 구조가 아니라 범위를
비교하게 된다. W2의 주 지표와 달라진 점이라 보고서에 적는다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w5_freeze.py
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

from waste_estimation.structured import CUMULATIVE_THRESHOLDS  # noqa: E402
from waste_estimation.tail import (  # noqa: E402
    CATASTROPHIC_MARGIN,
    CATASTROPHIC_NAME,
    HIGH_WASTE_THRESHOLD,
)

OUT = Path("research/waste_estimation/experiments/w5_structured")

RIDGE_ALPHA = 100.0
CLASSIFIER_ALPHA = 100.0

SELECTION_RULE = [
    "Gate 1) 전체 MAE(자른 예측, 바깥 fold 평균)가 S0보다 +0.02 넘게 나빠지면 탈락",
    "Gate 2) 통과 후보 중 치명적 과소추정 비율(pooled)이 가장 낮은 모델",
    "Tie) 치명적 비율 차이가 0.02 이내이면 고잔반 MAE가 낮은 쪽",
    "Tie2) 그래도 같으면 전체 MAE가 낮은 쪽",
    "Tie3) 그래도 같으면 더 단순한 모델 (S0 < S1 < S2)",
]

CANDIDATES = [
    {
        "name": "S0_STANDARD_RIDGE",
        "description": "W2/W4 baseline 그대로. frozen 임베딩 + Ridge alpha=100.",
        "outputHead": "direct point regression",
        "alpha": RIDGE_ALPHA,
    },
    {
        "name": "S1_CUMULATIVE_ORDINAL_EXPECTATION",
        "description": (
            "임계값마다 이진 분류기로 누적 확률을 내고, 단조 보정 후 구간 "
            "확률로 바꾸고, 학습 fold 구간 평균을 대표값으로 기대값을 만든다."
        ),
        "outputHead": "cumulative ordinal distribution -> expected value",
        "thresholds": list(CUMULATIVE_THRESHOLDS),
        "classifier": "LogisticRegression",
        "classifierAlpha": CLASSIFIER_ALPHA,
        "classWeight": "balanced",
        "monotonicRepair": "running minimum (누적 최솟값, 왼쪽에서 오른쪽으로)",
        "monotonicRepairNote": (
            "확률을 올리는 방향의 보정(오른쪽부터 최댓값)은 고잔반 확률을 "
            "낙관적으로 부풀린다. 과소추정이 관심사이므로 보수적인 쪽을 고른다."
        ),
        "binRepresentatives": "학습 fold 정답의 구간별 평균",
        "binRepresentativeFallback": "구간 중점 (빈 구간일 때만)",
        "differsFromW2T2": (
            "W2 T2는 단일 순서형 클래스 예측이었다. S1은 전체 누적 분포를 "
            "만들고 기대값으로 되돌린다. 같은 실험이 아니다."
        ),
    },
    {
        "name": "S2_SOFT_HIGH_LOW_MIXTURE",
        "description": (
            "고잔반 게이트 점수로 저잔반 전문가와 고잔반 전문가의 예측을 "
            "부드럽게 섞는다."
        ),
        "outputHead": "soft mixture of two ridge experts",
        "gate": "LogisticRegression(class_weight=balanced), target y >= 0.8",
        "gateAlpha": CLASSIFIER_ALPHA,
        "gateNote": (
            "게이트 출력은 gating score다. 보정된 확률이라고 부르지 않는다."
        ),
        "experts": {
            "nonHigh": "학습 fold의 y < 0.8 표본만, Ridge alpha=100",
            "high": "학습 fold의 y >= 0.8 표본만, Ridge alpha=100",
        },
        "hardSwitching": False,
        "hardSwitchingNote": (
            "딱딱한 전환은 경계에서 예측이 튄다. 여기서 재려는 것은 혼합 "
            "구조이지 전환 규칙이 아니다."
        ),
        "minExpertSamples": 10,
    },
]


def build() -> dict:
    return {
        "stage": "W5",
        "frozenAt": "2026-08-31",
        "purpose": (
            "표현과 정규화를 그대로 두고 **출력 구조만** 바꿔 꼬리 크기 예측이 "
            "개선되는지 본다. 결과를 보기 전에 정의를 봉한다."
        ),
        "priorStages": {
            "w4Freeze": (
                "05426ff78705d3c3a125a44552747460105335c5951ec8b7fa1982c0960e97e6"
            ),
            "w4CorrectedConclusion": (
                "frozen 표현은 고잔반 표본을 판별하는 신호를 상당히 가지고 있다"
                "(AUROC 0.912). 그러나 W4는 그 표현이 정확한 꼬리 크기 회귀에 "
                "충분한 정보를 담고 있음을 보이지 않았다. 확인된 것은 단순 "
                "재가중과 단순 보조 특징 결합이 부족하다는 것뿐이다."
            ),
            "w4CatastrophicAggregation": (
                "pooled(표본을 한 번씩)와 fold-macro(fold 비율의 평균)를 구분해 "
                "다시 적었다. W5는 pooled를 주 집계로 쓴다."
            ),
        },
        "representation": {
            "arm": "after",
            "backbone": "resnet18",
            "weights": "IMAGENET1K_V1",
            "frozen": True,
            "changed": False,
            "identicalAcrossCandidates": True,
            "note": "후보 간 표현 차이 0. fine-tuning·backbone 변경·증강 없음.",
        },
        "binEdges": list(ORDINAL_EDGES),
        "cumulativeThresholds": list(CUMULATIVE_THRESHOLDS),
        "binConvention": "왼쪽 닫힘 (y >= t). 마지막 칸이 고잔반 집합과 정확히 같다.",
        "binConventionNote": (
            "W4의 tail.bin_index는 to_class_index 규약(> edge)을 따른다. 정답이 "
            "정확히 0.8인 표본 1건에서 갈리며, W5는 왼쪽 닫힘으로 통일한다."
        ),
        "highWasteThreshold": HIGH_WASTE_THRESHOLD,
        "catastrophicName": CATASTROPHIC_NAME,
        "catastrophicRule": (
            f"정답 >= {HIGH_WASTE_THRESHOLD} 이고 정답 − 예측 >= {CATASTROPHIC_MARGIN}"
        ),
        "catastrophicMargin": CATASTROPHIC_MARGIN,
        "catastrophicPrimaryAggregation": "pooled",
        "recoveredDefinition": (
            "S0에서 치명적이었던 표본이 선택 모델에서 정답 − 예측 < 0.30이 되면 "
            "RECOVERED"
        ),
        "newRegressionDefinition": (
            "S0에서 치명적이 아니었는데 선택 모델에서 치명적이 되면 "
            "NEW_CATASTROPHIC_REGRESSION"
        ),
        "outerFolds": CV_FOLDS,
        "outerSeed": SPLIT_SEED,
        "groupKey": "food category",
        "foldReuseNote": "W4와 동일한 바깥 fold. 후보마다 새로 만들지 않는다.",
        "ridgeAlpha": RIDGE_ALPHA,
        "classifierAlpha": CLASSIFIER_ALPHA,
        "hyperparameterSearch": False,
        "candidates": CANDIDATES,
        "predictionPolicy": {
            "primary": "BOUNDED — clip(prediction, 0, 1) 후 지표 계산",
            "supplementary": "RAW — 자르지 않은 출력도 함께 기록",
            "reason": (
                "S1은 구조상 [0,1] 안에서만 답하고 제품 API도 [0,1]이다. raw로 "
                "재면 S1만 범위 밖 오차가 없어 유리해진다 — 구조가 아니라 범위를 "
                "비교하게 된다."
            ),
            "differsFromW2": (
                "W2의 주 지표는 raw였다. W5에서 BOUNDED로 바꾼 것을 보고서에 "
                "명시한다."
            ),
        },
        "selectionRule": SELECTION_RULE,
        "overallGuardMargin": 0.02,
        "catastrophicTieMargin": 0.02,
        "primaryMetrics": [
            "overall MAE (bounded)",
            "overall RMSE (bounded)",
            "overall Bias (bounded)",
        ],
        "tailMetrics": [
            "high-waste MAE",
            "high-waste Bias",
            "catastrophic count",
            "catastrophic rate (pooled, macro 둘 다 기록)",
        ],
        "guardMetrics": ["low-waste MAE", "medium-waste MAE", "p90 absolute error"],
        "statisticalClaimLimit": (
            "음식 범주 24개뿐이다. p-value를 억지로 만들지 않는다. fold 단위 "
            "일관성으로만 말한다."
        ),
        "reliabilityFrozen": True,
        "reliabilityNote": (
            "W3/W4에서 novelty 기반 신뢰가 고잔반 내부 안전장치가 아님을 확인했다. "
            "W5에서는 신뢰 후보를 추가하지 않고 재튜닝하지도 않는다. W4의 "
            "정규화 kNN coverage 결과는 그대로 유지한다."
        ),
        "analysisSplit": {
            "name": "W2_ANALYSIS",
            "size": 116,
            "groups": 10,
            "isFinalTest": False,
            "isGeneralizationTest": False,
            "isPristineValidation": False,
            "applications": 1,
            "resultName": "POST-HOC FROZEN ANALYSIS",
        },
        "freshTest": {
            "available": False,
            "note": (
                "동일 비율 의미 + VERIFIED 라이선스의 새 데이터셋을 확보하지 "
                "못했다. 찾기 위해 범위를 늘리지 않는다. structured-head "
                "development complete; pristine external validation pending."
            ),
        },
        "forbidden": [
            "ANALYSIS에서 후보 선택",
            "ANALYSIS에서 threshold 변경",
            "ANALYSIS에서 head 구조 변경",
            "결과를 보고 단조 보정 규칙 변경",
            "결과를 보고 구간 대표값 규칙 변경",
            "결과를 보고 예측 정책(bounded/raw) 변경",
            "평가 fold 정답으로 구간 대표값 계산",
            "평가 fold 표본으로 전문가 학습",
            "backbone 변경 / fine-tuning / 데이터 증강",
            "표본 재가중 추가 실험 (W4에서 끝났다)",
            "신뢰 방법 재튜닝",
            "분위 회귀 (이번 범위 밖. S1/S2 결과를 보고 판단)",
            "production fixture 교체 / FastAPI wiring",
            "사람 관찰자 값을 모델 입력으로 사용",
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
    print(f"후보 {len(record['candidates'])}개 · 주 지표 BOUNDED · 치명 집계 pooled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
