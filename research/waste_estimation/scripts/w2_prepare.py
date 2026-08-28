"""W2 준비 — 라벨 생성, 분할, 동결.

순서가 중요하다.

    1. 공식 배포본을 읽고 관문을 통과한 표본만 남긴다
    2. 음식 범주 단위로 DEVELOPMENT / TEST를 나눈다
    3. **TEST를 보기 전에** 설정과 판정 규칙을 동결하고 지문을 남긴다

TEST는 마지막에 한 번만 채점한다. 이 스크립트가 만드는 freeze.json이 그
약속의 기록이다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w2_prepare.py
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
_SERVICE = Path(__file__).resolve().parents[3] / "src"
if str(_SERVICE) not in sys.path:
    sys.path.insert(0, str(_SERVICE))

from waste_estimation import lefood  # noqa: E402
from waste_estimation.schemas import SampleSource, WasteSample  # noqa: E402
from waste_estimation.splits import (  # noqa: E402
    group_holdout_split,
    group_sizes,
    leaked_groups,
)

RAW = Path("research/waste_estimation/data/raw/lefood_set")
OUT = Path("research/waste_estimation/experiments/w2_external")

#: 결과를 보기 전에 정한 값들. 아래 어느 것도 TEST 결과를 보고 바꾸지 않는다.
TEST_RATIO = 0.30
SPLIT_SEED = 42

#: 순서형 구간. **등간격 5구간.** 개발 집합에서 최적화한 경계가 아니다.
#: 0~0.2 / 0.2~0.4 / 0.4~0.6 / 0.6~0.8 / 0.8~1.0 이고 경계는 아래 구간에
#: 들어간다(`metrics.to_class_index`의 규약).
ORDINAL_EDGES = (0.2, 0.4, 0.6, 0.8)

#: 오차를 잔반 수준별로 나눌 때 쓰는 경계. 순서형 구간과 **같은 규칙**을
#: 쓴다. 분석마다 다른 경계를 쓰면 표끼리 비교할 수 없다.
LEVEL_EDGES = (0.2, 0.8)
LEVEL_NAMES = ("low", "medium", "high")

#: DEVELOPMENT 그룹 교차검증의 폴드 수. 24개 범주를 5로 나눈다.
CV_FOLDS = 5

#: ridge 정규화 후보. 좁은 격자를 미리 고정한다. TEST를 보고 늘리지 않는다.
ALPHA_GRID = (1.0, 10.0, 100.0, 1000.0)

#: 후보 선택 규칙. DEVELOPMENT에서만 적용한다.
SELECTION_RULE = [
    "1) DEVELOPMENT 그룹 교차검증 MAE 최소",
    "2) 차이가 0.005 이내면 |Bias|가 작은 쪽",
    "3) 그래도 같으면 더 단순한 쪽 (특징 차원이 작은 쪽)",
]


def to_samples(records) -> list[WasteSample]:
    return [
        WasteSample(
            sample_id=record.row_id,
            source=SampleSource.PUBLIC,
            group_id=record.group_id,
            waste_rate=record.external_waste_ratio,
            image_reference=record.after_image,
        )
        for record in records
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    report = lefood.load(args.raw)
    summary = lefood.describe(report)
    print(f"표 행수        {summary['totalRows']}")
    print(f"사용 가능 표본 {summary['usableRecords']}")
    for reason, count in summary["rejections"].items():
        if count:
            print(f"  버림 {reason:24} {count}")
    print(f"음식 범주      {summary['foodCategories']}")

    records = report.records
    samples = to_samples(records)
    by_id = {record.row_id: record for record in records}

    development, test = group_holdout_split(
        samples, holdout_ratio=TEST_RATIO, seed=SPLIT_SEED
    )
    leaked = leaked_groups(development, test)
    if leaked:
        raise SystemExit(f"분할 누수: {sorted(leaked)}")

    print(f"\nDEVELOPMENT    {len(development)}건 / {len({s.group_id for s in development})}범주")
    print(f"TEST           {len(test)}건 / {len({s.group_id for s in test})}범주")
    print(f"누수           없음")

    args.out.mkdir(parents=True, exist_ok=True)

    (args.out / "dataset_metadata.json").write_text(
        json.dumps(
            {
                **summary,
                "citation": lefood.CITATION,
                "licenseUrl": lefood.LICENSE_URL,
                "note": (
                    "외부 벤치마크다. 밥플래너 급식소 데이터가 아니며 여기서 "
                    "나온 수치를 급식소 성능이라고 말하지 않는다."
                ),
                "groupKey": (
                    "food_category. 참가자/회차 식별자가 데이터에 없어 "
                    "지금 쓸 수 있는 가장 강한 묶음이다."
                ),
                "groupSizes": dict(sorted(group_sizes(samples).items())),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    split_manifest = {
        "splitSeed": SPLIT_SEED,
        "testRatio": TEST_RATIO,
        "development": {
            "count": len(development),
            "groups": sorted({s.group_id for s in development}),
            "sampleIds": sorted(s.sample_id for s in development),
        },
        "test": {
            "count": len(test),
            "groups": sorted({s.group_id for s in test}),
            "sampleIds": sorted(s.sample_id for s in test),
        },
    }
    (args.out / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    freeze = {
        "stage": "W2",
        "frozenAt": "2026-08-29",
        "purpose": (
            "TEST를 보기 전에 데이터·라벨·분할·모델 후보·판정 규칙을 고정한다."
        ),
        "dataset": {
            "name": "LeFood-Set",
            "source": lefood.SOURCE_URL,
            "doi": lefood.DOI,
            "version": lefood.VERSION,
            "license": lefood.LICENSE,
            "usableRecords": len(records),
        },
        "labelFormula": "external_waste_ratio = after_mass_g / before_mass_g",
        "labelName": (
            "external_waste_ratio. 밥플래너의 waste_rate와 계산식은 같지만 "
            "다른 데이터의 값이므로 이름을 분리한다."
        ),
        "labelGate": [
            "before_mass_g > 0",
            "after_mass_g >= 0",
            "after_mass_g <= before_mass_g",
            "before/after 이미지가 모두 배포본에 있어야 한다",
            "before/after 파일명 접두사가 같아야 한다",
        ],
        "split": {
            "unit": "food_category",
            "seed": SPLIT_SEED,
            "testRatio": TEST_RATIO,
            "developmentCount": len(development),
            "testCount": len(test),
        },
        "preprocessing": {
            "deterministic": True,
            "randomAugmentation": False,
            "note": (
                "짝 실험에서 before/after에 서로 다른 무작위 변환을 걸면 "
                "비교가 깨진다. 증강을 쓰지 않는다."
            ),
        },
        "backbone": {
            "candidates": ["mobilenet_v3_small", "resnet18"],
            "frozen": True,
            "fineTuning": False,
            "weightsNote": (
                "torchvision 배포 ImageNet 가중치. 소프트웨어(BSD-3-Clause)와 "
                "가중치의 조건은 별개다. 연구·평가 용도로 쓰며 상용 배포 "
                "적격이라고 말하지 않는다."
            ),
        },
        "regressor": "Ridge (표준화 + L2, 정규방정식)",
        "alphaGrid": list(ALPHA_GRID),
        "cvFolds": CV_FOLDS,
        "cvNote": (
            "DEVELOPMENT 안에서만 그룹 단위 교차검증으로 alpha와 후보를 "
            "고른다. TEST는 마지막에 한 번만 본다."
        ),
        "inputArms": {
            "after": "식후 이미지만. 운영 API와 같은 입력",
            "concat": "[before, after]",
            "concat_diff": "[before, after, after-before]",
        },
        "ordinalEdges": list(ORDINAL_EDGES),
        "levelEdges": list(LEVEL_EDGES),
        "levelNames": list(LEVEL_NAMES),
        "primaryMetric": "MAE (raw regression output)",
        "secondaryMetrics": ["RMSE", "Median AE", "Bias"],
        "clippingPolicy": (
            "raw와 clipped를 나란히 보고한다. 지표 계산 전에 몰래 자르지 "
            "않는다."
        ),
        "selectionRule": SELECTION_RULE,
        "informationGapMetric": "dMAE = MAE(after-only) - MAE(before+after)",
        "hypothesis": {
            "H-W2": (
                "before+after가 after-only보다 여러 분할에서 일관되게 좋다면, "
                "운영 API의 after-only 입력에는 분모가 보이지 않는 정보 격차가 "
                "있다. 한 번의 분할에서 0.01 좋아진 것으로는 지지하지 않는다."
            )
        },
        "forbidden": [
            "TEST를 보고 모델/임계값/구간 변경",
            "지표 전 clip",
            "픽셀 면적 비율을 질량비로 사용",
            "이 결과를 밥플래너 급식소 성능이라고 표현",
            "fine-tuning (W3 이후 후보)",
        ],
    }

    payload = json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    record = dict(freeze)
    record["sha256OfRecordWithoutHash"] = digest
    (args.out / "freeze.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"\n동결: {args.out / 'freeze.json'}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
