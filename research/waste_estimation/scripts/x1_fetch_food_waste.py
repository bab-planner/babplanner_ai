"""X1 — 공개 잔반 데이터(Voxel51/food-waste-dataset)를 내려받고 지문을 남긴다.

**원본은 저장소에 커밋하지 않는다.** 이 스크립트와 레지스트리만 커밋하고,
데이터는 gitignore 아래로만 들어간다. 다른 사람이 같은 revision을 다시
받아 같은 지문을 얻을 수 있으면 그것으로 재현된다.

revision을 고정하는 이유는 분명하다. 공개 데이터는 조용히 바뀐다. 어제
잰 숫자와 오늘 잰 숫자가 다를 때, 모델이 달라진 것인지 데이터가 달라진
것인지 구분할 수 있어야 한다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\x1_fetch_food_waste.py

이 데이터가 무엇이고 무엇이 아닌지는 레지스트리의 `usage` 항목에 적혀
있다. 요약하면, **유효 표본은 이미지 375장이 아니라 라벨 16개**다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path

REPO = "Voxel51/food-waste-dataset"

#: 고정한 revision. 바꾸면 레지스트리의 지문이 전부 달라진다.
REVISION = "ddaa9e2e8c03aa3914fb74988144f3665f8f9462"

BASE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"

#: 라벨과 스키마. 임베딩과 세그멘테이션은 우리가 쓰지 않지만 samples.json
#: 안에 같이 들어 있어 따로 뺄 수 없다.
METADATA_FILES = ("metadata.json", "samples.json")

EXPECTED_SAMPLES = 375


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(rel: str, dest: Path) -> Path:
    """이미 있으면 다시 받지 않는다. 375장을 매번 받을 이유가 없다."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    urllib.request.urlretrieve(BASE + rel, dest)
    return dest


def load_samples(root: Path) -> list[dict]:
    payload = json.loads((root / "samples.json").read_text(encoding="utf-8"))
    return payload["samples"] if isinstance(payload, dict) else payload


def verify(samples: list[dict]) -> dict:
    """받은 것이 우리가 분석한 그 데이터인지 확인한다.

    여기서 보는 것은 품질이 아니라 **동일성**이다. 어긋나면 revision이
    바뀌었거나 다운로드가 깨진 것이고, 어느 쪽이든 그대로 쓰면 안 된다.
    """
    problems: list[str] = []
    if len(samples) != EXPECTED_SAMPLES:
        problems.append(f"표본 수가 {len(samples)}입니다. {EXPECTED_SAMPLES}이어야 합니다.")

    plate_matches = 0
    labels: set[tuple] = set()
    for sample in samples:
        plate = sample.get("weight_per_plate") or []
        returns = sample.get("return_quantity") or []
        before = sample.get("weight_before")
        if before and abs(sum(plate) - before) < 1e-6:
            plate_matches += 1
        labels.add((before, tuple(returns)))

    if plate_matches != len(samples):
        # 이 등식이 깨지면 served 질량의 출처가 둘이 되고, 어느 쪽이 맞는지
        # 알 수 없다.
        problems.append(
            f"sum(weight_per_plate) == weight_before 가 {plate_matches}/{len(samples)}건만 "
            "성립합니다."
        )

    rates = sorted(
        {
            round(sum(s["return_quantity"]) / sum(s["weight_per_plate"]), 4)
            for s in samples
            if sum(s.get("weight_per_plate") or [0])
        }
    )
    return {
        "images": len(samples),
        "distinctLabels": len(labels),
        "distinctWasteRates": len(rates),
        "wasteRateMin": rates[0] if rates else None,
        "wasteRateMax": rates[-1] if rates else None,
        "highWasteSamples": sum(1 for r in rates if r >= 0.8),
        "plateSumMatchesWeightBefore": plate_matches,
        "splits": dict(Counter(s.get("split") for s in samples)),
        "distinctBonIds": len({s.get("bonid") for s in samples}),
        "distinctDishes": len({s.get("dish") for s in samples}),
        "problems": problems,
    }


def build_registry(root: Path, samples: list[dict], checks: dict) -> dict:
    files = {}
    for name in METADATA_FILES:
        path = root / name
        files[name] = {"sha256": sha256_of(path), "bytes": path.stat().st_size}

    images = sorted({s["filepath"] for s in samples})
    image_digest = hashlib.sha256()
    total_bytes = 0
    for rel in images:
        path = root / rel
        image_digest.update(sha256_of(path).encode())
        total_bytes += path.stat().st_size

    return {
        "datasetKey": "food_waste_voxel51",
        "source": f"https://huggingface.co/datasets/{REPO}",
        "originalSource": "https://huggingface.co/datasets/AI-ServicesBB/food-waste-dataset",
        "curatedBy": "L. Stroetmann, a la QUARTO, AI Service Center at Hasso Plattner Institute, Voxel51",
        "license": "MIT",
        "licenseVerifiedFrom": f"https://huggingface.co/api/datasets/{REPO} (tag license:mit)",
        "commercialUse": "허용 (MIT)",
        "revision": REVISION,
        "downloadedOn": date.today().isoformat(),
        "files": files,
        "images": {
            "count": len(images),
            "totalBytes": total_bytes,
            "sha256OfSortedFileDigests": image_digest.hexdigest(),
        },
        "checks": checks,
        "annotationMeaning": {
            "weight_per_plate": "재료별 배식 질량(g). 합이 weight_before와 같다.",
            "return_quantity": "재료별 잔반 질량(g). **원시값이다.**",
            "return_percentage": "return_quantity / weight_per_plate. 파생값이므로 쓰지 않는다.",
            "weight_before": "배식 총 질량(g) = sum(weight_per_plate)",
            "weight_after": (
                "**쓰지 않는다.** 375건 중 160건이 정확히 0이고(test split 전체), "
                "모든 *_after 필드가 *_before의 균일 배수라 측정값이 아니라 파생값이다. "
                "'먹은 양'과 30/375, '남긴 양'과 0/375만 일치한다."
            ),
            "bonid": "영수증 식별자. 같은 값이면 같은 식사이므로 분할 그룹 키로 쓴다.",
            "dish": "메뉴 이름(독일어). bonid와 1:1이다.",
            "yoloe_segmentation": "기계 생성 세그멘테이션. 우리 과제에 쓰지 않는다.",
            "dinov2-*-embeddings": "기계 생성 임베딩. 우리는 W2 backbone을 쓰므로 쓰지 않는다.",
        },
        "groundTruth": {
            "formula": "waste_rate = sum(return_quantity) / sum(weight_per_plate)",
            "matchesBabplannerContract": True,
            "note": "우리 leftover_edible_mass_g / served_edible_mass_g 와 같은 정의다.",
        },
        "usage": {
            "role": "external-domain transfer sanity check",
            "benchmark": "external-public",
            "effectiveSampleSize": checks["distinctLabels"],
            "warning": (
                "이미지 375장은 서로 다른 식사 375건이 아니다. 측정된 잔반 상태 "
                f"{checks['distinctLabels']}개를 여러 각도로 다시 찍은 것이며, 구조적으로 "
                "연출 파일럿(CONTROLLED_STAGED)과 같다."
            ),
            "forbidden": [
                "재학습 / fine-tuning",
                "threshold tuning",
                "모델 선택",
                "high-waste critical underestimation 성능 주장 (고잔반 표본 0건)",
                "밥플래너 성능으로 인용",
            ],
            "allowedReporting": [
                "전체 MAE",
                "prediction / GT 쌍",
                "오차 분포",
                "명백한 실패 사례",
            ],
        },
        "note": "원본 파일은 커밋하지 않는다. 지문만 남긴다.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("research/waste_estimation/data/external/food_waste"),
        help="원본을 둘 곳. gitignore 아래여야 한다",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("research/waste_estimation/datasets/x1_food_waste_registry.json"),
    )
    args = parser.parse_args(argv)

    print(f"{REPO} @ {REVISION[:12]}")
    for name in METADATA_FILES:
        fetch(name, args.root / name)
        print(f"  {name}")

    samples = load_samples(args.root)
    print(f"  표본 {len(samples)}건, 이미지 내려받는 중...")
    for index, sample in enumerate(samples, start=1):
        fetch(sample["filepath"], args.root / sample["filepath"])
        if index % 100 == 0:
            print(f"    {index}/{len(samples)}")

    checks = verify(samples)
    print()
    print(f"  이미지            {checks['images']}")
    print(f"  서로 다른 라벨    {checks['distinctLabels']}   <- 유효 표본 수")
    print(f"  잔반률 범위       {checks['wasteRateMin']} ~ {checks['wasteRateMax']}")
    print(f"  고잔반(>=0.8)     {checks['highWasteSamples']}")
    print(f"  bonid / dish      {checks['distinctBonIds']} / {checks['distinctDishes']}")
    print(f"  split             {checks['splits']}")

    registry = build_registry(args.root, samples, checks)
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    args.registry.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n레지스트리: {args.registry}")

    if checks["problems"]:
        print("\n받은 데이터가 기대와 다릅니다:")
        for problem in checks["problems"]:
            print(f"  - {problem}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
