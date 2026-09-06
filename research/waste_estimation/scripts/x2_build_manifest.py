"""X2 — 공개 잔반 데이터를 밥플래너 매니페스트 스키마로 옮긴다.

옮기는 것은 **질량 두 개와 그룹 키**뿐이다. 임베딩과 세그멘테이션은 기계가
만든 것이고 우리는 W2 backbone을 쓰므로 가져오지 않는다.

`weight_after` 는 **옮기지 않는다.** 375건 중 160건이 정확히 0이고(test
split 전체), 모든 `*_after` 필드가 `*_before`의 균일 배수라 측정값이 아니다.
쓰면 정답이 오염된다.

분할을 붙이지 않는다. 이 데이터는 전이 확인용이고 학습에 쓰지 않기로 했다.
전부 비어 있으면 검증기는 경고하지 않는다 — 일부만 붙어 있을 때만 말한다.
그러니 "분할이 없다"는 사실은 이 주석과 레지스트리가 지고 있다.
`make-splits` 를 이 매니페스트에 돌리지 않는다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\x2_build_manifest.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SERVICE = Path(__file__).resolve().parents[3] / "src"
if str(_SERVICE) not in sys.path:
    sys.path.insert(0, str(_SERVICE))

from babplanner_ai.dataset.manifest import write_manifest  # noqa: E402
from babplanner_ai.dataset.schema import (  # noqa: E402
    CURRENT_LABEL_VERSION,
    AnalysisLabel,
    DatasetSample,
    ServedMassSource,
)
from babplanner_ai.dataset.validation import sha256_of, validate_dataset  # noqa: E402


def menu_ids_of(samples: list[dict]) -> dict[str, int]:
    """메뉴 이름을 정수 id로. 정렬 순서를 쓰므로 다시 돌려도 같다.

    이 데이터에서 `dish`와 `bonid`는 1:1이다. 그래도 메뉴는 메뉴 이름에서
    만든다 — bonid를 메뉴 id로 쓰면 다른 날 같은 메뉴가 나왔을 때 다른
    메뉴가 되어 버린다.
    """
    return {name: index for index, name in enumerate(sorted({s["dish"] for s in samples}), start=1)}


def to_sample(raw: dict, root: Path, menu_ids: dict[str, int]) -> DatasetSample:
    served = float(sum(raw["weight_per_plate"]))
    leftover = float(sum(raw["return_quantity"]))
    relative = raw["filepath"]
    # 파일 이름을 그대로 쓴다. 개인 정보가 없고, 검증기가 sample_id와 파일
    # 이름이 같은지 보기 때문이다.
    sample_id = Path(relative).stem

    return DatasetSample(
        sample_id=sample_id,
        image_path=relative,
        menu_id=menu_ids[raw["dish"]],
        # 같은 영수증의 사진들은 같은 잔반을 다시 찍은 것이다. 참가자
        # 그룹으로 묶어야 분할이 새지 않는다.
        participant_group_id=f"bon_{raw['bonid']}",
        capture_session_id=f"bon_{raw['bonid']}",
        served_edible_mass_g=served,
        leftover_edible_mass_g=leftover,
        waste_rate=leftover / served,
        served_mass_source=ServedMassSource.MEASURED,
        analysis_label=AnalysisLabel.APPROVED,
        image_sha256=sha256_of(root / relative),
        # 산술식이 우리 정의와 같으므로 같은 버전을 쓴다.
        label_version=CURRENT_LABEL_VERSION,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("research/waste_estimation/data/external/food_waste"),
    )
    parser.add_argument("--out", type=Path, default=None, help="기본값은 root/manifest.csv")
    args = parser.parse_args(argv)

    out = args.out or (args.root / "manifest.csv")
    payload = json.loads((args.root / "samples.json").read_text(encoding="utf-8"))
    raws = payload["samples"] if isinstance(payload, dict) else payload

    menu_ids = menu_ids_of(raws)
    samples = [to_sample(raw, args.root, menu_ids) for raw in raws]
    write_manifest(out, samples)

    groups = {s.participant_group_id for s in samples}
    rates = sorted(s.waste_rate for s in samples if s.waste_rate is not None)
    print(f"표본 {len(samples)}건 -> {out}")
    print(f"  분할 그룹(bonid)   {len(groups)}")
    print(f"  메뉴               {len(menu_ids)}")
    print(f"  잔반률             {rates[0]:.4f} ~ {rates[-1]:.4f}")
    print(f"  고잔반(>=0.8)      {sum(1 for r in rates if r >= 0.8)}   <- 0이면 N/A로 보고한다")
    print(f"  분할               붙이지 않음 (전이 확인 전용)")

    report = validate_dataset(samples, args.root)
    print()
    if report.errors or report.warnings:
        print(report.render())
    print(f"\n오류 {len(report.errors)}건, 경고 {len(report.warnings)}건")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
