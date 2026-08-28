"""이미지에서 특징을 뽑아 캐시한다.

**한 번만 뽑고 재사용한다.** 같은 이미지를 실험마다 다시 통과시키면 시간도
들고, 무엇보다 두 갈래(after-only / before+after)가 서로 다른 전처리를 거칠
위험이 생긴다. 한 번에 뽑아 두면 그 위험이 구조적으로 사라진다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w2_features.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SERVICE = Path(__file__).resolve().parents[3] / "src"
if str(_SERVICE) not in sys.path:
    sys.path.insert(0, str(_SERVICE))

from waste_estimation import lefood  # noqa: E402
from waste_estimation.features import (  # noqa: E402
    SIMPLE_FEATURE_NAMES,
    FrozenBackbone,
    simple_image_features,
)

RAW = Path("research/waste_estimation/data/raw/lefood_set")
CACHE = Path("research/waste_estimation/data/features")


def main(argv: list[str] | None = None) -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument(
        "--backbones", nargs="+", default=["mobilenet_v3_small", "resnet18"]
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    report = lefood.load(args.raw)
    records = report.records
    print(f"표본 {len(records)}건")

    row_ids = [record.row_id for record in records]
    before_paths = [args.raw / record.before_image for record in records]
    after_paths = [args.raw / record.after_image for record in records]
    targets = np.array(
        [record.external_waste_ratio for record in records], dtype=float
    )
    groups = np.array([record.group_id for record in records])
    names = np.array([record.food_name for record in records])

    args.cache.mkdir(parents=True, exist_ok=True)

    # --- B1 단순 통계 ---------------------------------------------------
    started = time.perf_counter()
    simple_before = np.array(
        [simple_image_features(path) for path in before_paths], dtype=float
    )
    simple_after = np.array(
        [simple_image_features(path) for path in after_paths], dtype=float
    )
    print(
        f"단순 특징  {simple_after.shape}  {time.perf_counter() - started:.1f}s"
    )
    np.savez_compressed(
        args.cache / "simple.npz",
        row_ids=np.array(row_ids),
        groups=groups,
        names=names,
        targets=targets,
        before=simple_before,
        after=simple_after,
        feature_names=np.array(SIMPLE_FEATURE_NAMES),
    )

    # --- B2 frozen backbone ---------------------------------------------
    for name in args.backbones:
        backbone = FrozenBackbone(name=name, device=args.device).load()
        started = time.perf_counter()
        before = backbone.embed(before_paths)
        after = backbone.embed(after_paths)
        elapsed = time.perf_counter() - started
        print(
            f"{name:22} {after.shape}  {elapsed:.1f}s  "
            f"({elapsed / (2 * len(records)) * 1000:.1f} ms/image)  "
            f"weights={backbone.weights_name}"
        )
        np.savez_compressed(
            args.cache / f"{name}.npz",
            row_ids=np.array(row_ids),
            groups=groups,
            names=names,
            targets=targets,
            before=before,
            after=after,
            weights=np.array([backbone.weights_name]),
        )

    print(f"\n캐시: {args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
