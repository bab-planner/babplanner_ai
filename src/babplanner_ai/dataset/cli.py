"""데이터셋 도구.

    python -m babplanner_ai.dataset validate --manifest ... --root ...
    python -m babplanner_ai.dataset summarize --manifest ...
    python -m babplanner_ai.dataset make-splits --manifest ... --out ...

문제가 있으면 0이 아닌 값으로 끝난다. 자동화에서 조용히 지나가지 않게
하려는 것이다.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

from babplanner_ai.dataset.manifest import ManifestError, read_manifest, write_manifest
from babplanner_ai.dataset.schema import DatasetSample
from babplanner_ai.dataset.splitting import (
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    GROUPERS,
    SplitError,
    SplitRatios,
    assign_splits,
    split_counts,
)
from babplanner_ai.dataset.validation import group_key_of, validate_dataset

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _load(manifest: Path) -> list[DatasetSample]:
    return read_manifest(manifest)


def command_validate(args: argparse.Namespace) -> int:
    samples = _load(args.manifest)
    report = validate_dataset(samples, args.root)

    if report.errors or report.warnings:
        print(report.render())

    # 총계만 찍지 않는다. 뼈 메뉴가 많아 실제로 쓸 표본이 모자란 상황이
    # "표본 200건" 뒤에 숨지 않게 한다.
    usable = sum(1 for sample in samples if sample.is_supervised_usable)
    excluded = sum(1 for sample in samples if sample.is_gt_excluded)
    print(
        f"\n전체 표본 {len(samples)}건, 정답 사용 가능 {usable}건, "
        f"정답 제외 {excluded}건"
    )
    print(f"오류 {len(report.errors)}건, 경고 {len(report.warnings)}건")
    if not report.ok:
        return EXIT_FAILED
    print("문제 없음")
    return EXIT_OK


def command_summarize(args: argparse.Namespace) -> int:
    samples = _load(args.manifest)

    labels = Counter(sample.analysis_label.value for sample in samples)
    sources = Counter(
        sample.served_mass_source.value
        for sample in samples
        if sample.served_mass_source is not None
    )
    reasons = Counter(
        sample.rejection_reason.value
        for sample in samples
        if sample.rejection_reason is not None
    )
    # 제외 표본의 질량비는 그 식판의 잔반률을 뜻하지 않기로 한 값이다.
    # 분포에 섞으면 평균과 최대가 거짓말을 한다.
    rates = [
        sample.waste_rate
        for sample in samples
        if sample.is_supervised_usable and sample.waste_rate is not None
    ]

    print(f"표본            {len(samples)}")
    for label in sorted(labels):
        print(f"  {label:<12} {labels[label]}")

    if rates:
        print("\n잔반률 · 정답 사용 가능분")
        print(f"  평균          {statistics.fmean(rates):.4f}")
        print(f"  중앙값        {statistics.median(rates):.4f}")
        print(f"  최소          {min(rates):.4f}")
        print(f"  최대          {max(rates):.4f}")
    else:
        print("\n잔반률 · 정답 사용 가능분  없음")

    print("\n제공량 출처")
    for source in sorted(sources):
        print(f"  {source:<18} {sources[source]}")

    if reasons:
        print("\n거절 사유")
        for reason in sorted(reasons):
            print(f"  {reason:<18} {reasons[reason]}")

    excluded = [sample for sample in samples if sample.is_gt_excluded]
    usable = [sample for sample in samples if sample.is_supervised_usable]
    primary = [sample for sample in samples if sample.is_primary_benchmark]

    print("\n정답 사용 가능 여부")
    print(f"  전체 표본     {len(samples)}")
    print(f"  정답 사용 가능 {len(usable)}")
    print(f"  주 평가 대상   {len(primary)}")
    print(f"  정답 제외     {len(excluded)}")

    if excluded:
        by_reason = Counter(
            sample.gt_exclusion_reason.value  # type: ignore[union-attr]
            for sample in excluded
        )
        print("\n정답 제외 사유")
        for reason in sorted(by_reason):
            print(f"  {reason:<18} {by_reason[reason]}")

        by_menu = Counter(sample.menu_id for sample in excluded)
        print("\n메뉴별 정답 제외")
        for menu_id in sorted(by_menu):
            print(f"  menu {menu_id:<13} {by_menu[menu_id]}")

    # 관문을 채우는 수는 정답으로 쓸 수 있는 표본에서만 센다. 제외 표본이
    # 범주 수를 채워 주면 실제로 학습할 수 없는 데이터로 관문을 통과한다.
    print("\n정답 사용 가능 표본 기준")
    print(f"  메뉴          {len({sample.menu_id for sample in usable})}")
    print(f"  분할 그룹     {len({group_key_of(sample) for sample in usable})}")
    high_waste = sum(
        1 for sample in usable if sample.waste_rate is not None and sample.waste_rate >= 0.8
    )
    print(f"  고잔반(≥0.8)  {high_waste}")

    # 실제 식별값은 찍지 않는다. 몇 개인지만 센다.
    print(f"\n메뉴            {len({sample.menu_id for sample in samples})}")
    print(
        "참가자 그룹     "
        f"{len({sample.participant_group_id for sample in samples if sample.participant_group_id})}"
    )
    print(f"촬영 회차       {len({sample.capture_session_id for sample in samples})}")
    print(f"분할 그룹       {len({group_key_of(sample) for sample in samples})}")

    hashes = [sample.image_sha256 for sample in samples if sample.image_sha256]
    duplicates = sum(count - 1 for count in Counter(hashes).values() if count > 1)
    print(f"중복 이미지     {duplicates}")

    counts = split_counts(samples)
    print("\n분할")
    for name in sorted(counts):
        print(f"  {name:<12} {counts[name]}")
    return EXIT_OK


def command_make_splits(args: argparse.Namespace) -> int:
    samples = _load(args.manifest)
    ratios = SplitRatios(args.train_ratio, args.val_ratio, args.test_ratio)
    assigned = assign_splits(
        samples, ratios=ratios, seed=args.seed, group_by=args.group_by
    )

    write_manifest(args.out, assigned)

    counts = split_counts(assigned)
    print(f"seed {args.seed}, 그룹 기준 {args.group_by}")
    for name in sorted(counts):
        print(f"  {name:<12} {counts[name]}")
    print(f"\n{args.out} 에 썼습니다.")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m babplanner_ai.dataset",
        description="잔반 데이터셋 점검·요약·분할",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="매니페스트와 이미지를 점검한다")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--root", type=Path, required=True, help="이미지 루트")
    validate.set_defaults(handler=command_validate)

    summarize = subparsers.add_parser("summarize", help="구성을 요약한다")
    summarize.add_argument("--manifest", type=Path, required=True)
    summarize.set_defaults(handler=command_summarize)

    splits = subparsers.add_parser("make-splits", help="분할을 붙인다")
    splits.add_argument("--manifest", type=Path, required=True)
    splits.add_argument("--out", type=Path, required=True)
    splits.add_argument("--seed", type=int, default=DEFAULT_SEED)
    splits.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS[0])
    splits.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS[1])
    splits.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS[2])
    splits.add_argument(
        "--group-by",
        choices=sorted(GROUPERS),
        default="auto",
        help="auto=참가자 우선, 없으면 촬영 회차 / menu=처음 보는 메뉴 평가용",
    )
    splits.set_defaults(handler=command_make_splits)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ManifestError, SplitError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_FAILED
    except FileNotFoundError as error:
        print(f"ERROR: 파일을 찾을 수 없습니다: {error.filename}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
