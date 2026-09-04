"""현장 수집 도구.

    python -m babplanner_ai.collection init-session --root ... --dataset-version ...
    python -m babplanner_ai.collection new-participant
    python -m babplanner_ai.collection new-capture-session
    python -m babplanner_ai.collection tare-add --root ... --session ... ...
    python -m babplanner_ai.collection add-sample --root ... --session ... ...
    python -m babplanner_ai.collection correct-sample --root ... --session ... ...
    python -m babplanner_ai.collection export --root ... --session ...
    python -m babplanner_ai.collection qc --root ... --session ...
    python -m babplanner_ai.collection close-session --root ... --session ...

빠진 값이 있으면 무엇이 빠졌는지 말하고 0이 아닌 값으로 끝난다. 현장에서
잘못 적힌 것을 나중에 발견하는 것보다 그 자리에서 막는 편이 싸다.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from babplanner_ai.collection.exporter import ExportError, export
from babplanner_ai.collection.ids import (
    new_capture_session_id,
    new_collection_session_id,
    new_participant_group_id,
    new_sample_id,
    new_tare_profile_id,
)
from babplanner_ai.collection.image_processing import (
    ImageIngestError,
    IngestedImage,
    ingest,
)
from babplanner_ai.collection.journal import (
    CollectionStore,
    JournalError,
    correction_of,
)
from babplanner_ai.collection.measurement import compute_mass
from babplanner_ai.collection.models import (
    CollectionSession,
    MeasurementError,
    MenuExclusion,
    RawMeasurementRecord,
    TareProfile,
)
from babplanner_ai.collection.qc import build_report
from babplanner_ai.dataset.schema import (
    CURRENT_LABEL_VERSION,
    AnalysisLabel,
    GtExclusionReason,
    RejectionReason,
    ServedMassSource,
    check_gt_exclusion,
)

EXIT_OK = 0
EXIT_FAILED = 1

#: 수집 절차의 버전. 정답의 산술식(`label_version`)과는 다른 축이다.
#: v2에서 비가식물·식기 구성 규칙과 메뉴 단위 정답 제외가 들어왔다.
CURRENT_PROTOCOL_VERSION = "pilot-v2"

#: `--reason` 에서 제외를 푸는 값. enum에 넣지 않는다. "해제"는 사유가
#: 아니라 사유가 없는 상태이고, 그걸 사유처럼 저장하면 매니페스트에 남는다.
CLEAR_EXCLUSION = "NONE"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _store(args: argparse.Namespace) -> CollectionStore:
    return CollectionStore(args.root, args.session)


def command_init_session(args: argparse.Namespace) -> int:
    session_id = new_collection_session_id()
    session = CollectionSession(
        collection_session_id=session_id,
        dataset_version=args.dataset_version,
        label_version=args.label_version,
        protocol_version=args.protocol_version,
        started_at=_now(),
        site_code=args.site_code,
        scale_resolution_g=args.scale_resolution_g,
        notes=args.notes,
    )
    store = CollectionStore(args.root, session_id)
    store.create_session(session)

    print(f"수집 회차를 만들었습니다: {session_id}")
    print(f"  경로            {store.session_dir}")
    print(f"  데이터셋 버전   {session.dataset_version}")
    print(f"  정답 정의 버전  {session.label_version}")
    print(f"  저울 눈금       {session.scale_resolution_g} g")
    return EXIT_OK


def command_menu_exclusion(args: argparse.Namespace) -> int:
    """메뉴 하나를 정답 데이터에서 뺄지 정한다.

    한 번 정하면 그 메뉴의 표본에 자동으로 붙는다. 표본마다 다시 묻지 않는다.
    """
    store = _store(args)
    session = store.read_session()
    if not session.is_open:
        raise JournalError("이미 닫힌 회차입니다.")

    clearing = args.reason == CLEAR_EXCLUSION
    reason = None if clearing else GtExclusionReason(args.reason)
    # 모델을 만들기 전에 본다. pydantic이 내는 말은 현장에서 읽기 어렵다.
    check_gt_exclusion(reason, args.note)

    exclusion = MenuExclusion(
        menu_id=args.menu_id,
        gt_exclusion_reason=reason,
        gt_exclusion_note=args.note,
        declared_at=_now(),
    )
    store.append_menu_exclusion(exclusion)

    if clearing:
        print(f"메뉴 {args.menu_id} 의 정답 제외를 해제했습니다.")
        print("  이후에 기록하는 표본부터 정답 데이터에 들어갑니다.")
    else:
        print(f"메뉴 {args.menu_id} 를 정답 데이터에서 제외합니다.")
        print(f"  사유       {exclusion.gt_exclusion_reason.value}")  # type: ignore[union-attr]
        if exclusion.gt_exclusion_note:
            print(f"  설명       {exclusion.gt_exclusion_note}")
        print("  이후에 기록하는 이 메뉴의 표본은 학습·검증·평가에서 빠집니다.")

    # 이미 기록한 표본은 바뀌지 않는다. 조용히 소급하면 어제 내보낸
    # 매니페스트와 오늘 내보낸 것이 달라지고 이유를 알 수 없다.
    already = sum(
        1 for record in store.latest_records() if record.menu_id == args.menu_id
    )
    if already:
        print(
            f"\n이 메뉴로 이미 기록한 표본이 {already}건 있습니다. "
            "소급되지 않습니다."
        )
        print("  바꾸려면 correct-sample 로 표본마다 고치세요.")
    return EXIT_OK


def command_new_participant(args: argparse.Namespace) -> int:
    # 이름이나 학번을 받지 않는다. 이 값과 사람을 잇는 표가 필요하면
    # 운영자가 이 저장소 밖에서 따로 관리한다.
    print(new_participant_group_id())
    return EXIT_OK


def command_new_capture_session(args: argparse.Namespace) -> int:
    print(new_capture_session_id())
    return EXIT_OK


def command_tare_add(args: argparse.Namespace) -> int:
    store = _store(args)
    store.read_session()

    profile = TareProfile(
        tare_profile_id=new_tare_profile_id(),
        description=args.description,
        tare_measurements_g=args.weight,
        measured_at=_now(),
        notes=args.notes,
    )
    store.append_tare(profile)

    print(f"tare 프로파일을 만들었습니다: {profile.tare_profile_id}")
    print(f"  설명       {profile.description}")
    print(f"  측정 {profile.measurement_count}회  대표 {profile.tare_weight_g:.4g} g")
    if profile.measurement_count > 1:
        print(f"  편차       {profile.spread_g:.4g} g")
    return EXIT_OK


def command_add_sample(args: argparse.Namespace) -> int:
    store = _store(args)
    session = store.read_session()
    if not session.is_open:
        raise JournalError("이미 닫힌 회차입니다.")

    sample_id = new_sample_id()
    image = ingest(
        Path(args.image),
        root=store.session_dir,
        raw_dir=store.raw_images_dir,
        sanitized_dir=store.sanitized_images_dir,
        sample_id=sample_id,
    )

    try:
        return _record_sample(args, store, session, sample_id, image)
    except Exception:
        # 기록으로 남지 못한 이미지를 수집 폴더에 남겨 두지 않는다. 나중에
        # 어느 표본 것인지 알 수 없는 파일이 쌓인다.
        image.raw_path.unlink(missing_ok=True)
        image.sanitized_path.unlink(missing_ok=True)
        raise


def _record_sample(
    args: argparse.Namespace,
    store: CollectionStore,
    session: CollectionSession,
    sample_id: str,
    image: IngestedImage,
) -> int:
    existing = store.find_by_sanitized_hash(image.sanitized_sha256)
    if existing:
        # 자동으로 지우지 않는다. 어떤 표본과 겹치는지 알려 주고 사람이
        # 판단하게 한다.
        raise ImageIngestError("같은 이미지가 이미 있습니다: " + ", ".join(existing))

    label = AnalysisLabel(args.label)
    # 정답 제외는 메뉴에 걸린 선언에서 그대로 옮겨 적는다. 수집자는 표본마다
    # 다시 판단하지 않는다.
    exclusion = store.active_menu_exclusion(args.menu_id)
    record = RawMeasurementRecord(
        sample_id=sample_id,
        collection_session_id=session.collection_session_id,
        capture_session_id=args.capture_session,
        participant_group_id=args.participant,
        menu_id=args.menu_id,
        site_id=args.site_id,
        service_date=args.service_date,
        meal_type=args.meal_type,
        tare_profile_id=args.tare_profile,
        pre_meal_total_weight_g=args.pre_weight,
        post_meal_total_weight_g=args.post_weight,
        pre_meal_measurements_g=args.pre_repeat or [],
        post_meal_measurements_g=args.post_repeat or [],
        standard_served_edible_mass_g=args.standard_served,
        captured_at=args.captured_at,
        served_mass_source=(
            ServedMassSource(args.mass_source) if args.mass_source else None
        ),
        gt_exclusion_reason=exclusion.gt_exclusion_reason if exclusion else None,
        gt_exclusion_note=exclusion.gt_exclusion_note if exclusion else None,
        raw_image_path=str(image.raw_path.relative_to(store.session_dir)).replace(
            "\\", "/"
        ),
        sanitized_image_path=str(
            image.sanitized_path.relative_to(store.session_dir)
        ).replace("\\", "/"),
        raw_image_sha256=image.raw_sha256,
        sanitized_image_sha256=image.sanitized_sha256,
        analysis_label=label,
        rejection_reason=(
            RejectionReason(args.rejection_reason) if args.rejection_reason else None
        ),
        created_at=_now(),
    )

    if label is AnalysisLabel.APPROVED:
        # 계산해 보고 앞뒤가 맞는지 그 자리에서 확인한다. 나중에 내보낼 때
        # 알게 되면 그 사람은 이미 밥을 다 먹고 갔다.
        tare = store.read_tare_profiles().get(record.tare_profile_id or "")
        mass = compute_mass(record, tare)
        print(f"표본을 기록했습니다: {sample_id}")
        print(f"  제공량   {mass.served_edible_mass_g:.4g} g")
        print(f"  잔반량   {mass.leftover_edible_mass_g:.4g} g")
        print(f"  잔반률   {mass.waste_rate:.4f}")
    else:
        print(f"표본을 기록했습니다: {sample_id}")
        print(f"  판정     {label.value} ({record.rejection_reason})")

    if exclusion is not None:
        # 조용히 붙이지 않는다. 이 표본이 학습에 안 들어간다는 것을 그 자리에서
        # 알아야, 하루 종일 모으고 나서 쓸 수 있는 표본이 없다는 것을 늦게
        # 발견하는 일이 없다.
        print(
            f"  정답 제외 {exclusion.gt_exclusion_reason.value}"  # type: ignore[union-attr]
            f" (메뉴 {args.menu_id} 선언)"
        )
        print("           학습·검증·평가와 관문 계산에서 빠집니다.")

    store.append_record(record)
    print(f"  이미지   {record.sanitized_image_path}")
    return EXIT_OK


def command_correct_sample(args: argparse.Namespace) -> int:
    store = _store(args)
    latest = {record.sample_id: record for record in store.latest_records()}
    original = latest.get(args.sample_id)
    if original is None:
        raise JournalError(f"표본을 찾을 수 없습니다: {args.sample_id}")

    updates: dict[str, object] = {}
    if args.pre_weight is not None:
        updates["pre_meal_total_weight_g"] = args.pre_weight
    if args.post_weight is not None:
        updates["post_meal_total_weight_g"] = args.post_weight
    if args.tare_profile is not None:
        updates["tare_profile_id"] = args.tare_profile
    if args.label is not None:
        updates["analysis_label"] = AnalysisLabel(args.label)
    if args.rejection_reason is not None:
        updates["rejection_reason"] = RejectionReason(args.rejection_reason)
    if args.gt_exclusion_reason is not None:
        # 표본 하나만 메뉴 선언과 다르게 두어야 할 때 쓴다. 평소에는
        # menu-exclusion 으로 메뉴에 걸고 여기는 건드리지 않는다.
        clearing = args.gt_exclusion_reason == CLEAR_EXCLUSION
        reason = None if clearing else GtExclusionReason(args.gt_exclusion_reason)
        note = None if clearing else args.gt_exclusion_note
        # 모델을 만들기 전에 본다. pydantic이 내는 말은 현장에서 읽기 어렵다.
        check_gt_exclusion(reason, note)
        updates["gt_exclusion_reason"] = reason
        updates["gt_exclusion_note"] = note
    elif args.gt_exclusion_note is not None:
        updates["gt_exclusion_note"] = args.gt_exclusion_note
    if not updates:
        raise JournalError("고칠 값이 없습니다.")

    corrected = correction_of(original, updates, now=_now(), note=args.note)
    if corrected.analysis_label is AnalysisLabel.APPROVED:
        tare = store.read_tare_profiles().get(corrected.tare_profile_id or "")
        mass = compute_mass(corrected, tare)
        print(f"잔반률 {mass.waste_rate:.4f}")
    if "gt_exclusion_reason" in updates:
        reason = corrected.gt_exclusion_reason
        print(
            f"정답 제외 {reason.value}" if reason else "정답 제외를 해제했습니다."
        )

    store.append_record(corrected)
    # 원래 기록은 지우지 않는다. 무엇이 어떻게 바뀌었는지가 남아야 한다.
    print(f"{args.sample_id} 을(를) 고쳤습니다. 개정 {corrected.revision}")
    return EXIT_OK


def command_export(args: argparse.Namespace) -> int:
    store = _store(args)
    result = export(store)

    print(f"표본 {len(result.samples)}건을 내보냈습니다.")
    print(f"  매니페스트  {result.manifest_path}")
    print(f"  메타데이터  {result.metadata_path}")
    if result.report.warnings:
        print("\n" + "\n".join(f.render() for f in result.report.warnings))
    if result.skipped:
        print("\n내보내지 못한 표본:")
        for sample_id, message in result.skipped:
            print(f"  {sample_id}  {message}")
        return EXIT_FAILED
    return EXIT_OK


def command_qc(args: argparse.Namespace) -> int:
    print(build_report(_store(args)).render())
    return EXIT_OK


def command_close_session(args: argparse.Namespace) -> int:
    store = _store(args)
    session = store.read_session()
    if not session.is_open:
        print(f"이미 닫혀 있습니다: {session.closed_at}")
        return EXIT_OK

    store.write_session(session.model_copy(update={"closed_at": _now()}))
    print(f"회차를 닫았습니다: {session.collection_session_id}")
    return EXIT_OK


def _add_session_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, required=True, help="수집 루트")
    parser.add_argument("--session", required=True, help="수집 회차 id")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m babplanner_ai.collection",
        description="잔반 파일럿 현장 수집",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init-session", help="수집 회차를 만든다")
    init.add_argument("--root", type=Path, required=True)
    init.add_argument("--dataset-version", required=True)
    init.add_argument("--label-version", default=CURRENT_LABEL_VERSION)
    init.add_argument("--protocol-version", default=CURRENT_PROTOCOL_VERSION)
    init.add_argument("--site-code", default=None, help="급식소 짧은 코드")
    init.add_argument("--scale-resolution-g", type=float, required=True)
    init.add_argument("--notes", default=None)
    init.set_defaults(handler=command_init_session)

    exclusion = subparsers.add_parser(
        "menu-exclusion",
        help="메뉴 하나를 정답 데이터에서 뺀다 (표본마다 고르지 않는다)",
    )
    _add_session_args(exclusion)
    exclusion.add_argument("--menu-id", type=int, required=True)
    exclusion.add_argument(
        "--reason",
        required=True,
        choices=[reason.value for reason in GtExclusionReason] + [CLEAR_EXCLUSION],
        help=f"{CLEAR_EXCLUSION} 을 주면 제외를 해제한다",
    )
    exclusion.add_argument(
        "--note", default=None, help="왜 빼는지. OTHER 에는 반드시 필요하다"
    )
    exclusion.set_defaults(handler=command_menu_exclusion)

    participant = subparsers.add_parser(
        "new-participant", help="비식별 참가자 id를 만든다"
    )
    participant.set_defaults(handler=command_new_participant)

    capture = subparsers.add_parser(
        "new-capture-session", help="촬영 회차 id를 만든다"
    )
    capture.set_defaults(handler=command_new_capture_session)

    tare = subparsers.add_parser("tare-add", help="식기 빈 무게를 등록한다")
    _add_session_args(tare)
    tare.add_argument("--description", required=True, help="식기 구성")
    tare.add_argument(
        "--weight", type=float, action="append", required=True, help="그램. 여러 번 줄 수 있다"
    )
    tare.add_argument("--notes", default=None)
    tare.set_defaults(handler=command_tare_add)

    sample = subparsers.add_parser("add-sample", help="표본을 기록한다")
    _add_session_args(sample)
    sample.add_argument("--image", required=True, help="촬영한 원본 파일")
    sample.add_argument("--capture-session", required=True)
    sample.add_argument("--participant", default=None)
    sample.add_argument("--menu-id", type=int, required=True)
    sample.add_argument("--site-id", type=int, default=None)
    sample.add_argument("--service-date", default=None)
    sample.add_argument(
        "--meal-type", choices=["BREAKFAST", "LUNCH", "DINNER"], default=None
    )
    sample.add_argument("--tare-profile", default=None)
    sample.add_argument("--pre-weight", type=float, default=None, help="배식 후 전체 중량(g)")
    sample.add_argument("--post-weight", type=float, default=None, help="식사 후 전체 중량(g)")
    sample.add_argument("--pre-repeat", type=float, action="append", default=None)
    sample.add_argument("--post-repeat", type=float, action="append", default=None)
    sample.add_argument("--standard-served", type=float, default=None)
    sample.add_argument("--captured-at", default=None)
    sample.add_argument(
        "--mass-source",
        choices=["MEASURED", "STANDARD_PORTION"],
        default=None,
        help="눈대중 추정치는 정답 경로로 받지 않는다",
    )
    sample.add_argument(
        "--label", choices=["APPROVED", "REJECTED"], default="APPROVED"
    )
    sample.add_argument(
        "--rejection-reason",
        choices=[reason.value for reason in RejectionReason],
        default=None,
    )
    sample.set_defaults(handler=command_add_sample)

    correct = subparsers.add_parser("correct-sample", help="잘못 적은 값을 고친다")
    _add_session_args(correct)
    correct.add_argument("--sample-id", required=True)
    correct.add_argument("--pre-weight", type=float, default=None)
    correct.add_argument("--post-weight", type=float, default=None)
    correct.add_argument("--tare-profile", default=None)
    correct.add_argument("--label", choices=["APPROVED", "REJECTED"], default=None)
    correct.add_argument(
        "--rejection-reason",
        choices=[reason.value for reason in RejectionReason],
        default=None,
    )
    correct.add_argument(
        "--gt-exclusion-reason",
        choices=[reason.value for reason in GtExclusionReason] + [CLEAR_EXCLUSION],
        default=None,
        help=(
            "표본 하나만 메뉴 선언과 다르게 둘 때. "
            f"{CLEAR_EXCLUSION} 을 주면 제외를 해제한다"
        ),
    )
    correct.add_argument("--gt-exclusion-note", default=None)
    correct.add_argument("--note", default=None, help="왜 고쳤는지")
    correct.set_defaults(handler=command_correct_sample)

    exporter = subparsers.add_parser("export", help="학습용 매니페스트를 만든다")
    _add_session_args(exporter)
    exporter.set_defaults(handler=command_export)

    quality = subparsers.add_parser("qc", help="수집 상태를 본다")
    _add_session_args(quality)
    quality.set_defaults(handler=command_qc)

    close = subparsers.add_parser("close-session", help="회차를 닫는다")
    _add_session_args(close)
    close.set_defaults(handler=command_close_session)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        JournalError,
        MeasurementError,
        ImageIngestError,
        ExportError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
