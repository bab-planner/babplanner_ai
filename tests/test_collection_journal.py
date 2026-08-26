"""기록 저장소와 내보내기."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from babplanner_ai.collection.exporter import ExportError, build_metadata, export
from babplanner_ai.collection.ids import (
    is_safe_id,
    is_safe_sample_id,
    new_capture_session_id,
    new_collection_session_id,
    new_participant_group_id,
    new_sample_id,
    new_tare_profile_id,
)
from babplanner_ai.collection.image_processing import ingest
from babplanner_ai.collection.journal import (
    CollectionStore,
    JournalError,
    correction_of,
)
from babplanner_ai.collection.models import (
    FORBIDDEN_FIELDS,
    CollectionSession,
    EntryType,
    RawMeasurementRecord,
    TareProfile,
    check_no_forbidden_fields,
)
from babplanner_ai.dataset.schema import (
    CURRENT_LABEL_VERSION,
    AnalysisLabel,
    DatasetSample,
    RejectionReason,
    ServedMassSource,
)
from babplanner_ai.dataset.validation import validate_dataset

NOW = "2026-08-27T04:00:00Z"
SESSION_ID = "cs_000000000001"
TARE_ID = "tp_000000000001"


@pytest.fixture
def store(tmp_path: Path) -> CollectionStore:
    store = CollectionStore(tmp_path, SESSION_ID)
    store.create_session(
        CollectionSession(
            collection_session_id=SESSION_ID,
            dataset_version="test-2026-08",
            label_version=CURRENT_LABEL_VERSION,
            protocol_version="pilot-v1",
            started_at=NOW,
            scale_resolution_g=1.0,
        )
    )
    store.append_tare(
        TareProfile(
            tare_profile_id=TARE_ID,
            description="4칸 식판",
            tare_measurements_g=[400.0],
            measured_at=NOW,
        )
    )
    return store


def add_image(store: CollectionStore, sample_id: str, colour: tuple[int, int, int]):
    source = store.session_dir / f"src-{sample_id}.png"
    source.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), colour).save(source, format="PNG")
    image = ingest(
        source,
        root=store.session_dir,
        raw_dir=store.raw_images_dir,
        sanitized_dir=store.sanitized_images_dir,
        sample_id=sample_id,
    )
    source.unlink()
    return image


def add_sample(
    store: CollectionStore,
    *,
    colour: tuple[int, int, int],
    post: float = 500.0,
    participant: str = "p_000000000001",
    capture: str = "cap_000000000001",
    menu_id: int = 3,
    **overrides: object,
) -> RawMeasurementRecord:
    sample_id = new_sample_id()
    image = add_image(store, sample_id, colour)
    values: dict[str, object] = {
        "sample_id": sample_id,
        "collection_session_id": SESSION_ID,
        "capture_session_id": capture,
        "participant_group_id": participant,
        "menu_id": menu_id,
        "tare_profile_id": TARE_ID,
        "pre_meal_total_weight_g": 900.0,
        "post_meal_total_weight_g": post,
        "served_mass_source": ServedMassSource.MEASURED,
        "analysis_label": AnalysisLabel.APPROVED,
        "raw_image_path": str(image.raw_path.relative_to(store.session_dir)).replace(
            "\\", "/"
        ),
        "sanitized_image_path": str(
            image.sanitized_path.relative_to(store.session_dir)
        ).replace("\\", "/"),
        "raw_image_sha256": image.raw_sha256,
        "sanitized_image_sha256": image.sanitized_sha256,
        "created_at": NOW,
    }
    values.update(overrides)
    record = RawMeasurementRecord(**values)  # type: ignore[arg-type]
    store.append_record(record)
    return record


# -- 식별자 ----------------------------------------------------------------


def test_generated_ids_are_opaque() -> None:
    for maker in (
        new_collection_session_id,
        new_participant_group_id,
        new_capture_session_id,
        new_tare_profile_id,
    ):
        value = maker()
        assert is_safe_id(value), value
    assert is_safe_sample_id(new_sample_id())


def test_generated_ids_are_unique() -> None:
    assert len({new_participant_group_id() for _ in range(200)}) == 200


def test_unsafe_session_id_is_refused(tmp_path: Path) -> None:
    # 경로 조각으로 쓰이므로 우리가 만든 모양만 받는다.
    with pytest.raises(JournalError, match="올바르지 않은"):
        CollectionStore(tmp_path, "../escaped")


# -- 기록 ------------------------------------------------------------------


def test_records_are_appended(store: CollectionStore) -> None:
    add_sample(store, colour=(10, 20, 30))
    add_sample(store, colour=(40, 50, 60))

    assert len(store.read_records()) == 2
    assert len(store.journal_path.read_text("utf-8").strip().splitlines()) == 2


def test_correction_keeps_the_original(store: CollectionStore) -> None:
    original = add_sample(store, colour=(10, 20, 30), post=500.0)

    corrected = correction_of(
        original, {"post_meal_total_weight_g": 450.0}, now=NOW, note="잘못 읽음"
    )
    store.append_record(corrected)

    all_records = store.read_records()
    assert len(all_records) == 2
    assert all_records[0].post_meal_total_weight_g == 500.0
    assert all_records[1].post_meal_total_weight_g == 450.0
    assert all_records[1].entry_type is EntryType.CORRECTION
    assert all_records[1].revision == 2


def test_latest_revision_wins(store: CollectionStore) -> None:
    original = add_sample(store, colour=(10, 20, 30), post=500.0)
    store.append_record(
        correction_of(original, {"post_meal_total_weight_g": 450.0}, now=NOW, note=None)
    )

    latest = store.latest_records()

    assert len(latest) == 1
    assert latest[0].post_meal_total_weight_g == 450.0


def test_next_revision_counts_up(store: CollectionStore) -> None:
    original = add_sample(store, colour=(10, 20, 30))

    assert store.next_revision(original.sample_id) == 2
    assert store.next_revision("unknown") == 1


def test_broken_journal_line_is_reported(store: CollectionStore) -> None:
    add_sample(store, colour=(10, 20, 30))
    with store.journal_path.open("a", encoding="utf-8") as handle:
        handle.write("{ not json\n")

    with pytest.raises(JournalError, match="깨졌습니다"):
        store.read_records()


def test_duplicate_hash_is_findable(store: CollectionStore) -> None:
    record = add_sample(store, colour=(10, 20, 30))

    found = store.find_by_sanitized_hash(record.sanitized_image_sha256 or "")

    assert found == [record.sample_id]


# -- 내보내기 ---------------------------------------------------------------


def test_export_reuses_the_dataset_schema(store: CollectionStore) -> None:
    add_sample(store, colour=(10, 20, 30))

    result = export(store)

    assert all(isinstance(sample, DatasetSample) for sample in result.samples)


def test_export_computes_the_rate(store: CollectionStore) -> None:
    add_sample(store, colour=(10, 20, 30), post=500.0)

    sample = export(store).samples[0]

    assert sample.served_edible_mass_g == 500.0
    assert sample.leftover_edible_mass_g == 100.0
    assert sample.waste_rate == pytest.approx(0.2)


def test_export_points_at_the_sanitized_copy(store: CollectionStore) -> None:
    add_sample(store, colour=(10, 20, 30))

    sample = export(store).samples[0]

    assert sample.image_path.startswith("sanitized_images/")
    assert "raw_images" not in sample.image_path


def test_export_passes_the_existing_validator(store: CollectionStore) -> None:
    add_sample(store, colour=(10, 20, 30))
    add_sample(store, colour=(40, 50, 60), participant="p_000000000002")

    result = export(store)
    report = validate_dataset(result.samples, store.session_dir)

    assert report.ok


def test_export_uses_the_latest_correction(store: CollectionStore) -> None:
    original = add_sample(store, colour=(10, 20, 30), post=500.0)
    store.append_record(
        correction_of(original, {"post_meal_total_weight_g": 450.0}, now=NOW, note=None)
    )

    sample = export(store).samples[0]

    assert sample.leftover_edible_mass_g == 50.0
    assert sample.waste_rate == pytest.approx(0.1)


def test_rejected_sample_carries_no_rate(store: CollectionStore) -> None:
    add_sample(
        store,
        colour=(10, 20, 30),
        analysis_label=AnalysisLabel.REJECTED,
        rejection_reason=RejectionReason.SEVERE_BLUR,
        served_mass_source=None,
    )

    sample = export(store).samples[0]

    assert sample.analysis_label is AnalysisLabel.REJECTED
    assert sample.waste_rate is None
    assert not sample.is_primary_benchmark


def test_rejected_sample_drops_measurements_it_may_have(store: CollectionStore) -> None:
    # 계량값이 있더라도 추정 불가로 판단한 사진에는 정답을 붙이지 않는다.
    add_sample(
        store,
        colour=(10, 20, 30),
        analysis_label=AnalysisLabel.REJECTED,
        rejection_reason=RejectionReason.INVALID_VIEW,
        served_mass_source=None,
        post_meal_total_weight_g=500.0,
    )

    sample = export(store).samples[0]

    assert sample.served_edible_mass_g is None
    assert sample.leftover_edible_mass_g is None


def test_invalid_measurement_is_skipped_and_named(store: CollectionStore) -> None:
    add_sample(store, colour=(10, 20, 30))
    broken = add_sample(store, colour=(40, 50, 60), participant="p_000000000002")
    store.append_record(
        correction_of(
            broken, {"post_meal_total_weight_g": 100.0}, now=NOW, note=None
        )
    )

    result = export(store)

    assert len(result.samples) == 1
    assert [sample_id for sample_id, _ in result.skipped] == [broken.sample_id]
    assert not result.ok


def test_export_writes_atomically(store: CollectionStore) -> None:
    add_sample(store, colour=(10, 20, 30))

    result = export(store)

    assert result.manifest_path is not None
    assert result.manifest_path.is_file()
    # 임시 파일이 남지 않는다.
    assert not list(store.exports_dir.glob("*.tmp"))


def test_metadata_comes_from_the_session(store: CollectionStore) -> None:
    session = store.read_session()

    metadata = build_metadata(session)

    assert metadata.dataset_version == session.dataset_version
    assert metadata.label_version == session.label_version
    assert metadata.scale_resolution_g == session.scale_resolution_g
    assert metadata.protocol_version == session.protocol_version


def test_metadata_is_written(store: CollectionStore) -> None:
    add_sample(store, colour=(10, 20, 30))

    result = export(store)

    assert result.metadata_path is not None
    payload = json.loads(result.metadata_path.read_text("utf-8"))
    assert payload["label_version"] == CURRENT_LABEL_VERSION


def test_export_with_nothing_is_refused(store: CollectionStore) -> None:
    with pytest.raises(ExportError, match="내보낼 표본이 없습니다"):
        export(store)


def test_invalid_manifest_is_not_finalised(store: CollectionStore) -> None:
    record = add_sample(store, colour=(10, 20, 30))
    # 사본이 사라진 상태를 만든다.
    (store.session_dir / (record.sanitized_image_path or "")).unlink()

    with pytest.raises(ExportError, match="검증을 통과하지 못해"):
        export(store)

    assert not (store.exports_dir / "manifest.csv").exists()


# -- 개인정보 ---------------------------------------------------------------


def test_records_have_no_personal_fields() -> None:
    for model in (CollectionSession, TareProfile, RawMeasurementRecord):
        check_no_forbidden_fields(model)


def test_participant_group_id_is_allowed() -> None:
    assert "participant_group_id" in RawMeasurementRecord.model_fields
    assert "participant_group_id" not in FORBIDDEN_FIELDS


def test_operator_identity_is_not_recorded() -> None:
    # 누가 입력했는지는 데이터셋에 필요하지 않다.
    assert "operator" not in RawMeasurementRecord.model_fields
    assert "operator_name" not in RawMeasurementRecord.model_fields


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(Exception):
        RawMeasurementRecord(
            sample_id="s1",
            collection_session_id=SESSION_ID,
            capture_session_id="cap_1",
            menu_id=3,
            analysis_label=AnalysisLabel.REJECTED,
            rejection_reason=RejectionReason.OTHER,
            created_at=NOW,
            student_id="20201234",
        )
