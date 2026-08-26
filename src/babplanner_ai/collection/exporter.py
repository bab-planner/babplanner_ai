"""원시 기록을 학습용 데이터셋으로 내보낸다.

계산은 여기서 한다. 원시 기록에는 저울 숫자만 있고, 정답은 그것을 계산한
결과다. 계산식이 바뀌면 같은 원시 기록에서 다시 내보내면 된다.

내보낸 결과는 dataset 쪽 검증기를 그대로 통과해야 한다. 통과하지 못한
매니페스트를 최종 산출물로 확정하지 않는다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from babplanner_ai.collection.image_processing import atomic_write_text
from babplanner_ai.collection.journal import CollectionStore
from babplanner_ai.collection.measurement import compute_mass
from babplanner_ai.collection.models import (
    CollectionSession,
    MeasurementError,
    RawMeasurementRecord,
)
from babplanner_ai.dataset.manifest import write_manifest
from babplanner_ai.dataset.schema import (
    AnalysisLabel,
    DatasetMetadata,
    DatasetSample,
)
from babplanner_ai.dataset.validation import ValidationReport, validate_dataset


class ExportError(ValueError):
    """내보낼 수 없다."""


@dataclass
class ExportResult:
    samples: list[DatasetSample]
    metadata: DatasetMetadata
    report: ValidationReport
    skipped: list[tuple[str, str]] = field(default_factory=list)
    manifest_path: Path | None = None
    metadata_path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.report.ok and not self.skipped


def to_sample(
    record: RawMeasurementRecord,
    session: CollectionSession,
    tare_profiles: dict[str, object],
) -> DatasetSample:
    """기록 하나를 정답 한 건으로."""
    if record.sanitized_image_path is None:
        raise MeasurementError("비식별 사본 경로가 없습니다.")

    common: dict[str, object] = {
        "sample_id": record.sample_id,
        # 매니페스트는 학습에 실제로 쓰는 사본을 가리킨다. 원본이 아니다.
        "image_path": record.sanitized_image_path,
        "menu_id": record.menu_id,
        "site_id": record.site_id,
        "service_date": record.service_date,
        "meal_type": record.meal_type,
        "captured_at": record.captured_at,
        "participant_group_id": record.participant_group_id,
        "capture_session_id": record.capture_session_id,
        "analysis_label": record.analysis_label,
        "image_sha256": record.sanitized_image_sha256,
        "label_version": session.label_version,
    }

    if record.analysis_label is not AnalysisLabel.APPROVED:
        # 추정할 수 없다고 판단한 사진에는 정답을 붙이지 않는다. 계량값이
        # 있더라도 마찬가지다.
        return DatasetSample(
            **common, rejection_reason=record.rejection_reason
        )  # type: ignore[arg-type]

    mass = compute_mass(record, tare_profiles.get(record.tare_profile_id))  # type: ignore[arg-type]
    return DatasetSample(
        **common,  # type: ignore[arg-type]
        served_edible_mass_g=mass.served_edible_mass_g,
        leftover_edible_mass_g=mass.leftover_edible_mass_g,
        waste_rate=mass.waste_rate,
        served_mass_source=record.served_mass_source,
    )


def build_metadata(session: CollectionSession) -> DatasetMetadata:
    """회차 정보에서 데이터셋 메타데이터를 만든다.

    버전 문자열을 여러 곳에 따로 적지 않는다. 출처는 회차 하나다.
    """
    return DatasetMetadata(
        dataset_version=session.dataset_version,
        label_version=session.label_version,
        created_at=session.started_at,
        collection_site=session.site_code,
        scale_resolution_g=session.scale_resolution_g,
        protocol_version=session.protocol_version,
        notes=session.notes,
    )


def export(store: CollectionStore, *, write: bool = True) -> ExportResult:
    """가장 최신 고침만 모아 매니페스트로 만든다."""
    session = store.read_session()
    tare_profiles = store.read_tare_profiles()

    samples: list[DatasetSample] = []
    skipped: list[tuple[str, str]] = []
    for record in store.latest_records():
        try:
            samples.append(to_sample(record, session, tare_profiles))  # type: ignore[arg-type]
        except (MeasurementError, ValueError) as error:
            # 한 건이 잘못됐다고 나머지를 못 내보내면 현장에서 쓸 수 없다.
            # 대신 무엇이 빠졌는지 분명히 남긴다.
            skipped.append((record.sample_id, str(error)))

    if not samples:
        raise ExportError("내보낼 표본이 없습니다.")

    metadata = build_metadata(session)
    report = validate_dataset(samples, store.session_dir)

    result = ExportResult(
        samples=samples, metadata=metadata, report=report, skipped=skipped
    )

    if not write:
        return result

    if not report.ok:
        # 검증을 통과하지 못한 매니페스트를 최종 산출물로 두지 않는다.
        raise ExportError(
            "검증을 통과하지 못해 내보내지 않았습니다.\n" + report.render()
        )

    store.exports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = store.exports_dir / "manifest.csv"
    metadata_path = store.exports_dir / "metadata.json"

    _atomic_manifest(manifest_path, samples)
    atomic_write_text(
        metadata_path,
        json.dumps(metadata.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
    )

    result.manifest_path = manifest_path
    result.metadata_path = metadata_path
    return result


def _atomic_manifest(path: Path, samples: list[DatasetSample]) -> None:
    """다 쓰고 나서 자리를 바꾼다. 도중에 멈춰도 반쪽 파일이 남지 않는다."""
    temporary = path.with_suffix(".csv.tmp")
    write_manifest(temporary, samples)
    os.replace(temporary, path)
