"""수집 기록 저장소.

기록은 덧붙이기만 한다. 잘못 적은 값을 고칠 때도 원래 줄을 지우지 않고 고친
줄을 새로 붙인다. 현장에서 숫자를 잘못 읽었다는 사실 자체가 나중에 알아야
할 정보이기 때문이다.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from babplanner_ai.collection.ids import is_safe_id
from babplanner_ai.collection.image_processing import atomic_write_text
from babplanner_ai.collection.models import (
    CollectionSession,
    EntryType,
    RawMeasurementRecord,
    TareProfile,
)


class JournalError(ValueError):
    """기록을 읽거나 쓸 수 없다."""


class CollectionStore:
    """한 수집 회차의 파일들.

    회차 하나가 디렉터리 하나다. 옮기거나 백업할 때 통째로 다루면 된다.
    """

    def __init__(self, root: Path, collection_session_id: str) -> None:
        if not is_safe_id(collection_session_id):
            # 경로 조각으로 쓰이므로 우리가 만든 모양인지 확인한다.
            raise JournalError(f"올바르지 않은 회차 식별자입니다: {collection_session_id}")
        self.root = root
        self.session_id = collection_session_id
        self.session_dir = root / "sessions" / collection_session_id

    @property
    def session_path(self) -> Path:
        return self.session_dir / "session.json"

    @property
    def tare_path(self) -> Path:
        return self.session_dir / "tare_profiles.jsonl"

    @property
    def journal_path(self) -> Path:
        return self.session_dir / "journal.jsonl"

    @property
    def raw_images_dir(self) -> Path:
        return self.session_dir / "raw_images"

    @property
    def sanitized_images_dir(self) -> Path:
        return self.session_dir / "sanitized_images"

    @property
    def exports_dir(self) -> Path:
        return self.session_dir / "exports"

    # -- 회차 --------------------------------------------------------------

    def create_session(self, session: CollectionSession) -> None:
        if self.session_path.exists():
            raise JournalError("이미 있는 회차입니다.")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.write_session(session)

    def write_session(self, session: CollectionSession) -> None:
        atomic_write_text(
            self.session_path,
            json.dumps(session.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n",
        )

    def read_session(self) -> CollectionSession:
        if not self.session_path.exists():
            raise JournalError(f"회차를 찾을 수 없습니다: {self.session_id}")
        try:
            return CollectionSession(**json.loads(self.session_path.read_text("utf-8")))
        except (ValidationError, json.JSONDecodeError) as error:
            raise JournalError(f"회차 정보를 읽을 수 없습니다: {error}") from error

    # -- tare --------------------------------------------------------------

    def append_tare(self, profile: TareProfile) -> None:
        _append(self.tare_path, profile.model_dump(mode="json"))

    def read_tare_profiles(self) -> dict[str, TareProfile]:
        profiles: dict[str, TareProfile] = {}
        for payload in _read(self.tare_path):
            profile = TareProfile(**payload)
            # 같은 id를 다시 적으면 나중 것이 이긴다. 다시 잰 경우다.
            profiles[profile.tare_profile_id] = profile
        return profiles

    # -- 표본 --------------------------------------------------------------

    def append_record(self, record: RawMeasurementRecord) -> None:
        _append(self.journal_path, record.model_dump(mode="json"))

    def read_records(self) -> list[RawMeasurementRecord]:
        """적힌 순서 그대로 전부. 고친 줄도 원래 줄도 다 들어 있다."""
        records: list[RawMeasurementRecord] = []
        for line, payload in enumerate(_read(self.journal_path), start=1):
            try:
                records.append(RawMeasurementRecord(**payload))
            except ValidationError as error:
                raise JournalError(f"{line}번째 기록을 읽을 수 없습니다: {error}") from error
        return records

    def latest_records(self) -> list[RawMeasurementRecord]:
        """표본마다 가장 마지막 고침만.

        내보낼 때 쓴다. 과거 값은 기록에 그대로 남는다.
        """
        latest: dict[str, RawMeasurementRecord] = {}
        for record in self.read_records():
            current = latest.get(record.sample_id)
            if current is None or record.revision >= current.revision:
                latest[record.sample_id] = record
        return [latest[key] for key in sorted(latest)]

    def next_revision(self, sample_id: str) -> int:
        revisions = [
            record.revision
            for record in self.read_records()
            if record.sample_id == sample_id
        ]
        return max(revisions) + 1 if revisions else 1

    def find_by_sanitized_hash(self, digest: str) -> list[str]:
        """같은 사본 해시를 가진 표본들."""
        return sorted(
            {
                record.sample_id
                for record in self.latest_records()
                if record.sanitized_image_sha256 == digest
            }
        )


def _append(path: Path, payload: dict[str, object]) -> None:
    """한 줄 덧붙인다.

    줄 단위로 붙이므로 중간에 멈춰도 앞의 기록은 온전하다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def _read(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payloads: list[dict[str, object]] = []
    for line, text in enumerate(path.read_text("utf-8").splitlines(), start=1):
        if not text.strip():
            continue
        try:
            payloads.append(json.loads(text))
        except json.JSONDecodeError as error:
            raise JournalError(f"{path.name} {line}번째 줄이 깨졌습니다.") from error
    return payloads


def correction_of(
    original: RawMeasurementRecord, updates: dict[str, object], *, now: str, note: str | None
) -> RawMeasurementRecord:
    """고친 기록을 만든다. 원래 기록은 그대로 둔다."""
    payload = original.model_dump()
    payload.update(updates)
    payload["revision"] = original.revision + 1
    payload["entry_type"] = EntryType.CORRECTION
    payload["correction_note"] = note
    payload["created_at"] = now
    return RawMeasurementRecord(**payload)
