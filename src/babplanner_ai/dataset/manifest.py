"""매니페스트 읽고 쓰기.

CSV를 쓴다. 사람이 열어 볼 수 있고, 변경 이력이 줄 단위로 남고, 새 의존성이
필요 없다.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pydantic import ValidationError

from babplanner_ai.dataset.schema import DatasetMetadata, DatasetSample

#: 이 이름을 가진 열은 매니페스트에 있을 수 없다. 값까지 뒤지지는 않는다.
#: 개인정보가 새는 가장 흔한 경로가 "일단 넣어 둔 열"이라 이름부터 막는다.
FORBIDDEN_COLUMNS = frozenset(
    {
        "name",
        "phone",
        "phone_number",
        "student_id",
        "email",
        "user_id",
        "access_token",
        "refresh_token",
        "verification_token",
        "password",
    }
)

#: 정답이 아닌 것을 정답처럼 적어 두는 열. confidence는 사람이 붙이는 값이
#: 아니라 모델의 불확실성에서 나와야 한다.
FORBIDDEN_TARGET_COLUMNS = frozenset({"confidence", "confidence_target"})


class ManifestError(ValueError):
    """매니페스트를 읽을 수 없다."""


class RowError(ManifestError):
    """행 하나가 잘못됐다."""

    def __init__(self, line: int, sample_id: str, message: str) -> None:
        super().__init__(f"{line}행({sample_id or '?'}): {message}")
        self.line = line
        self.sample_id = sample_id


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _coerce(row: dict[str, str]) -> dict[str, object]:
    """CSV의 빈 칸을 없음으로, 숫자 칸을 숫자로 바꾼다.

    CSV는 모든 값이 문자열이라 이 단계가 필요하다. 숫자로 바꾸지 못하면
    그대로 넘겨 스키마 쪽에서 어디가 틀렸는지 말하게 한다.
    """
    numeric = {
        "menu_id",
        "site_id",
        "served_edible_mass_g",
        "leftover_edible_mass_g",
        "waste_rate",
    }
    coerced: dict[str, object] = {}
    for key, raw in row.items():
        value = _blank_to_none(raw)
        if value is not None and key in numeric:
            try:
                coerced[key] = float(value) if "." in value or key in {
                    "served_edible_mass_g",
                    "leftover_edible_mass_g",
                    "waste_rate",
                } else int(value)
            except ValueError:
                coerced[key] = value
        else:
            coerced[key] = value
    return coerced


def check_columns(fieldnames: list[str]) -> None:
    """열 이름만 보고 거를 수 있는 것을 거른다."""
    lowered = {name.strip().lower() for name in fieldnames}

    forbidden = sorted(lowered & FORBIDDEN_COLUMNS)
    if forbidden:
        raise ManifestError(
            "개인정보로 보이는 열이 있습니다: " + ", ".join(forbidden)
        )

    targets = sorted(lowered & FORBIDDEN_TARGET_COLUMNS)
    if targets:
        raise ManifestError(
            "정답으로 쓸 수 없는 열이 있습니다: "
            + ", ".join(targets)
            + ". confidence는 사람이 붙이는 정답이 아니라 모델이 내는 값입니다."
        )


def read_manifest(path: Path) -> list[DatasetSample]:
    """매니페스트를 읽는다. 한 행이라도 어긋나면 어디인지 말하고 멈춘다."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ManifestError("매니페스트가 비어 있습니다.")
        check_columns(list(reader.fieldnames))

        samples: list[DatasetSample] = []
        for line, row in enumerate(reader, start=2):
            data = _coerce(row)
            try:
                samples.append(DatasetSample(**data))  # type: ignore[arg-type]
            except ValidationError as error:
                message = "; ".join(
                    detail.get("msg", "") for detail in error.errors()
                )
                raise RowError(line, str(row.get("sample_id", "")), message) from error

    if not samples:
        raise ManifestError("표본이 하나도 없습니다.")
    return samples


def write_manifest(path: Path, samples: list[DatasetSample]) -> None:
    """매니페스트를 쓴다. 열 순서를 고정해 변경 이력이 읽히게 한다."""
    fieldnames = list(DatasetSample.model_fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            dumped = sample.model_dump(mode="json")
            writer.writerow({key: "" if value is None else value for key, value in dumped.items()})


def read_metadata(path: Path) -> DatasetMetadata:
    with path.open(encoding="utf-8") as handle:
        return DatasetMetadata(**json.load(handle))
