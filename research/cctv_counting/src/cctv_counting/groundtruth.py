"""정답 데이터의 모양.

성능 숫자는 정답이 있어야 나온다. 아직 정답이 없으므로, 먼저 정답이 어떤
모양이어야 하는지를 닫아 둔다. 나중에 누가 annotation을 붙일 때 열 이름을
새로 고민하지 않도록.

**정답도 두 종류로 나뉜다.** occupancy와 entry는 다른 값이므로 정답 파일도
다르다.

occupancy_ground_truth.csv
    특정 프레임/시각에 그 공간에 실제로 몇 명이 있었는가. 프레임 전부에
    붙일 필요는 없다. 일정 간격으로 뽑은 프레임만 세어도 MAE는 계산된다.

crossing_ground_truth.csv
    누가(어떤 사람인지가 아니라 몇 번째 사건인지) 언제 어느 방향으로 선을
    넘었는가. 이 파일에도 신원 열은 없다.

**condition 열은 비워 둘 수 있고, 비워 두는 것이 지어내는 것보다 낫다.**
영상을 보고 "주간/저조도" 정도는 사람이 붙일 수 있지만, 붙이지 않은 것을
붙인 것처럼 만들면 조건별 표 전체가 거짓이 된다.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

OCCUPANCY_GT_FIELDS: tuple[str, ...] = (
    "video_id",
    "frame_index",
    "timestamp_seconds",
    "actual_count",
    "condition",
    "annotator",
    "notes",
)

CROSSING_GT_FIELDS: tuple[str, ...] = (
    "video_id",
    "event_index",
    "frame_index",
    "timestamp_seconds",
    "direction",
    "condition",
    "annotator",
    "notes",
)


class GroundTruthError(ValueError):
    """정답 파일을 읽을 수 없다."""


class Direction(str, Enum):
    IN = "IN"
    OUT = "OUT"


@dataclass(frozen=True)
class OccupancyGroundTruth:
    video_id: str
    frame_index: int
    timestamp_seconds: float
    actual_count: int
    condition: str | None = None
    annotator: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.actual_count < 0:
            raise GroundTruthError(f"actual_count가 음수입니다: {self.actual_count}")
        if self.frame_index < 0:
            raise GroundTruthError(f"frame_index가 음수입니다: {self.frame_index}")


@dataclass(frozen=True)
class CrossingGroundTruth:
    video_id: str
    event_index: int
    frame_index: int
    timestamp_seconds: float
    direction: Direction
    condition: str | None = None
    annotator: str | None = None
    notes: str | None = None


def _optional(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _check_header(actual: list[str] | None, expected: tuple[str, ...], path: Path) -> None:
    if actual is None or tuple(actual) != expected:
        raise GroundTruthError(f"{path}의 열 구성이 다릅니다: {actual} != {list(expected)}")


def read_occupancy_ground_truth(path: Path) -> list[OccupancyGroundTruth]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _check_header(reader.fieldnames, OCCUPANCY_GT_FIELDS, path)
        rows = []
        for row in reader:
            rows.append(
                OccupancyGroundTruth(
                    video_id=row["video_id"],
                    frame_index=int(row["frame_index"]),
                    timestamp_seconds=float(row["timestamp_seconds"]),
                    actual_count=int(row["actual_count"]),
                    condition=_optional(row["condition"]),
                    annotator=_optional(row["annotator"]),
                    notes=_optional(row["notes"]),
                )
            )
        return rows


def read_crossing_ground_truth(path: Path) -> list[CrossingGroundTruth]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _check_header(reader.fieldnames, CROSSING_GT_FIELDS, path)
        rows = []
        for row in reader:
            rows.append(
                CrossingGroundTruth(
                    video_id=row["video_id"],
                    event_index=int(row["event_index"]),
                    frame_index=int(row["frame_index"]),
                    timestamp_seconds=float(row["timestamp_seconds"]),
                    direction=Direction(row["direction"]),
                    condition=_optional(row["condition"]),
                    annotator=_optional(row["annotator"]),
                    notes=_optional(row["notes"]),
                )
            )
        return rows


def write_template(path: Path, fields: tuple[str, ...]) -> None:
    """헤더만 있는 빈 정답 파일을 만든다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(fields)


def entries_of(rows: list[CrossingGroundTruth], video_id: str) -> int:
    """한 영상의 실제 입장 인원."""
    return sum(
        1
        for row in rows
        if row.video_id == video_id and row.direction is Direction.IN
    )
