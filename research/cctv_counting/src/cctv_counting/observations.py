"""프레임 단위 관측과 그 CSV 표현.

CSV 열은 여기서 한 번만 정한다. 열 이름이 코드 여러 군데에 흩어지면 조용히
어긋난다.

**개인정보 관점에서 이 스키마가 무엇을 하지 않는지가 중요하다.** 여기에는
track_id 열이 없다. 사람마다 한 줄씩 남기지도 않는다. 한 줄은 한 프레임의
집계값이다. track_id를 시간과 함께 파일로 남기는 순간 그것은 사람별 동선
기록이 되고, 익명 계수를 하겠다는 전제가 무너진다. track_id는 메모리 안에서
프레임을 잇는 데만 쓰고 디스크에 닿지 않는다.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from cctv_counting.reliability import ReliabilityFlag

#: CSV 열 순서. 바꾸면 기존 분석 스크립트가 깨진다.
OBSERVATION_FIELDS: tuple[str, ...] = (
    "frame_index",
    "timestamp_seconds",
    "occupancy_count",
    "entries_total",
    "exits_total",
    "mean_detection_confidence",
    "min_detection_confidence",
    "active_track_count",
    "reliability_flag",
)

#: 개인을 가리킬 수 있어 이 스키마에 들어와서는 안 되는 이름들. 테스트가
#: 이 목록으로 스키마를 검사한다.
FORBIDDEN_FIELD_TOKENS: tuple[str, ...] = (
    "track_id",
    "person_id",
    "user_id",
    "student_id",
    "name",
    "phone",
    "face",
    "embedding",
    "identity",
)


class ObservationError(ValueError):
    """관측값이 성립하지 않는다."""


def _finite_non_negative(name: str, value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        raise ObservationError(f"{name}이(가) 유한한 숫자가 아닙니다: {value}")
    if value < 0:
        raise ObservationError(f"{name}은(는) 0 이상이어야 합니다: {value}")
    return float(value)


@dataclass(frozen=True)
class FrameObservation:
    """프레임 하나의 관측 결과.

    entries_total / exits_total은 영상 시작부터의 **누적**이다. 이 프레임에서
    몇 명이 넘었는가가 아니다. 누적으로 두면 중간부터 읽어도 그 시점까지의
    이용 인원을 바로 알 수 있다.

    추적을 끄고 돌린 baseline에서는 입장을 셀 수 없다. 그때 두 값은 None이고
    CSV에는 빈 칸으로 나간다. 0으로 적으면 "아무도 안 들어왔다"로 읽히는데
    그것은 측정 결과가 아니라 측정하지 않았다는 뜻이다.
    """

    frame_index: int
    timestamp_seconds: float
    occupancy_count: int
    entries_total: int | None
    exits_total: int | None
    mean_detection_confidence: float | None
    min_detection_confidence: float | None
    active_track_count: int
    reliability_flag: ReliabilityFlag

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ObservationError(f"frame_index는 0 이상이어야 합니다: {self.frame_index}")
        _finite_non_negative("timestamp_seconds", self.timestamp_seconds)
        if self.occupancy_count < 0:
            raise ObservationError(
                f"occupancy_count는 0 이상이어야 합니다: {self.occupancy_count}"
            )
        if self.active_track_count < 0:
            raise ObservationError(
                f"active_track_count는 0 이상이어야 합니다: {self.active_track_count}"
            )
        for name in ("entries_total", "exits_total"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ObservationError(f"{name}은(는) 0 이상이어야 합니다: {value}")
        for name in ("mean_detection_confidence", "min_detection_confidence"):
            value = getattr(self, name)
            if value is None:
                continue
            if math.isnan(value) or math.isinf(value):
                raise ObservationError(f"{name}이(가) 유한하지 않습니다: {value}")
            if not 0.0 <= value <= 1.0:
                raise ObservationError(f"{name}은(는) 0~1이어야 합니다: {value}")

    def to_row(self) -> dict[str, str]:
        """CSV 한 줄. None은 빈 칸이다."""

        def optional(value: float | int | None, digits: int | None = None) -> str:
            if value is None:
                return ""
            if digits is None:
                return str(value)
            return f"{value:.{digits}f}"

        return {
            "frame_index": str(self.frame_index),
            "timestamp_seconds": f"{self.timestamp_seconds:.3f}",
            "occupancy_count": str(self.occupancy_count),
            "entries_total": optional(self.entries_total),
            "exits_total": optional(self.exits_total),
            "mean_detection_confidence": optional(self.mean_detection_confidence, 4),
            "min_detection_confidence": optional(self.min_detection_confidence, 4),
            "active_track_count": str(self.active_track_count),
            "reliability_flag": self.reliability_flag.value,
        }


def write_observations(path: Path, observations: list[FrameObservation]) -> None:
    """관측을 CSV로 쓴다. 관측이 없어도 헤더는 남긴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OBSERVATION_FIELDS))
        writer.writeheader()
        for observation in observations:
            writer.writerow(observation.to_row())


def read_observations(path: Path) -> list[FrameObservation]:
    """CSV를 다시 읽는다. 평가 스크립트가 쓴다."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != OBSERVATION_FIELDS:
            raise ObservationError(
                f"열 구성이 다릅니다: {reader.fieldnames} != {list(OBSERVATION_FIELDS)}"
            )

        def optional_int(text: str) -> int | None:
            return int(text) if text != "" else None

        def optional_float(text: str) -> float | None:
            return float(text) if text != "" else None

        rows: list[FrameObservation] = []
        for row in reader:
            rows.append(
                FrameObservation(
                    frame_index=int(row["frame_index"]),
                    timestamp_seconds=float(row["timestamp_seconds"]),
                    occupancy_count=int(row["occupancy_count"]),
                    entries_total=optional_int(row["entries_total"]),
                    exits_total=optional_int(row["exits_total"]),
                    mean_detection_confidence=optional_float(
                        row["mean_detection_confidence"]
                    ),
                    min_detection_confidence=optional_float(
                        row["min_detection_confidence"]
                    ),
                    active_track_count=int(row["active_track_count"]),
                    reliability_flag=ReliabilityFlag(row["reliability_flag"]),
                )
            )
        return rows
