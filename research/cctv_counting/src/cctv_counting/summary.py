"""영상 하나의 요약.

키 이름은 백엔드(Spring)와 맞추려고 camelCase다. 나중에 이 요약이 그대로
서비스 쪽으로 넘어갈 것을 염두에 둔다.

**NaN과 Infinity를 절대 내보내지 않는다.** 파이썬 json 모듈은 기본으로
`NaN`, `Infinity`를 그냥 써 버리는데, 그것은 JSON 표준이 아니다. Jackson을
쓰는 Spring 쪽에서 파싱이 터지거나, 더 나쁘게는 조용히 이상한 값으로
들어간다. 그래서 쓰기 직전에 한 번 훑고, 발견하면 예외를 던진다. 값이 없는
것은 null로 보낸다. null은 "모른다"이고 NaN은 "계산이 깨졌다"이다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cctv_counting.observations import FrameObservation


class SummaryError(ValueError):
    """요약을 만들 수 없다."""


@dataclass(frozen=True)
class VideoSummary:
    """영상 하나에 대한 baseline 결과."""

    video: str
    duration_seconds: float
    frames_processed: int
    peak_occupancy: int
    average_occupancy: float
    entries_total: int | None
    exits_total: int | None
    processing_fps: float | None
    low_reliability_frame_ratio: float
    #: 추적 없이 detection만 썼을 때의 순진한 선 통과 대리 지표. baseline
    #: 비교용이며 입장 인원이 아니다.
    naive_line_band_detections: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "durationSeconds": self.duration_seconds,
            "framesProcessed": self.frames_processed,
            "peakOccupancy": self.peak_occupancy,
            "averageOccupancy": self.average_occupancy,
            "entriesTotal": self.entries_total,
            "exitsTotal": self.exits_total,
            "processingFps": self.processing_fps,
            "lowReliabilityFrameRatio": self.low_reliability_frame_ratio,
            "naiveLineBandDetections": self.naive_line_band_detections,
        }


def reject_non_finite(payload: Any, path: str = "$") -> None:
    """NaN/Infinity가 있으면 어디에 있는지 알려주며 터진다."""
    if isinstance(payload, bool):
        return
    if isinstance(payload, float):
        if math.isnan(payload) or math.isinf(payload):
            raise SummaryError(f"{path}에 JSON으로 쓸 수 없는 값이 있습니다: {payload}")
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            reject_non_finite(value, f"{path}.{key}")
        return
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            reject_non_finite(value, f"{path}[{index}]")


def build_summary(
    *,
    video: str,
    observations: list[FrameObservation],
    duration_seconds: float,
    processing_fps: float | None,
    naive_line_band_detections: int | None = None,
) -> VideoSummary:
    """관측들로부터 요약을 만든다.

    관측이 하나도 없으면 만들지 않는다. 프레임을 한 장도 못 읽었는데
    "평균 0명, 최대 0명"이 나가면, 읽지 못한 것이 텅 빈 식당으로 보인다.
    """
    if not observations:
        raise SummaryError("관측이 없습니다. 영상을 한 프레임도 읽지 못했습니다.")
    if math.isnan(duration_seconds) or math.isinf(duration_seconds):
        raise SummaryError(f"durationSeconds가 유한하지 않습니다: {duration_seconds}")

    counts = [item.occupancy_count for item in observations]
    last = observations[-1]
    low = sum(1 for item in observations if item.reliability_flag.value != "OK")

    return VideoSummary(
        video=video,
        duration_seconds=float(duration_seconds),
        frames_processed=len(observations),
        peak_occupancy=max(counts),
        average_occupancy=sum(counts) / len(counts),
        entries_total=last.entries_total,
        exits_total=last.exits_total,
        processing_fps=processing_fps,
        low_reliability_frame_ratio=low / len(observations),
        naive_line_band_detections=naive_line_band_detections,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """JSON을 쓴다. 쓰기 전에 반드시 검사한다."""
    reject_non_finite(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
