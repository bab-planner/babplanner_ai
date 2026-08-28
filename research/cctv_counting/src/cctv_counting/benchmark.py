"""공개 benchmark 시퀀스에서 정답을 만든다.

지금 다루는 것은 CAVIAR(EC Funded CAVIAR project / IST 2001 37540,
CC BY-SA)의 CVML annotation이다.

**여기 있는 정답 생성기는 예측 코드와 독립이다.** `crossing.py`의
`LineCrossingCounter`를 재사용하지 않는다. 같은 코드로 정답과 예측을 모두
만들면 그 코드의 버그가 양쪽에서 똑같이 나타나 오차가 0으로 보인다. 그래서
여기서는 전혀 다른 방식으로 쓴다. 예측 쪽은 프레임을 하나씩 받는 상태
기계이고, 이쪽은 **완성된 궤적 전체를 한 번에 훑는** 오프라인 계산이다.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from cctv_counting.geometry import BoundingBox


class BenchmarkError(ValueError):
    """정답을 만들 수 없다."""


#: CAVIAR CVML의 role 중 사람으로 세는 것.
#:
#: 이 목록에 없는 role이 나오면 조용히 무시하지 않고 예외를 던진다. CAVIAR
#: 전체에는 가방 같은 비사람 객체(leftObject 계열)가 있는 시퀀스도 있는데,
#: 그것을 사람으로 세면 정답 자체가 틀린다. 모르는 것이 나오면 사람이 보고
#: 판단해야 한다.
PERSON_ROLES = frozenset({"walker", "browser"})

#: 상자가 실제로 그려져 있는 상태들. 세 값 모두 사람이 화면에 보인다.
#: appear는 등장 프레임, disappear는 퇴장 프레임이다.
VISIBLE_APPEARANCES = frozenset({"appear", "visible", "disappear"})


@dataclass(frozen=True)
class GroundTruthBox:
    """정답 상자 하나. track_id는 그 시퀀스 안에서만 유효하다."""

    track_id: int
    bbox: BoundingBox
    role: str


def parse_cvml(path: Path) -> dict[int, list[GroundTruthBox]]:
    """CVML annotation을 프레임별 사람 상자로 바꾼다.

    **`grouplist`는 읽지 않는다.** 그것은 사람 여럿을 묶은 집합이라 같이
    세면 이중 계수가 된다. 사람은 `objectlist/object`에만 있다.
    """
    root = ET.parse(path).getroot()
    frames: dict[int, list[GroundTruthBox]] = {}

    for frame in root.findall("frame"):
        number = int(frame.get("number", "-1"))
        if number < 0:
            raise BenchmarkError(f"{path.name}: frame number가 없습니다.")
        boxes: list[GroundTruthBox] = []

        for objectlist in frame.findall("objectlist"):
            for obj in objectlist.findall("object"):
                appearance = (obj.findtext("appearance") or "").strip()
                if appearance and appearance not in VISIBLE_APPEARANCES:
                    continue

                roles = {
                    (h.findtext("role") or "").strip() for h in obj.iter("hypothesis")
                }
                roles.discard("")
                unknown = roles - PERSON_ROLES
                if unknown:
                    raise BenchmarkError(
                        f"{path.name} frame {number}: 처음 보는 role {sorted(unknown)}. "
                        "사람인지 사람이 판단해야 합니다."
                    )
                if not roles:
                    continue

                box = obj.find("box")
                if box is None:
                    raise BenchmarkError(f"{path.name} frame {number}: box가 없습니다.")
                xc = float(box.get("xc"))  # type: ignore[arg-type]
                yc = float(box.get("yc"))  # type: ignore[arg-type]
                w = float(box.get("w"))  # type: ignore[arg-type]
                h = float(box.get("h"))  # type: ignore[arg-type]
                if w <= 0 or h <= 0:
                    continue
                boxes.append(
                    GroundTruthBox(
                        track_id=int(obj.get("id", "-1")),
                        bbox=BoundingBox(xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2),
                        role=sorted(roles)[0],
                    )
                )
        frames[number] = boxes

    if not frames:
        raise BenchmarkError(f"{path.name}: frame이 하나도 없습니다.")
    return frames


def occupancy_ground_truth(
    frames: dict[int, list[GroundTruthBox]],
    sample_every: int,
) -> dict[int, int]:
    """프레임별 사람 수. sample_every 간격으로만 뽑는다.

    표본 간격을 결과를 보고 바꾸지 않는다. 모든 시퀀스에 같은 값을 쓴다.
    """
    if sample_every < 1:
        raise BenchmarkError(f"sample_every는 1 이상이어야 합니다: {sample_every}")
    return {
        number: len(boxes)
        for number, boxes in sorted(frames.items())
        if number % sample_every == 0
    }


class CrossingDirection(str, Enum):
    """세로선 기준. 예측 쪽과 같은 규약을 쓴다."""

    #: 오른쪽에서 왼쪽으로. 예측 설정의 in_direction=POSITIVE와 같은 방향.
    IN = "IN"
    #: 왼쪽에서 오른쪽으로.
    OUT = "OUT"


@dataclass(frozen=True)
class GroundTruthCrossing:
    frame_index: int
    direction: CrossingDirection
    track_id: int


def crossing_ground_truth(
    frames: dict[int, list[GroundTruthBox]],
    *,
    frame_width: int,
    frame_height: int,
    line_x_normalized: float,
    min_confirm_distance: float,
) -> list[GroundTruthCrossing]:
    """정답 궤적에서 세로선 통과 사건을 만든다.

    **예측 코드와 다른 방식으로 계산한다.** 궤적을 통째로 모은 뒤, 각
    track의 발 x좌표 수열에서 확정된 쪽이 바뀌는 지점을 찾는다. 프레임 단위
    상태 기계도, 불감대와 이력의 분리도, 재계수 금지 구간도 쓰지 않는다.
    두 구현이 우연히 같은 실수를 하지 않게 하려는 것이다.

    확정 규칙만 예측과 맞춘다. 선에서 min_confirm_distance보다 가까운 표본은
    어느 쪽인지 판정하지 않고 건너뛴다. 이것이 없으면 선 위를 스치는 궤적이
    통과로 세인다.
    """
    if not 0.0 <= line_x_normalized <= 1.0:
        raise BenchmarkError(
            f"line_x_normalized는 0~1이어야 합니다: {line_x_normalized}"
        )
    if min_confirm_distance < 0:
        raise BenchmarkError(
            f"min_confirm_distance는 0 이상이어야 합니다: {min_confirm_distance}"
        )
    if frame_width <= 0 or frame_height <= 0:
        raise BenchmarkError(f"프레임 크기가 올바르지 않습니다: {frame_width}x{frame_height}")

    # 1) 궤적을 통째로 모은다. 예측 쪽은 이렇게 하지 못한다.
    trajectories: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for number, boxes in frames.items():
        for box in boxes:
            foot = box.bbox.bottom_center
            trajectories[box.track_id].append((number, foot.x / frame_width))

    # 2) track마다 확정된 쪽의 변화를 센다.
    events: list[GroundTruthCrossing] = []
    for track_id, samples in trajectories.items():
        samples.sort()
        confirmed: str | None = None
        for number, x in samples:
            offset = x - line_x_normalized
            if abs(offset) < min_confirm_distance:
                continue  # 선에 너무 가깝다. 판정하지 않는다.
            side = "left" if offset < 0 else "right"
            if confirmed is None:
                confirmed = side  # 처음 확정. 사건이 아니다.
                continue
            if side == confirmed:
                continue
            direction = (
                CrossingDirection.IN if side == "left" else CrossingDirection.OUT
            )
            events.append(GroundTruthCrossing(number, direction, track_id))
            confirmed = side

    events.sort(key=lambda e: (e.frame_index, e.track_id))
    return events
