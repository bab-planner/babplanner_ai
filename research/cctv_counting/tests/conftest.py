"""연구 트랙 테스트 공용 도구.

여기 있는 detector는 가짜다. 모델 weight도 torch도 opencv도 없이 pipeline
전체를 그대로 돌리기 위한 것이다. 계수 로직의 정확성은 실제 모델이 무엇을
보는지와 무관하게 검증될 수 있어야 하고, 그러려면 detector가 무엇을 낼지
테스트가 정할 수 있어야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from cctv_counting.detection import COCO_PERSON_CLASS_ID, Detection
from cctv_counting.geometry import BoundingBox


@dataclass
class ScriptedDetector:
    """프레임마다 미리 정해 둔 상자를 낸다.

    detect()가 받는 frame 인자는 무시한다. 여기서는 프레임 내용이 아니라
    호출 순서가 곧 프레임 번호다.
    """

    script: list[list[Detection]]
    calls: int = field(default=0, init=False)

    def detect(self, frame: object) -> list[Detection]:
        if self.calls >= len(self.script):
            self.calls += 1
            return []
        detections = self.script[self.calls]
        self.calls += 1
        return list(detections)


def person(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 0.9,
    class_id: int = COCO_PERSON_CLASS_ID,
) -> Detection:
    """사람 상자 하나를 짧게 만든다."""
    return Detection(
        bbox=BoundingBox(x1, y1, x2, y2),
        confidence=confidence,
        class_id=class_id,
    )


def walker(
    frames: int,
    *,
    start_y: float,
    end_y: float,
    x: float = 500.0,
    width: float = 60.0,
    height: float = 160.0,
    confidence: float = 0.9,
) -> list[list[Detection]]:
    """한 사람이 y축을 따라 곧게 걸어가는 대본.

    상자 크기가 일정하므로 프레임 사이 IoU가 충분히 겹친다. 추적이 끊기지
    않아야 하는 상황을 만들 때 쓴다.
    """
    if frames < 2:
        raise ValueError("frames는 2 이상이어야 합니다.")
    script: list[list[Detection]] = []
    for index in range(frames):
        ratio = index / (frames - 1)
        bottom = start_y + (end_y - start_y) * ratio
        script.append(
            [
                person(
                    x - width / 2,
                    bottom - height,
                    x + width / 2,
                    bottom,
                    confidence=confidence,
                )
            ]
        )
    return script


@pytest.fixture
def scripted() -> type[ScriptedDetector]:
    return ScriptedDetector
