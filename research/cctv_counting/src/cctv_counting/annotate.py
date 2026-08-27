"""프레임 위에 겹쳐 그리기.

시연에서 바로 트는 화면이다. 여기에 보이는 것이 곧 심사자가 이해하는 것이다.
그래서 상자만 그리지 않고, 어떤 규칙으로 세고 있는지가 같이 보이게 한다.
ROI와 선이 어디 있는지, 지금 몇 명인지, 지금까지 몇 명이 드나들었는지,
그리고 이 프레임을 믿을 만한지.

**신원에 해당하는 것은 그리지 않는다.** 얼굴을 확대하거나 흐리게 처리하는
기능도 두지 않는다. 흐림 처리는 원본에 얼굴이 있다는 전제 위에서만 의미가
있는데, 이 저장소는 애초에 얼굴을 다루지 않는다. 상자 옆의 번호는 그 영상
안에서만 유효한 임시 번호라는 것을 화면에서도 알 수 있게 `#`를 붙인다.
"""

from __future__ import annotations

from typing import Any

from cctv_counting.config import CctvConfig
from cctv_counting.pipeline import FrameResult
from cctv_counting.reliability import ReliabilityFlag


class AnnotationError(RuntimeError):
    """겹쳐 그릴 수 없다."""


def _cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise AnnotationError(
            "겹쳐 그리기에는 opencv가 필요합니다. pip install -e \".[cctv]\""
        ) from error
    return cv2


#: BGR. 신뢰 상태마다 색을 달리해서, 소리 없이 보고 있어도 구분되게 한다.
_COLOR_BOX = (60, 220, 60)
_COLOR_LINE = (0, 200, 255)
_COLOR_ROI = (200, 200, 200)
_COLOR_TEXT = (255, 255, 255)
_COLOR_WARN = (0, 80, 255)
#: 검출은 되었으나 아직 세지 않는 상자. 굵은 초록과 구분되어야 한다.
_COLOR_PENDING = (150, 150, 150)


def draw_frame(
    frame: Any,
    result: FrameResult,
    config: CctvConfig,
    *,
    width: int,
    height: int,
) -> Any:
    """겹쳐 그린 새 프레임을 돌려준다. 원본 배열은 건드리지 않는다."""
    cv2 = _cv2()
    canvas = frame.copy()

    roi = config.roi.to_rect()
    if roi is not None:
        box = roi.to_pixels(width, height)
        cv2.rectangle(
            canvas,
            (int(box.x1), int(box.y1)),
            (int(box.x2), int(box.y2)),
            _COLOR_ROI,
            1,
        )
        cv2.putText(
            canvas,
            "ROI",
            (int(box.x1) + 4, int(box.y1) + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _COLOR_ROI,
            1,
            cv2.LINE_AA,
        )

    line = config.line.to_line()
    if line is not None:
        start, end = line.to_pixels(width, height)
        cv2.line(
            canvas,
            (int(start.x), int(start.y)),
            (int(end.x), int(end.y)),
            _COLOR_LINE,
            2,
        )
        cv2.putText(
            canvas,
            f"IN: {config.line.in_direction.value}",
            (int(start.x) + 4, int(start.y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _COLOR_LINE,
            1,
            cv2.LINE_AA,
        )

    # **화면에 굵게 그린 상자의 개수와 CURRENT 숫자가 반드시 같아야 한다.**
    #
    # 예전에는 추적 결과가 비면 원시 detection으로 대신 그렸다. 그래서 영상
    # 시작 직후 min_hits를 채우는 동안 "상자 네 개 + CURRENT: 0"이 함께
    # 보였다. 실제 영상을 돌려 보고서야 발견했다. 시연에서 이것을 보면
    # 계수가 고장 났다고 읽는다. 실제로는 아직 세지 않기로 한 것뿐이다.
    #
    # 이제 세는 것과 아직 안 세는 것을 다르게 그린다.
    if config.tracker.enabled:
        counted = [(person.bbox, f"#{person.track_id}") for person in result.tracked]
        confirmed = [person.bbox for person in result.tracked]
        pending = [
            detection.bbox
            for detection in result.detections
            if detection.bbox not in confirmed
        ]
    else:
        counted = [(detection.bbox, "") for detection in result.detections]
        pending = []

    # 아직 세지 않는 상자는 얇은 회색 점선 느낌으로. 번호는 붙이지 않는다.
    for bbox in pending:
        cv2.rectangle(
            canvas,
            (int(bbox.x1), int(bbox.y1)),
            (int(bbox.x2), int(bbox.y2)),
            _COLOR_PENDING,
            1,
        )

    for bbox, label in counted:
        cv2.rectangle(
            canvas,
            (int(bbox.x1), int(bbox.y1)),
            (int(bbox.x2), int(bbox.y2)),
            _COLOR_BOX,
            2,
        )
        if label:
            cv2.putText(
                canvas,
                label,
                (int(bbox.x1), max(12, int(bbox.y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                _COLOR_BOX,
                1,
                cv2.LINE_AA,
            )

    observation = result.observation
    lines = [f"CURRENT: {observation.occupancy_count}"]
    if observation.entries_total is None:
        lines.append("ENTERED: N/A (detection only)")
        lines.append("EXITED:  N/A (detection only)")
    else:
        lines.append(f"ENTERED: {observation.entries_total}")
        lines.append(f"EXITED:  {observation.exits_total}")

    for offset, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (12, 28 + offset * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            _COLOR_TEXT,
            2,
            cv2.LINE_AA,
        )

    if observation.reliability_flag is not ReliabilityFlag.OK:
        cv2.putText(
            canvas,
            observation.reliability_flag.value.replace("_", " "),
            (12, 28 + len(lines) * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            _COLOR_WARN,
            2,
            cv2.LINE_AA,
        )

    return canvas
