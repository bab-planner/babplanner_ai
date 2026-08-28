"""시간적 확인만 하는 occupancy baseline.

**왜 이것이 필요한가.**

A2에서 detection-only(MAE 2.13)보다 추적(1.70)이 나았다. 그런데 추적의
`min_hits`를 1로 낮추자 결과가 detection-only와 정확히 같아졌다. 즉 그
개선이 어디서 왔는지 알 수 없다. 후보가 둘이다.

    (a) 시간적 연결(association)  — 같은 사람을 프레임 사이로 잇는 것
    (b) 시간적 확인(confirmation) — 여러 프레임에 걸쳐 반복해서 보인
                                    상자만 세는 것

추적은 이 둘을 동시에 한다. 그래서 추적만으로는 분해할 수 없다.

여기 있는 것은 **(b)만 하는 baseline**이다. (a)를 뺐다. 이것이 추적과
비슷하게 나오면 개선의 상당 부분은 확인에서 온 것이고, 추적이 일관되게
더 좋으면 연결 자체의 추가 이득이 있다는 뜻이다.

**무엇을 하지 않는가.** 이 점이 이 baseline의 정의다.

    안 함   장기 track 신원 (id를 부여하지 않는다)
    안 함   re-identification
    안 함   선 통과 계수 (입장/퇴장을 셀 수 없다)
    안 함   운동 예측, 가림 복구, 나이 든 track 살려두기

**무엇을 하는가.** 직전 프레임의 상자와 IoU가 겹치면 "같은 관측이 이어졌다"고
보고 연속 횟수를 하나 올린다. 연속 횟수가 min_observations 이상인 상자만
센다. 그것뿐이다.

직전 한 프레임만 본다. 궤적을 누적하지 않고, 한 프레임이라도 끊기면 연속
횟수가 0으로 돌아간다. 이 최소한의 프레임 간 대응은 "여러 프레임에 걸쳐
반복되었다"를 판정하는 데 필요한 것이지 추적이 아니다. **이것을 tracking
baseline이라고 부르지 않는다.**
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cctv_counting.detection import Detection
from cctv_counting.geometry import BoundingBox


class ConfirmationError(ValueError):
    """설정이 성립하지 않는다."""


@dataclass
class _Observed:
    """직전 프레임에서 살아남은 상자 하나.

    번호가 없다. 이 객체는 프레임이 넘어가면 새로 만들어지고, 밖으로 나가지
    않는다. 사람을 가리키지 않는다.
    """

    bbox: BoundingBox
    streak: int


@dataclass
class TemporalConfirmationCounter:
    """직전 프레임과의 겹침만으로 연속 관측을 세는 occupancy baseline.

    min_observations: 이만큼 연속으로 보인 상자만 센다. 1이면 확인 없이
                      프레임별 상자 수와 같아진다(= detection only).
    iou_threshold:    직전 프레임의 어떤 상자와 이만큼 겹치면 같은 관측이
                      이어진 것으로 본다.
    """

    min_observations: int = 3
    iou_threshold: float = 0.3

    _previous: list[_Observed] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.min_observations < 1:
            raise ConfirmationError(
                f"min_observations는 1 이상이어야 합니다: {self.min_observations}"
            )
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ConfirmationError(
                f"iou_threshold는 0~1이어야 합니다: {self.iou_threshold}"
            )

    def reset(self) -> None:
        """영상이 바뀌면 부른다."""
        self._previous.clear()

    def update(self, detections: list[Detection]) -> list[Detection]:
        """한 프레임을 넣고, 확인된 상자들만 받는다.

        추적처럼 보이지 않도록 돌려주는 것도 Detection 그대로다. 번호가 없다.
        """
        current: list[_Observed] = []
        used: set[int] = set()

        for detection in detections:
            # 직전 프레임에서 가장 많이 겹치는 상자를 하나 찾는다.
            best_index = -1
            best_iou = self.iou_threshold
            for index, previous in enumerate(self._previous):
                if index in used:
                    continue
                iou = detection.bbox.iou(previous.bbox)
                if iou >= best_iou:
                    best_iou = iou
                    best_index = index

            if best_index >= 0:
                used.add(best_index)
                streak = self._previous[best_index].streak + 1
            else:
                # 이어지지 않았다. 연속 횟수는 1부터 다시 시작한다.
                streak = 1
            current.append(_Observed(bbox=detection.bbox, streak=streak))

        self._previous = current
        return [
            detection
            for detection, observed in zip(detections, current)
            if observed.streak >= self.min_observations
        ]

    @property
    def tracked_state_count(self) -> int:
        """직전 프레임에서 들고 있는 상자 수. 상태가 새는지 보는 용도다."""
        return len(self._previous)
