"""관측을 얼마나 믿을 수 있는가.

숫자 하나만 내보내면 받는 쪽은 그것을 항상 같은 무게로 받는다. 역광이라
사람이 반쯤만 잡히는 프레임과, 잘 보이는 프레임이 같은 값으로 나가면 안
된다. 그래서 관측마다 신뢰 신호를 같이 낸다.

**confidence를 정확도로 읽지 마라.** detector가 낸 confidence 0.8은 "이 상자가
사람일 가능성에 대해 모델이 매긴 점수"이지 "사람 수를 80% 맞혔다"가 아니다.
둘은 다른 축이다. 실제로 이런 일이 일어난다.

- 사람이 겹쳐 있어 두 명을 한 상자로 잡았다. 그 상자의 confidence는 높다.
  숫자는 확신하지만 개수는 틀렸다.
- 사람이 아닌 것을 사람으로 잡았는데 confidence가 0.9다.
- 사람이 아예 안 잡혔다. 못 잡은 것에는 confidence 자체가 없다. **누락은
  이 신호로 절대 알 수 없다.**

그러니 여기 있는 flag는 "이 값이 틀렸다"가 아니라 "이 프레임은 조건이 나쁘니
자동으로 믿지 말고 사람이 확인할 후보로 두자"라는 뜻이다. 이 flag와 실제
계수 오차 사이의 관계는 정답이 붙은 데이터로 재 보기 전에는 모른다. 그것이
RQ3이고, 아직 측정하지 않았다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class ReliabilityError(ValueError):
    """신뢰 신호 설정이 성립하지 않는다."""


class ReliabilityFlag(str, Enum):
    """이 관측을 어떻게 다룰 것인가."""

    #: 조건이 나쁘다고 볼 이유가 없다.
    OK = "OK"

    #: 상자는 있는데 신뢰도가 낮다. 조명, 가림, 원거리에서 잘 나온다.
    LOW_CONFIDENCE = "LOW_CONFIDENCE"

    #: 아무도 없다. 이것은 낮은 신뢰도와 다르다. 빈 식당일 수도 있고
    #: detector가 전부 놓친 것일 수도 있는데, 그 둘을 여기서는 구분할 수
    #: 없다. 구분할 수 없다는 사실 자체를 값으로 남긴다.
    NO_DETECTION = "NO_DETECTION"

    #: 사람이 너무 많아 상자가 서로 겹친다. 이 구간의 계수는 대체로
    #: 과소 추정이다.
    CROWDED = "CROWDED"


@dataclass(frozen=True)
class ReliabilitySignal:
    """관측 하나에 붙는 신뢰 정보."""

    mean_detection_confidence: float | None
    min_detection_confidence: float | None
    active_track_count: int
    flag: ReliabilityFlag

    @property
    def is_low_reliability(self) -> bool:
        """downstream에서 보류 후보로 볼 것인가.

        NO_DETECTION을 낮은 신뢰로 묶는 이유는, 빈 화면과 전부 놓친 화면을
        구분할 수 없기 때문이다. 구분할 수 없으면 자동으로 믿지 않는다.
        """
        return self.flag is not ReliabilityFlag.OK


@dataclass(frozen=True)
class ReliabilityPolicy:
    """어떤 조건에서 낮은 신뢰로 볼 것인가.

    min_mean_confidence: 평균 신뢰도가 이 아래면 LOW_CONFIDENCE.
    crowded_track_count: 동시 track이 이 이상이면 CROWDED. None이면 끈다.

    기본값은 임의로 정한 것이고 근거가 되는 측정이 아직 없다. 정답 데이터가
    생기면 이 값들은 재조정 대상이다.
    """

    min_mean_confidence: float = 0.45
    crowded_track_count: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_mean_confidence <= 1.0:
            raise ReliabilityError(
                f"min_mean_confidence는 0~1이어야 합니다: {self.min_mean_confidence}"
            )
        if self.crowded_track_count is not None and self.crowded_track_count < 1:
            raise ReliabilityError(
                f"crowded_track_count는 1 이상이어야 합니다: {self.crowded_track_count}"
            )


def evaluate_reliability(
    confidences: list[float],
    active_track_count: int,
    policy: ReliabilityPolicy,
) -> ReliabilitySignal:
    """이 프레임의 신뢰 신호를 만든다.

    confidences가 비면 평균과 최소는 None이다. 0.0으로 채우지 않는다. 0.0은
    "신뢰도가 0이었다"로 읽히고, 그것은 "잴 것이 없었다"와 다르다. 나중에
    평균을 내면 그 차이가 그대로 지표를 왜곡한다.
    """
    if active_track_count < 0:
        raise ReliabilityError(
            f"active_track_count는 0 이상이어야 합니다: {active_track_count}"
        )
    for value in confidences:
        if math.isnan(value) or math.isinf(value):
            raise ReliabilityError(f"신뢰도가 유한하지 않습니다: {value}")
        if not 0.0 <= value <= 1.0:
            raise ReliabilityError(f"신뢰도는 0~1이어야 합니다: {value}")

    if not confidences:
        return ReliabilitySignal(
            mean_detection_confidence=None,
            min_detection_confidence=None,
            active_track_count=active_track_count,
            flag=ReliabilityFlag.NO_DETECTION,
        )

    mean_confidence = sum(confidences) / len(confidences)
    min_confidence = min(confidences)

    if (
        policy.crowded_track_count is not None
        and active_track_count >= policy.crowded_track_count
    ):
        flag = ReliabilityFlag.CROWDED
    elif mean_confidence < policy.min_mean_confidence:
        flag = ReliabilityFlag.LOW_CONFIDENCE
    else:
        flag = ReliabilityFlag.OK

    return ReliabilitySignal(
        mean_detection_confidence=mean_confidence,
        min_detection_confidence=min_confidence,
        active_track_count=active_track_count,
        flag=flag,
    )
