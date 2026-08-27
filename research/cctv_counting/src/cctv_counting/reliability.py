"""관측을 얼마나 믿을 수 있는가.

숫자 하나만 내보내면 받는 쪽은 그것을 항상 같은 무게로 받는다. 역광이라
사람이 반쯤만 잡히는 프레임과, 잘 보이는 프레임이 같은 값으로 나가면 안
된다. 그래서 관측마다 신뢰 신호를 같이 낸다.

**여기는 detector가 낸 신뢰도와 상자의 유무만 본다.** 장면이 어떤 장면인지는
보지 않는다. 사람이 많다, 어둡다, 가려져 있다 같은 것은 **scene condition**
이고 별개의 축이다. 두 축을 한 필드에 담으면 그 필드는 아무것도 뜻하지 않게
된다. 조건 태그는 평가 metadata 쪽(`groundtruth.py`의 condition 열)에서
다루고, 데이터셋이 제공했거나 사람이 직접 붙인 경우에만 쓴다.

**특히 "사람이 많다"는 여기 들어오지 않는다.** 이유가 셋이다.

1. 혼잡도와 신뢰도는 다른 개념이다. 붐빈다고 관측이 자동으로 못 믿을
   것이 되지는 않는다.
2. 백엔드에 이미 `CROWDED`라는 업무 등급이 있다. 같은 이름을 다른 뜻으로
   쓰면 반드시 섞인다.
3. 몇 명부터 붐비는 것인지는 급식소 좌석 수에 달렸고, 그 임계값은 아직
   정해진 적이 없다. 정해지지 않은 정책을 모델 출력에 박을 수 없다.

**이것은 보정된(calibrated) 불확실성 추정이 아니다.** heuristic 신호다. 이
신호가 실제 계수 오차와 상관이 있는지는 아직 재지 않았다. 그것이 RQ3이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class ReliabilityError(ValueError):
    """신뢰 신호 설정이 성립하지 않는다."""


class ReliabilityFlag(str, Enum):
    """detector 출력만 보고 매긴, 이 관측을 자동으로 믿을지에 대한 신호."""

    #: 상자가 있고 신뢰도가 낮다고 볼 이유가 없다.
    OK = "OK"

    #: 상자는 있는데 신뢰도가 낮다. 조명, 가림, 원거리에서 잘 나온다.
    LOW_CONFIDENCE = "LOW_CONFIDENCE"

    #: detector가 유효한 사람 상자를 하나도 내지 않았다.
    #:
    #: 이 값은 "식당이 비었다"가 아니고 "모델이 실패했다"도 아니다. 그 둘을
    #: 런타임에서 구분할 방법이 없다는 뜻일 뿐이다. 실제로 아무도 없었을
    #: 수도 있고, 있었는데 전부 놓쳤을 수도 있다. 구분할 수 없다는 사실
    #: 자체를 값으로 남긴다.
    NO_DETECTION = "NO_DETECTION"


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

        NO_DETECTION을 여기 묶는 이유는, 빈 화면과 전부 놓친 화면을 구분할
        수 없기 때문이다. 구분할 수 없으면 자동으로 믿지 않는다.
        """
        return self.flag is not ReliabilityFlag.OK


@dataclass(frozen=True)
class ReliabilityPolicy:
    """어떤 조건에서 낮은 신뢰로 볼 것인가.

    min_mean_confidence: 평균 신뢰도가 이 아래면 LOW_CONFIDENCE.

    기본값은 임의로 정한 것이고 근거가 되는 측정이 아직 없다. 정답 데이터가
    생기면 재조정 대상이다.
    """

    min_mean_confidence: float = 0.45

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_mean_confidence <= 1.0:
            raise ReliabilityError(
                f"min_mean_confidence는 0~1이어야 합니다: {self.min_mean_confidence}"
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

    **평균 신뢰도가 높아도 누락은 전혀 잡히지 않는다.** 실제로 10명이 있는데
    5명만 잡았고 그 5개의 신뢰도가 0.95라면, 평균 신뢰도는 높고 flag는 OK이며
    계수는 절반이다. 못 잡은 것에는 confidence 자체가 없기 때문이다. 이 함수의
    출력은 그런 상황에 대해 아무것도 말해 주지 않는다.
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
    flag = (
        ReliabilityFlag.LOW_CONFIDENCE
        if mean_confidence < policy.min_mean_confidence
        else ReliabilityFlag.OK
    )

    return ReliabilitySignal(
        mean_detection_confidence=mean_confidence,
        min_detection_confidence=min_confidence,
        active_track_count=active_track_count,
        flag=flag,
    )
