"""운영 혼잡도 단계 정책.

**이것은 AI 모델이 아니다.** 연구가 정의한 출력은 `peopleCount`까지다.
"40명이 붐비는가"는 그 식당의 좌석 수와 회전율이 정하는 것이고, 모델이
알 수 있는 값이 아니다. 그래서 인원수를 단계로 바꾸는 규칙을 여기로
완전히 분리한다.

    연구 출력       peopleCount            (A1~A6에서 측정하고 검증한 것)
    운영 정책       peopleCount -> level   (여기. 측정된 적 없다)

**임계값을 코드에 박지 않는다.** `if count < 10: COMFORTABLE` 같은 줄이
소스에 들어가는 순간, 그 숫자가 어디서 왔는지 아무도 설명할 수 없게 된다.
값은 site별 설정 파일에서만 온다.

**보정되지 않은 값임을 값 자체가 들고 다닌다.** 지금 저장소에 있는 예시
임계값은 실제 급식소에서 측정한 것이 아니다. 그 사실을 주석이 아니라
`calibration` 필드로 남겨서, 나중에 이 숫자를 인용하는 사람이 출처를 볼 수
있게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CongestionPolicyError(ValueError):
    """혼잡도 정책이 성립하지 않는다."""


class CongestionLevel(str, Enum):
    """백엔드 `CongestionLevel`과 문자열이 정확히 같아야 한다.

    Flutter가 이 문자열을 그대로 읽는다. 하나라도 어긋나면 앱이 단계를
    표시하지 못한다.
    """

    COMFORTABLE = "COMFORTABLE"
    NORMAL = "NORMAL"
    CROWDED = "CROWDED"
    VERY_CROWDED = "VERY_CROWDED"


class Calibration(str, Enum):
    """이 임계값이 어디서 왔는가."""

    #: 시연을 위해 사람이 임의로 고른 값. 측정된 적 없다.
    DEMO_ONLY_UNCALIBRATED = "DEMO_ONLY_UNCALIBRATED"

    #: 해당 식당의 좌석 수/회전율로 실제 정한 값.
    SITE_CALIBRATED = "SITE_CALIBRATED"


@dataclass(frozen=True)
class CountThresholdPolicy:
    """인원수 구간으로 단계를 정한다.

        count <= comfortable_max   -> COMFORTABLE
        count <= normal_max        -> NORMAL
        count <= crowded_max       -> CROWDED
        그 위                       -> VERY_CROWDED

    경계값은 아래 단계에 포함된다. `comfortable_max = 5`면 5명은 여유다.
    이 규약을 바꾸면 같은 설정 파일이 다른 뜻이 되므로 테스트로 고정한다.
    """

    comfortable_max: int
    normal_max: int
    crowded_max: int
    calibration: Calibration

    def __post_init__(self) -> None:
        values = (self.comfortable_max, self.normal_max, self.crowded_max)
        names = ("comfortable_max", "normal_max", "crowded_max")
        for name, value in zip(names, values):
            if isinstance(value, bool) or not isinstance(value, int):
                raise CongestionPolicyError(f"{name}은(는) 정수여야 합니다: {value!r}")
            if value < 0:
                raise CongestionPolicyError(f"{name}은(는) 0 이상이어야 합니다: {value}")
        if not self.comfortable_max < self.normal_max < self.crowded_max:
            raise CongestionPolicyError(
                "임계값은 오름차순이어야 합니다: "
                f"comfortable_max({self.comfortable_max}) < "
                f"normal_max({self.normal_max}) < "
                f"crowded_max({self.crowded_max})"
            )

    @property
    def is_calibrated(self) -> bool:
        return self.calibration is Calibration.SITE_CALIBRATED

    def level_for(self, people_count: int) -> CongestionLevel:
        """인원수를 단계로 바꾼다."""
        if isinstance(people_count, bool) or not isinstance(people_count, int):
            raise CongestionPolicyError(
                f"people_count는 정수여야 합니다: {people_count!r}"
            )
        if people_count < 0:
            raise CongestionPolicyError(
                f"people_count는 0 이상이어야 합니다: {people_count}"
            )
        if people_count <= self.comfortable_max:
            return CongestionLevel.COMFORTABLE
        if people_count <= self.normal_max:
            return CongestionLevel.NORMAL
        if people_count <= self.crowded_max:
            return CongestionLevel.CROWDED
        return CongestionLevel.VERY_CROWDED

    def describe(self) -> dict[str, object]:
        """기록에 남길 형태. 이 값이 어디서 왔는지가 같이 나간다."""
        return {
            "type": "count_thresholds",
            "comfortableMax": self.comfortable_max,
            "normalMax": self.normal_max,
            "crowdedMax": self.crowded_max,
            "calibration": self.calibration.value,
        }
