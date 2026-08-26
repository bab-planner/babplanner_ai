"""평가 지표.

주 지표는 MAE다. 잔반률이 0~1이므로 MAE 0.08은 "평균적으로 잔반률을 8
퍼센트포인트만큼 틀린다"로 그대로 읽힌다. 이 직관이 유지되는 것이 중요해서
정규화하거나 다른 축척으로 바꾸지 않는다.

여기에 "잔반 20% 이하를 맞혔는가" 같은 분류 지표는 없다. 그런 경계는 아직
정해진 운영 규칙이 아니고, 지금 만들면 정해지지도 않은 정책이 평가 숫자로
굳어 버린다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


class MetricError(ValueError):
    """지표를 계산할 수 없다."""


def _checked(actual: Sequence[float], predicted: Sequence[float]) -> list[float]:
    if len(actual) != len(predicted):
        raise MetricError(
            f"개수가 다릅니다: 정답 {len(actual)}건, 예측 {len(predicted)}건"
        )
    if not actual:
        # 빈 입력에 0을 돌려주면 완벽한 성능처럼 보인다.
        raise MetricError("입력이 비어 있습니다.")

    errors: list[float] = []
    for index, (truth, guess) in enumerate(zip(actual, predicted)):
        for name, value in (("정답", truth), ("예측", guess)):
            if math.isnan(value):
                raise MetricError(f"{index}번째 {name}이 숫자가 아닙니다.")
            if not 0.0 <= value <= 1.0:
                raise MetricError(
                    f"{index}번째 {name}이 0~1 범위를 벗어났습니다: {value}"
                )
        errors.append(abs(truth - guess))
    return errors


def mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """주 지표."""
    errors = _checked(actual, predicted)
    return sum(errors) / len(errors)


def root_mean_squared_error(
    actual: Sequence[float], predicted: Sequence[float]
) -> float:
    """큰 오차에 더 민감한 보조 지표."""
    errors = _checked(actual, predicted)
    return math.sqrt(sum(error * error for error in errors) / len(errors))


def median_absolute_error(
    actual: Sequence[float], predicted: Sequence[float]
) -> float:
    """이상치에 덜 흔들리는 보조 지표."""
    errors = sorted(_checked(actual, predicted))
    middle = len(errors) // 2
    if len(errors) % 2:
        return errors[middle]
    return (errors[middle - 1] + errors[middle]) / 2


@dataclass(frozen=True)
class RegressionScores:
    mae: float
    rmse: float
    median_ae: float
    count: int

    def render(self) -> str:
        return (
            f"표본 {self.count}건\n"
            f"MAE        {self.mae:.4f}  (잔반률 {self.mae * 100:.2f}%p 평균 오차)\n"
            f"RMSE       {self.rmse:.4f}\n"
            f"Median AE  {self.median_ae:.4f}"
        )


def score(actual: Sequence[float], predicted: Sequence[float]) -> RegressionScores:
    return RegressionScores(
        mae=mean_absolute_error(actual, predicted),
        rmse=root_mean_squared_error(actual, predicted),
        median_ae=median_absolute_error(actual, predicted),
        count=len(actual),
    )
