"""평가 지표.

occupancy와 entry는 서로 다른 값이므로 지표도 따로 둔다. 둘을 하나의
"정확도"로 합치지 않는다.

**occupancy의 주 지표는 MAE다.** 단위가 사람 수 그대로라 "평균 1.8명 틀린다"로
읽힌다. RMSE는 큰 오차에 민감한 보조 지표로 같이 본다. bias(평균 부호 있는
오차)도 같이 내는데, MAE가 같아도 계속 적게 세는 것과 계속 많이 세는 것은
운영에서 완전히 다른 문제이기 때문이다.

**MAPE는 주 지표로 쓰지 않는다.** 식당은 비는 시간이 있고, 실제 인원이 0인
구간에서 MAPE는 정의되지 않거나 무한대로 튄다. 0을 빼고 계산하면 가장 쉬운
구간만 남는다. 어느 쪽이든 숫자가 실제보다 좋아 보인다.

**entry는 개수 오차로 본다.** 영상 하나에 대해 |예측 - 실제|이고, 영상이
여럿이면 그 평균이다. 사건 단위 annotation이 충분해지면 precision/recall/F1로
넘어갈 수 있지만, 그 전에 그 숫자를 쓰면 정답이 없는데 있는 것처럼 된다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


class MetricError(ValueError):
    """지표를 계산할 수 없다."""


def _checked_counts(
    actual: Sequence[float], predicted: Sequence[float]
) -> list[tuple[float, float]]:
    if len(actual) != len(predicted):
        raise MetricError(
            f"개수가 다릅니다: 정답 {len(actual)}건, 예측 {len(predicted)}건"
        )
    if not actual:
        # 빈 입력에 0을 돌려주면 완벽한 성능처럼 보인다.
        raise MetricError("입력이 비어 있습니다.")

    pairs: list[tuple[float, float]] = []
    for index, (truth, guess) in enumerate(zip(actual, predicted)):
        for name, value in (("정답", truth), ("예측", guess)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MetricError(f"{index}번째 {name}이 숫자가 아닙니다: {value!r}")
            if math.isnan(value) or math.isinf(value):
                raise MetricError(f"{index}번째 {name}이 유한하지 않습니다: {value}")
            if value < 0:
                raise MetricError(f"{index}번째 {name}이 음수입니다: {value}")
        pairs.append((float(truth), float(guess)))
    return pairs


def occupancy_mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """주 지표. 단위는 사람 수다."""
    pairs = _checked_counts(actual, predicted)
    return sum(abs(truth - guess) for truth, guess in pairs) / len(pairs)


def occupancy_rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """큰 오차에 더 민감한 보조 지표."""
    pairs = _checked_counts(actual, predicted)
    return math.sqrt(
        sum((truth - guess) ** 2 for truth, guess in pairs) / len(pairs)
    )


def occupancy_bias(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """평균 부호 있는 오차. 양수면 과대 추정이다."""
    pairs = _checked_counts(actual, predicted)
    return sum(guess - truth for truth, guess in pairs) / len(pairs)


@dataclass(frozen=True)
class OccupancyScores:
    mae: float
    rmse: float
    bias: float
    count: int

    def render(self) -> str:
        return (
            f"표본 {self.count}프레임\n"
            f"MAE   {self.mae:.3f}명\n"
            f"RMSE  {self.rmse:.3f}명\n"
            f"Bias  {self.bias:+.3f}명  (양수면 실제보다 많이 셈)"
        )


def score_occupancy(
    actual: Sequence[float], predicted: Sequence[float]
) -> OccupancyScores:
    return OccupancyScores(
        mae=occupancy_mae(actual, predicted),
        rmse=occupancy_rmse(actual, predicted),
        bias=occupancy_bias(actual, predicted),
        count=len(actual),
    )


def entry_count_error(actual: int, predicted: int) -> int:
    """영상 하나의 입장 인원 절대 오차."""
    for name, value in (("정답", actual), ("예측", predicted)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise MetricError(f"{name} 입장 인원이 정수가 아닙니다: {value!r}")
        if value < 0:
            raise MetricError(f"{name} 입장 인원이 음수입니다: {value}")
    return abs(predicted - actual)


def entry_count_mae(actual: Sequence[int], predicted: Sequence[int]) -> float:
    """여러 영상에 걸친 입장 인원 평균 절대 오차."""
    if len(actual) != len(predicted):
        raise MetricError(
            f"영상 수가 다릅니다: 정답 {len(actual)}개, 예측 {len(predicted)}개"
        )
    if not actual:
        raise MetricError("입력이 비어 있습니다.")
    errors = [
        entry_count_error(truth, guess) for truth, guess in zip(actual, predicted)
    ]
    return sum(errors) / len(errors)


@dataclass(frozen=True)
class ConditionScores:
    """조건 하나에 대한 결과 한 줄."""

    condition: str
    samples: int
    occupancy: OccupancyScores | None
    entry_error: float | None

    def render(self) -> str:
        occupancy_mae_text = (
            f"{self.occupancy.mae:.3f}" if self.occupancy is not None else "N/A"
        )
        entry_text = f"{self.entry_error:.2f}" if self.entry_error is not None else "N/A"
        return (
            f"| {self.condition} | {self.samples} | "
            f"{occupancy_mae_text} | {entry_text} |"
        )


def score_by_condition(
    samples: Sequence[tuple[str, float, float]],
) -> list[ConditionScores]:
    """(조건 이름, 실제 인원, 예측 인원) 목록을 조건별로 묶어 채점한다.

    조건 태그를 만들어 내지 않는다. 여기 들어오는 태그는 데이터셋이 제공한
    것이거나 사람이 직접 붙인 것이어야 한다. 없는 조건에 대해 이 함수는
    아무것도 돌려주지 않는다. 그것이 맞다. 조명 조건을 모르는데 "주간"이라고
    적으면 표는 채워지지만 그 표는 거짓이다.
    """
    grouped: dict[str, list[tuple[float, float]]] = {}
    for condition, truth, guess in samples:
        if not condition or not condition.strip():
            raise MetricError("조건 이름이 비어 있습니다.")
        grouped.setdefault(condition, []).append((truth, guess))

    results: list[ConditionScores] = []
    for condition in sorted(grouped):
        pairs = grouped[condition]
        actual = [truth for truth, _ in pairs]
        predicted = [guess for _, guess in pairs]
        results.append(
            ConditionScores(
                condition=condition,
                samples=len(pairs),
                occupancy=score_occupancy(actual, predicted),
                entry_error=None,
            )
        )
    return results


def render_condition_table(rows: Sequence[ConditionScores]) -> str:
    """보고서에 그대로 붙일 수 있는 표."""
    header = (
        "| Condition | Samples | Occupancy MAE | Entry Error |\n"
        "|---|---:|---:|---:|"
    )
    if not rows:
        return header + "\n| (조건 annotation 없음) | 0 | N/A | N/A |"
    return "\n".join([header, *(row.render() for row in rows)])
