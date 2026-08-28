"""연구용 지표.

**서비스 지표를 다시 구현하지 않는다.** MAE / RMSE / Median AE는 이미
`src/babplanner_ai/dataset/metrics.py`에 있고 평가 규약
(`docs/waste_evaluation_protocol.md`)이 MAE를 주 지표로 못박아 두었다.
같은 것을 여기 한 번 더 쓰면 두 구현이 갈라지는 날이 온다. 그래서 그것들은
그대로 가져다 쓰고, 여기에는 **연구에서 새로 필요해진 것만** 둔다.

    가져다 쓰는 것   MAE, RMSE, Median AE   (**정답이 0~1일 때**)
    여기서 더하는 것 bias, ordinal 지표, 범위 제한 없는 회귀 지표

**왜 범위 제한 없는 판이 따로 필요한가.** 서비스 지표는 값이 0~1을 벗어나면
거부한다. 그것은 옳다 — 잔반률 라벨이 1.2라면 측정이나 기록이 틀린 것이다.
그런데 **회귀 모델의 raw 출력은 0~1을 벗어나는 것이 정상이다.** 그 값을
지표 계산 전에 잘라 버리면 성능이 좋아 보이고, 자르지 않으면 서비스 지표가
계산을 거부한다. 그래서 예측 쪽에는 제한 없는 판을 쓰고, 자르는 것은
**정책**으로 따로 보고한다(`baselines.clip`).

**bias를 왜 더하는가.** MAE는 부호를 지운다. 모델이 늘 적게 예측하는 것과
늘 많게 예측하는 것은 MAE가 같아도 운영에서 다른 문제다. CCTV 트랙에서
집계 bias가 0에 가깝다는 이유로 정확하다고 읽을 뻔한 일이 있었다. 같은
실수를 반복하지 않으려고 처음부터 같이 낸다.

**구간 경계를 여기에 박지 않는다.** "잔반 20% 이하" 같은 경계는 정해진
운영 규칙이 아니다. `docs/waste_evaluation_protocol.md`가 그런 지표를 만들지
않기로 한 이유가 그것이다. 그래서 구간을 쓰는 함수는 **경계를 인자로
요구한다.** 기본값을 두면 아무도 고르지 않은 정책이 지표에 굳는다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from babplanner_ai.dataset.metrics import (
    MetricError,
    mean_absolute_error,
    median_absolute_error,
    root_mean_squared_error,
)

__all__ = [
    "regression_scores",
    "unbounded_bias",
    "unbounded_mae",
    "unbounded_median_ae",
    "unbounded_rmse",
    "MetricError",
    "bias",
    "class_index_mae",
    "confusion_matrix",
    "macro_f1",
    "mean_absolute_error",
    "median_absolute_error",
    "quadratic_weighted_kappa",
    "root_mean_squared_error",
    "to_class_index",
]


def _checked_pair(
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
        for label, value in (("정답", truth), ("예측", guess)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MetricError(f"{index}번째 {label}이 숫자가 아닙니다: {value!r}")
            if math.isnan(value) or math.isinf(value):
                raise MetricError(f"{index}번째 {label}이 유한하지 않습니다: {value}")
        pairs.append((float(truth), float(guess)))
    return pairs


def bias(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """평균 부호 있는 오차. 양수면 잔반률을 과대 추정한다.

    MAE와 반드시 같이 읽는다. bias가 0에 가깝다는 것은 치우침이 없다는 뜻이
    아니라 큰 과대와 큰 과소가 서로를 지웠다는 뜻일 수도 있다.
    """
    pairs = _checked_pair(actual, predicted)
    return sum(guess - truth for truth, guess in pairs) / len(pairs)


def to_class_index(value: float, edges: Sequence[float]) -> int:
    """연속 잔반률을 구간 번호로 바꾼다.

    edges는 **오름차순 내부 경계**다. `[0.2, 0.5]`면 구간이 셋이다.

        value <= 0.2        -> 0
        0.2 < value <= 0.5  -> 1
        0.5 < value         -> 2

    경계값은 아래 구간에 들어간다. 이 규약을 바꾸면 같은 경계 목록이 다른
    뜻이 되므로 테스트로 고정한다.

    **기본 경계를 두지 않는다.** 부르는 쪽이 어떤 경계를 왜 골랐는지 함께
    기록해야 한다.
    """
    if not edges:
        raise MetricError("구간 경계가 비어 있습니다.")
    previous: float | None = None
    for edge in edges:
        if isinstance(edge, bool) or not isinstance(edge, (int, float)):
            raise MetricError(f"경계가 숫자가 아닙니다: {edge!r}")
        if previous is not None and edge <= previous:
            raise MetricError(f"경계는 오름차순이어야 합니다: {list(edges)}")
        previous = float(edge)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricError(f"값이 숫자가 아닙니다: {value!r}")

    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)


def class_index_mae(
    actual: Sequence[int], predicted: Sequence[int]
) -> float:
    """구간 번호의 평균 절대 오차.

    **순서형에서는 정확도만 보면 안 된다.** 세 칸 떨어진 오답과 한 칸
    떨어진 오답은 다른 크기의 실수인데 정확도는 둘을 같게 센다. 잔반 구간은
    순서가 있으므로 거리를 보는 지표가 필요하다.
    """
    if len(actual) != len(predicted):
        raise MetricError(
            f"개수가 다릅니다: 정답 {len(actual)}건, 예측 {len(predicted)}건"
        )
    if not actual:
        raise MetricError("입력이 비어 있습니다.")
    total = 0
    for index, (truth, guess) in enumerate(zip(actual, predicted)):
        for label, value in (("정답", truth), ("예측", guess)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise MetricError(
                    f"{index}번째 {label} 구간 번호가 정수가 아닙니다: {value!r}"
                )
            if value < 0:
                raise MetricError(
                    f"{index}번째 {label} 구간 번호가 음수입니다: {value}"
                )
        total += abs(truth - guess)
    return total / len(actual)


def _checked_classes(
    actual: Sequence[int], predicted: Sequence[int]
) -> list[tuple[int, int]]:
    if len(actual) != len(predicted):
        raise MetricError(
            f"개수가 다릅니다: 정답 {len(actual)}건, 예측 {len(predicted)}건"
        )
    if not actual:
        raise MetricError("입력이 비어 있습니다.")
    pairs = []
    for index, (truth, guess) in enumerate(zip(actual, predicted)):
        for label, value in (("정답", truth), ("예측", guess)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise MetricError(
                    f"{index}번째 {label} 구간 번호가 정수가 아닙니다: {value!r}"
                )
            if value < 0:
                raise MetricError(f"{index}번째 {label}이 음수입니다: {value}")
        pairs.append((int(truth), int(guess)))
    return pairs


def confusion_matrix(
    actual: Sequence[int], predicted: Sequence[int], classes: int
) -> list[list[int]]:
    """행이 정답, 열이 예측."""
    if classes < 2:
        raise MetricError(f"classes는 2 이상이어야 합니다: {classes}")
    pairs = _checked_classes(actual, predicted)
    matrix = [[0] * classes for _ in range(classes)]
    for truth, guess in pairs:
        if truth >= classes or guess >= classes:
            raise MetricError(
                f"구간 번호가 범위를 벗어났습니다: 정답 {truth}, 예측 {guess}"
            )
        matrix[truth][guess] += 1
    return matrix


def macro_f1(
    actual: Sequence[int], predicted: Sequence[int], classes: int
) -> float:
    """구간별 F1의 단순 평균.

    **정확도를 쓰지 않는 이유가 여기 있다.** 잔반 분포가 한쪽으로 몰려
    있으면 다수 구간만 맞혀도 정확도가 높게 나온다. macro F1은 구간마다
    같은 무게를 준다.

    표본이 하나도 없는 구간은 평균에서 뺀다. 0으로 세면 등장하지 않는
    구간 때문에 점수가 내려간다.
    """
    matrix = confusion_matrix(actual, predicted, classes)
    scores: list[float] = []
    for index in range(classes):
        true_positive = matrix[index][index]
        actual_count = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        if actual_count == 0:
            continue
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / actual_count
        scores.append(
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
    if not scores:
        raise MetricError("정답에 등장하는 구간이 없습니다.")
    return sum(scores) / len(scores)


def quadratic_weighted_kappa(
    actual: Sequence[int], predicted: Sequence[int], classes: int
) -> float | None:
    """구간 거리를 제곱으로 벌주는 일치도.

    순서형에서 널리 쓰인다. 1이면 완전 일치, 0이면 우연 수준, 음수면 우연만
    못하다.

    **기대 행렬이 0이 되는 경우 None을 돌려준다.** 그럴 때 0을 돌려주면
    "우연 수준"이라는 뜻이 되는데, 실제로는 계산이 성립하지 않은 것이다.
    """
    matrix = confusion_matrix(actual, predicted, classes)
    total = sum(sum(row) for row in matrix)
    actual_counts = [sum(row) for row in matrix]
    predicted_counts = [sum(row[index] for row in matrix) for index in range(classes)]

    numerator = 0.0
    denominator = 0.0
    for i in range(classes):
        for j in range(classes):
            weight = ((i - j) ** 2) / ((classes - 1) ** 2)
            expected = actual_counts[i] * predicted_counts[j] / total
            numerator += weight * matrix[i][j]
            denominator += weight * expected
    if denominator == 0:
        return None
    return 1.0 - numerator / denominator


# --- 범위 제한 없는 회귀 지표 ------------------------------------------------
#
# 정답은 0~1이지만 예측은 그렇지 않을 수 있다. 아래 함수들은 예측값의 범위를
# 검사하지 않는다. 대신 유한한 숫자인지는 확인한다 — NaN이 섞이면 평균이
# 조용히 NaN이 되고, 그 표는 아무것도 말하지 않는다.


def unbounded_mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    pairs = _checked_pair(actual, predicted)
    return sum(abs(truth - guess) for truth, guess in pairs) / len(pairs)


def unbounded_rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    pairs = _checked_pair(actual, predicted)
    return math.sqrt(
        sum((truth - guess) ** 2 for truth, guess in pairs) / len(pairs)
    )


def unbounded_median_ae(
    actual: Sequence[float], predicted: Sequence[float]
) -> float:
    pairs = _checked_pair(actual, predicted)
    errors = sorted(abs(truth - guess) for truth, guess in pairs)
    middle = len(errors) // 2
    if len(errors) % 2:
        return errors[middle]
    return (errors[middle - 1] + errors[middle]) / 2


def unbounded_bias(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return bias(actual, predicted)


def regression_scores(
    actual: Sequence[float], predicted: Sequence[float]
) -> dict[str, float | int]:
    """회귀 지표 한 묶음. 예측 범위를 제한하지 않는다."""
    return {
        "mae": unbounded_mae(actual, predicted),
        "rmse": unbounded_rmse(actual, predicted),
        "medianAe": unbounded_median_ae(actual, predicted),
        "bias": unbounded_bias(actual, predicted),
        "count": len(actual),
    }
