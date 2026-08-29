"""출력 구조 — 점 하나를 바로 뱉지 않는 회귀 머리.

W4가 남긴 자리는 이렇다.

    frozen 표현은 고잔반 표본을 **판별**할 수 있다   AUROC 0.912, recall 0.877
    그런데 단순 재가중(R1)은 값이 너무 비쌌고        꼬리 −17%에 저잔반 +74%
    단순 특징 결합(R2)은 아무것도 바꾸지 않았다      소수점 세 자리까지 동일

W4가 보이지 **않은** 것도 분명히 해 둔다. "판별할 수 있다"와 "크기를 맞힐
만큼의 정보가 있다"는 다른 문장이다. 0.85와 1.00을 둘 다 고잔반으로 옳게
분류하면서 둘 다 0.3이라고 말하는 것은 모순이 아니다.

그래서 W5는 **출력 구조**를 바꿔 본다. 표현도 그대로, 정규화도 그대로 두고
"어떤 형태로 답을 내는가"만 바꾼다. 여기서도 안 되면 다음 용의자는 표현이다.

    S1  누적 순서형 분포 → 기대값
    S2  고잔반 게이트 + 두 전문가의 부드러운 혼합

**두 구조 모두 출력이 [0,1] 안에 있다.** 그래서 W5의 주 지표는 자른 예측
기준이다(W2와 달라진 점이며 보고서에 적는다).
"""

from __future__ import annotations

from typing import Any


class StructuredError(ValueError):
    """구조화된 출력을 만들 수 없다."""


#: 누적 확률을 재는 지점. W2가 고정한 등간격 5구간의 내부 경계 그대로다.
CUMULATIVE_THRESHOLDS = (0.2, 0.4, 0.6, 0.8)


def cumulative_targets(targets: Any, thresholds: tuple[float, ...] = CUMULATIVE_THRESHOLDS) -> Any:
    """`z_t = 1 if y >= t else 0`. 임계값마다 이진 라벨 하나.

    **`>=` 이다.** 구간이 `[0.8, 1.0]`처럼 왼쪽 닫힘이라는 뜻이고, 그래야
    마지막 칸이 `is_high_waste`(정답 >= 0.8)와 정확히 같은 집합이 된다.

    W4의 `tail.bin_index`는 `to_class_index` 규약을 따라 `> edge`를 쓴다.
    정답이 정확히 0.8인 표본 1건에서 둘이 갈린다 — 여기서는 왼쪽 닫힘으로
    통일한다. 한 문서 안에서 두 규약을 섞지 않기 위해서다.
    """
    import numpy as np

    array = np.asarray(targets, dtype=float)
    if array.ndim != 1:
        raise StructuredError(f"정답은 1차원이어야 합니다: {array.shape}")
    if not thresholds:
        raise StructuredError("임계값이 비어 있습니다.")
    if list(thresholds) != sorted(thresholds):
        raise StructuredError(f"임계값은 오름차순이어야 합니다: {thresholds}")
    return np.stack([(array >= t).astype(float) for t in thresholds], axis=1)


def cumulative_bin_index(
    targets: Any, thresholds: tuple[float, ...] = CUMULATIVE_THRESHOLDS
) -> Any:
    """넘은 임계값의 개수 = 구간 번호. 왼쪽 닫힘 규약이다."""
    import numpy as np

    return cumulative_targets(targets, thresholds).sum(axis=1).astype(int)


def monotonic_repair(cumulative: Any) -> Any:
    """누적 확률을 비증가로 만든다. **누적 최솟값(running minimum)** 규칙.

    임계값마다 독립된 이진 분류기를 쓰므로 `P(y>=0.4) > P(y>=0.2)` 같은
    위반이 생길 수 있다. 확률로 성립하지 않는 값이고, 그대로 두면 아래
    `bin_probabilities`에서 음수 확률이 나온다.

    규칙을 하나로 고정한다.

        q'_1 = q_1
        q'_t = min(q'_{t-1}, q_t)

    **왼쪽부터 훑으며 최솟값을 끌고 간다.** 오른쪽부터 최댓값을 끌고 오는
    방식(각 q를 뒤쪽 최댓값으로 올리는 것)도 유효한 보정이지만, 그 방식은
    확률을 **올리는** 쪽이라 고잔반 확률을 낙관적으로 부풀린다. 이 단계의
    관심사가 과소추정이므로 보수적인 쪽을 고른다.

    **결과를 보고 이 규칙을 바꾸지 않는다.**
    """
    import numpy as np

    matrix = np.asarray(cumulative, dtype=float)
    if matrix.ndim != 2:
        raise StructuredError(f"누적 확률은 2차원이어야 합니다: {matrix.shape}")
    if matrix.size == 0:
        raise StructuredError("누적 확률이 비어 있습니다.")
    if (matrix < -1e-9).any() or (matrix > 1 + 1e-9).any():
        raise StructuredError("누적 확률이 0~1 밖입니다.")
    return np.minimum.accumulate(np.clip(matrix, 0.0, 1.0), axis=1)


def bin_probabilities(cumulative: Any) -> Any:
    """보정된 누적 확률에서 구간 확률로.

        p_0 = 1 − q_1
        p_b = q_b − q_{b+1}
        p_B = q_B

    보정을 거친 입력이면 모든 `p >= 0`이고 합이 정확히 1이다. 그 성질을
    테스트로 고정한다 — 깨지면 기대값이 [0,1] 밖으로 나갈 수 있다.
    """
    import numpy as np

    repaired = monotonic_repair(cumulative)
    first = 1.0 - repaired[:, :1]
    middle = repaired[:, :-1] - repaired[:, 1:]
    last = repaired[:, -1:]
    probabilities = np.hstack([first, middle, last])
    # 부동소수 잔여만 정리한다. 음수가 크게 나오면 보정이 틀린 것이다.
    if (probabilities < -1e-9).any():
        raise StructuredError("음수 구간 확률이 나왔습니다. 보정을 확인하세요.")
    return np.clip(probabilities, 0.0, None)


def bin_representatives(
    train_targets: Any,
    thresholds: tuple[float, ...] = CUMULATIVE_THRESHOLDS,
    fallback: tuple[float, ...] | None = None,
) -> Any:
    """구간마다의 대표값. **학습 fold 정답의 구간별 평균.**

    고정 중점(0.1, 0.3, …)을 쓸 수도 있지만, 이 데이터의 정답이 0과 1에
    몰려 있다는 것을 이미 안다 — 마지막 칸의 실제 평균은 0.96이고 중점 0.9가
    아니다. 중점을 쓰면 구조가 맞아도 기대값이 체계적으로 낮게 나온다.

    **평가 fold의 정답을 보지 않는다.** 부르는 쪽이 학습 fold 정답만 넘겨야
    하고, 그 규칙은 테스트가 확인한다.

    빈 구간이 있으면 `fallback`(기본값: 구간 중점)을 쓴다. 평가 라벨을 보고
    채우지 않는다. 이 규칙도 결과 전에 정했다.
    """
    import numpy as np

    array = np.asarray(train_targets, dtype=float)
    if array.size == 0:
        raise StructuredError("학습 정답이 없습니다.")
    edges = (0.0,) + tuple(thresholds) + (1.0,)
    if fallback is None:
        fallback = tuple(
            (edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)
        )
    if len(fallback) != len(thresholds) + 1:
        raise StructuredError(
            f"대체값 개수가 구간 수와 다릅니다: {len(fallback)} vs {len(thresholds) + 1}"
        )

    index = cumulative_bin_index(array, thresholds)
    representatives = np.array(fallback, dtype=float)
    for b in range(len(thresholds) + 1):
        mask = index == b
        if mask.any():
            representatives[b] = float(array[mask].mean())
    return representatives


def expected_value(probabilities: Any, representatives: Any) -> Any:
    """`y_hat = Σ p_b · r_b`.

    대표값이 전부 [0,1] 안에 있고 확률 합이 1이므로 결과도 [0,1] 안이다.
    구조 자체가 범위를 지킨다 — 나중에 잘라서 지키는 것이 아니다.
    """
    import numpy as np

    p = np.asarray(probabilities, dtype=float)
    r = np.asarray(representatives, dtype=float)
    if p.ndim != 2:
        raise StructuredError(f"구간 확률은 2차원이어야 합니다: {p.shape}")
    if p.shape[1] != len(r):
        raise StructuredError(f"구간 수가 다릅니다: {p.shape[1]} vs {len(r)}")
    return p @ r


def soft_mixture(gate_score: Any, low_prediction: Any, high_prediction: Any) -> Any:
    """`y_hat = (1 − g) · y_low + g · y_high`.

    **`g`를 보정된 확률이라고 부르지 않는다.** 로지스틱 출력을 게이트 점수로
    쓸 뿐이고, 0.7이 "70% 확률로 고잔반"이라는 뜻이 아니다.

    **딱딱한 전환(`g > 0.5`면 high 전문가)을 쓰지 않는다.** 경계에서 예측이
    튀고, 그 불연속 자체가 새로운 실패를 만든다. 여기서 재려는 것은 혼합
    구조이지 전환 규칙이 아니다.
    """
    import numpy as np

    g = np.asarray(gate_score, dtype=float)
    low = np.asarray(low_prediction, dtype=float)
    high = np.asarray(high_prediction, dtype=float)
    if not (len(g) == len(low) == len(high)):
        raise StructuredError(f"길이가 다릅니다: {len(g)}, {len(low)}, {len(high)}")
    if (g < -1e-9).any() or (g > 1 + 1e-9).any():
        raise StructuredError("게이트 점수가 0~1 밖입니다.")
    g = np.clip(g, 0.0, 1.0)
    return (1.0 - g) * low + g * high


def prediction_quantiles(values: Any) -> dict[str, float]:
    """예측 분포 요약. 구조가 단순히 분산만 키운 것인지 보기 위해서다."""
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise StructuredError("값이 비어 있습니다.")
    return {
        "min": float(array.min()),
        "q25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "q75": float(np.percentile(array, 75)),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }
