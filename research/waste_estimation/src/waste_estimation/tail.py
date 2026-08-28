"""꼬리(tail) — 고잔반 구간을 다루는 정의와 도구.

W2와 W3이 같은 자리를 두 번 가리켰다.

    W2   low MAE 0.059 / high MAE 0.305
    W3   high 구간 안에서 선택적 예측이 아무것도 못 함 (0.250 -> 0.250)
    W3   큰 오차 19건 중 10건이 APPROVED로 통과

대표 실패는 전부 같은 모양이다 — 정답 0.85~1.00, 예측 0.03~0.56.
**거의 다 남긴 식판을 거의 다 먹었다고 말한다.**

W4가 묻는 것은 "이게 신뢰 신호로 막을 문제인가"가 아니라 **"애초에 회귀가
평균으로 수축하는 문제인가"** 다. 여기 있는 것들은 그 질문을 재기 위한
정의와 도구다.

**정의를 결과 보고 바꾸지 않는다.** 구간 경계도, 고잔반 기준도, 치명적
과소추정 기준도 전부 앞 단계에서 이미 쓰던 값을 그대로 가져왔다.
"""

from __future__ import annotations

from typing import Any


class TailError(ValueError):
    """꼬리 계산을 할 수 없다."""


#: 고잔반 기준. W2의 순서형 구간 마지막 칸(`ORDINAL_EDGES` 0.8)과 같다.
#: 새로 만든 경계가 아니다.
HIGH_WASTE_THRESHOLD = 0.8

#: 치명적 과소추정의 크기. W3의 큰 오차 분석에서 쓴 값과 같다.
CATASTROPHIC_MARGIN = 0.30

#: 이름을 코드와 보고서에서 같게 쓴다.
CATASTROPHIC_NAME = "HIGH_WASTE_CATASTROPHIC_UNDERESTIMATION"


def is_high_waste(targets: Any) -> Any:
    """정답이 고잔반 구간인가. **정답을 본다 — 평가에만 쓴다.**"""
    import numpy as np

    return np.asarray(targets, dtype=float) >= HIGH_WASTE_THRESHOLD


def signed_error(predictions: Any, targets: Any) -> Any:
    """예측 − 정답. **음수가 과소추정이다.**

    절댓값만 보면 방향을 잃는다. 잔반 0.9를 0.3이라 말하는 것과 0.3을 0.9라
    말하는 것은 크기가 같아도 운영에서 전혀 다른 사고다. 전자는 남은 음식을
    못 보고 지나가는 것이고, 후자는 헛경보다.
    """
    import numpy as np

    predicted = np.asarray(predictions, dtype=float)
    actual = np.asarray(targets, dtype=float)
    if len(predicted) != len(actual):
        raise TailError(f"길이가 다릅니다: {len(predicted)}, {len(actual)}")
    return predicted - actual


def catastrophic_underestimation(predictions: Any, targets: Any) -> Any:
    """고잔반인데 그만큼 낮게 말한 표본.

        정답 >= 0.8  그리고  정답 − 예측 >= 0.30

    두 조건을 모두 요구하는 이유가 있다. 큰 오차만 세면 저잔반을 높게 부른
    헛경보도 같이 세어지고, 고잔반만 세면 잘 맞힌 것까지 들어간다. **제품이
    무서워하는 칸은 이 둘의 교집합 하나다.**
    """
    import numpy as np

    predicted = np.asarray(predictions, dtype=float)
    actual = np.asarray(targets, dtype=float)
    if len(predicted) != len(actual):
        raise TailError(f"길이가 다릅니다: {len(predicted)}, {len(actual)}")
    return is_high_waste(actual) & ((actual - predicted) >= CATASTROPHIC_MARGIN)


def catastrophic_rate(predictions: Any, targets: Any) -> float:
    """고잔반 표본 중 치명적 과소추정의 비율.

    분모는 전체가 아니라 **고잔반 표본 수**다. 전체로 나누면 고잔반이 드문
    집합에서 비율이 저절로 작아 보인다.
    """
    import numpy as np

    high = is_high_waste(targets)
    count = int(high.sum())
    if count == 0:
        raise TailError("고잔반 표본이 없어 비율을 정의할 수 없습니다.")
    return float(catastrophic_underestimation(predictions, targets).sum()) / count


def percentile_abs_error(predictions: Any, targets: Any, percentile: float = 90.0) -> float:
    """상위 백분위 절대 오차. 평균이 가리는 꼬리를 본다."""
    import numpy as np

    predicted = np.asarray(predictions, dtype=float)
    actual = np.asarray(targets, dtype=float)
    if len(predicted) == 0:
        raise TailError("표본이 없습니다.")
    return float(np.percentile(np.abs(predicted - actual), percentile))


# --- 구간 균형 가중치 -------------------------------------------------------


def bin_index(targets: Any, edges: tuple[float, ...]) -> Any:
    """정답을 고정 구간 번호로. 경계는 아래 구간에 들어간다."""
    import numpy as np

    array = np.asarray(targets, dtype=float)
    index = np.zeros(len(array), dtype=int)
    for edge in edges:
        index += (array > edge).astype(int)
    return index


def inverse_sqrt_bin_weights(targets: Any, edges: tuple[float, ...]) -> Any:
    """구간 빈도의 역제곱근 가중치. 평균이 1이 되도록 정규화한다.

        w_i = (1 / sqrt(n_b)) / mean(1 / sqrt(n_b))

    **완전한 역빈도가 아니라 역제곱근인 이유.** 이 데이터에서 가장 드문 구간은
    가장 흔한 구간보다 수십 배 적다. 역빈도를 그대로 쓰면 표본 몇 개가 적합
    전체를 끌고 가고, 그 몇 개가 흔들리면 모델도 같이 흔들린다. 제곱근은 그
    영향을 눌러 준다 — 방향은 같고 세기만 보수적이다.

    평균 1 정규화는 alpha의 의미를 유지하기 위해서다. 가중치 총합이 변하면
    같은 alpha가 사실상 다른 정규화 세기가 되고, 그러면 "가중치 효과"와
    "정규화 효과"를 구분할 수 없다.

    **빈도는 부르는 쪽이 넘긴 정답에서만 센다.** 학습 fold의 정답만 넘겨야
    한다. 평가 fold의 분포가 섞이면 그 순간 누수다.
    """
    import numpy as np

    array = np.asarray(targets, dtype=float)
    if len(array) == 0:
        raise TailError("정답이 없습니다.")
    index = bin_index(array, edges)
    counts = np.bincount(index, minlength=len(edges) + 1).astype(float)
    if (counts[index] <= 0).any():  # 도달 불가 — 방어적으로 남긴다
        raise TailError("빈 구간에서 가중치를 만들 수 없습니다.")
    weights = 1.0 / np.sqrt(counts[index])
    return weights / weights.mean()


# --- 참조 분포로 정규화한 novelty --------------------------------------------


def reference_loo_distances(reference: Any, k: int) -> Any:
    """참조 집합 내부의 leave-one-out kNN 거리 분포.

    **자기 자신을 이웃으로 세지 않는다.** 세면 거리가 0이 섞여 분포 전체가
    왼쪽으로 밀리고, 그 분포로 정규화한 점수는 전부 부풀려진다.

    이 분포가 "이 참조 집합에서 정상적인 거리는 이 정도"라는 자를 만든다.
    """
    import numpy as np

    from .reliability import l2_normalize

    matrix = l2_normalize(reference)
    if len(matrix) <= k:
        raise TailError(f"참조가 {len(matrix)}개인데 k={k}를 요구했습니다.")
    distance = 1.0 - matrix @ matrix.T
    np.fill_diagonal(distance, np.inf)  # 자기 자신 제외
    nearest = np.partition(distance, k - 1, axis=1)[:, :k]
    return nearest.mean(axis=1)


def normalized_novelty(query: Any, reference: Any, k: int = 3) -> Any:
    """참조 분포에 대한 경험적 백분위로 바꾼 novelty.

    **W3이 놓친 자리가 여기다.** W3은 raw 코사인 거리에 절대 임계값을 걸었다.
    그런데 kNN 거리는 참조 집합이 커지면 저절로 짧아진다 — 이웃이 많아지니
    가장 가까운 셋도 더 가까워진다. 그래서 318건으로 잡은 임계값을 398건
    참조에 그대로 옮겼더니, 목표 50% coverage가 62.9%로 실현됐다.

    백분위로 바꾸면 그 축척이 사라진다. 0.90은 "이 참조 집합 내부 거리의 90
    백분위보다 멀다"는 뜻이고, 그 문장은 참조가 몇 건이든 같은 뜻이다.

    **이 값은 보정된 정확도 확률이 아니다.** 0.90이 "90% 확률로 틀린다"는
    뜻이 전혀 아니다. 여전히 순위를 매기는 값이다.
    """
    import numpy as np

    from .reliability import knn_novelty

    raw = knn_novelty(query, reference, k=k)
    baseline = np.sort(reference_loo_distances(reference, k))
    # 참조 분포에서 raw보다 작은 값의 비율. 오른쪽 삽입점을 쓰면 동점이
    # 백분위를 낮추지 않는다.
    rank = np.searchsorted(baseline, raw, side="right")
    return rank / len(baseline)


# --- coverage 이동 ----------------------------------------------------------


def realized_coverage(scores: Any, threshold: float) -> float:
    """임계값을 걸었을 때 실제로 승인되는 비율."""
    import numpy as np

    array = np.asarray(scores, dtype=float)
    if len(array) == 0:
        raise TailError("점수가 비어 있습니다.")
    return float((array <= threshold).mean())


def coverage_drift(target: float, realized: float) -> float:
    """목표 coverage와 실현 coverage의 차이. **작을수록 좋다.**

    W3에서 이 값을 재지 않았다. 재지 않으면 "80% coverage에서 MAE 0.14"라는
    문장이 실제로 몇 %를 승인했는지 모른 채 돌아다닌다.
    """
    if not 0.0 <= target <= 1.0:
        raise TailError(f"목표 coverage는 0~1이어야 합니다: {target}")
    if not 0.0 <= realized <= 1.0:
        raise TailError(f"실현 coverage는 0~1이어야 합니다: {realized}")
    return abs(realized - target)
