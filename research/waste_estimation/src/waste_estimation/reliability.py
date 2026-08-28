"""신뢰 신호(reliability signal).

W2가 남긴 문제는 오차의 **크기**가 아니라 **분포**였다.

    LOW 구간    MAE 0.059
    HIGH 구간   MAE 0.305      많이 남긴 식판을 거의 다 먹었다고 말한다

전체 MAE 하나로는 이 모델을 설명할 수 없다. 그래서 W3이 묻는 것은
"더 잘 맞히는가"가 아니라 **"틀릴 때를 미리 아는가"** 다.

**여기 있는 어떤 신호도 정답을 보지 않는다.** 추론 시점에 계산할 수 있는
것만 쓴다. 오차나 잔차를 신뢰 신호로 쓰면 그것은 예측이 아니라 채점이다.

    C0  embedding novelty      학습 임베딩에서 얼마나 멀리 있는가
    C1  fold disagreement      여러 모델이 얼마나 다르게 말하는가
    C2  prediction extremeness 예측이 학습 평균에서 얼마나 먼가

**부호를 하나로 통일한다.** 모든 점수는 **높을수록 덜 믿을 만하다**.
`confidence`라고 부르지 않는 이유가 여기 있다 — 방향이 반대이고, 무엇보다
이 값은 **보정된 정확도 확률이 아니다.** "0.82"가 "82% 확률로 맞다"는 뜻이
아니다. CCTV 트랙에서 배운 것과 같다.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ReliabilityError(ValueError):
    """신뢰 신호를 계산할 수 없다."""


class Signal(str, Enum):
    """후보 신호. 이름이 곧 기록에 남는 식별자다."""

    EMBEDDING_NOVELTY = "C0_embedding_novelty"
    FOLD_DISAGREEMENT = "C1_fold_disagreement"
    PREDICTION_EXTREMENESS = "C2_prediction_extremeness"
    COMBINED = "combined"


def _as_matrix(values: Any) -> Any:
    import numpy as np

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ReliabilityError(f"임베딩은 2차원이어야 합니다: {matrix.shape}")
    if matrix.size == 0:
        raise ReliabilityError("임베딩이 비어 있습니다.")
    return matrix


def l2_normalize(matrix: Any) -> Any:
    """행마다 길이를 1로. 코사인 거리를 내적으로 계산하기 위해서다."""
    import numpy as np

    array = _as_matrix(matrix)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    # 길이가 0인 행은 그대로 둔다. 나누면 NaN이 되고 그 NaN이 조용히 퍼진다.
    norms[norms < 1e-12] = 1.0
    return array / norms


def knn_novelty(
    query: Any, reference: Any, k: int = 5
) -> Any:
    """C0. 학습 임베딩에서 얼마나 떨어져 있는가.

    코사인 거리(1 - 코사인 유사도) 기준 k개 최근접 이웃의 평균.
    **값이 클수록 처음 보는 모양**이고, 처음 보는 모양에서 회귀가 흔들린다는
    것이 이 신호의 가정이다.

    **자기 자신을 이웃으로 넣지 않는다.** OOF에서 reference는 그 표본이
    속하지 않은 학습 fold의 임베딩이어야 한다. 자기 임베딩이 섞이면 거리가
    0에 가까워지고 신호가 무너진다. 그 규칙은 부르는 쪽이 지키고, 테스트가
    확인한다.

    거리를 쓰는 이유는 단순함이다. Mahalanobis는 공분산 추정이 필요한데
    표본 400건에 특징 512차원이면 그 추정 자체가 불안정하다.
    """
    import numpy as np

    if k < 1:
        raise ReliabilityError(f"k는 1 이상이어야 합니다: {k}")
    query_n = l2_normalize(query)
    reference_n = l2_normalize(reference)
    if query_n.shape[1] != reference_n.shape[1]:
        raise ReliabilityError(
            f"차원이 다릅니다: {query_n.shape[1]} vs {reference_n.shape[1]}"
        )
    if len(reference_n) < k:
        raise ReliabilityError(
            f"참조 임베딩이 {len(reference_n)}개인데 k={k}를 요구했습니다."
        )

    similarity = query_n @ reference_n.T
    distance = 1.0 - similarity
    # 가장 가까운 k개. 전체 정렬 없이 부분 정렬로 충분하다.
    nearest = np.partition(distance, k - 1, axis=1)[:, :k]
    return nearest.mean(axis=1)


def bootstrap_disagreement(
    fit_predict: Any,
    train_index: Any,
    query_features: Any,
    members: int = 5,
    seed: int = 42,
) -> Any:
    """C1. 여러 모델이 얼마나 다르게 말하는가.

    deep ensemble을 새로 학습하지 않는다. **같은 frozen 임베딩** 위에서
    학습 표본만 부트스트랩으로 바꿔 회귀기를 여러 개 맞추고, 그 예측의
    표준편차를 쓴다. backbone은 건드리지 않으므로 변화는 오직 회귀 적합에서만
    온다.

    `fit_predict(indices) -> predictions` 를 받는다. 회귀기의 종류를 여기서
    정하지 않기 위해서다 — W2가 고른 것을 그대로 넘겨받는다.
    """
    import numpy as np

    if members < 2:
        raise ReliabilityError(f"members는 2 이상이어야 합니다: {members}")
    indices = np.asarray(train_index)
    if len(indices) == 0:
        raise ReliabilityError("학습 표본이 없습니다.")

    rng = np.random.default_rng(seed)
    predictions = []
    for _ in range(members):
        sample = rng.choice(indices, size=len(indices), replace=True)
        predictions.append(np.asarray(fit_predict(sample), dtype=float))
    stacked = np.vstack(predictions)
    if stacked.shape[1] != len(query_features):
        raise ReliabilityError(
            f"예측 개수가 질의 표본 수와 다릅니다: {stacked.shape[1]} vs "
            f"{len(query_features)}"
        )
    return stacked.std(axis=0, ddof=0)


def prediction_extremeness(predictions: Any, reference_mean: float) -> Any:
    """C2. 예측이 학습 평균에서 얼마나 먼가.

    **가정이 들어 있는 신호다.** "극단적인 예측일수록 불확실하다"는 것은
    참일 수도 거짓일 수도 있고, 이 데이터에서는 오히려 반대일 수 있다 —
    W2에서 모델은 극단을 잘 못 맞히는 것이 아니라 **극단을 아예 예측하지
    않는** 쪽으로 치우쳐 있었다.

    그래서 이것을 주 후보로 두지 않고, C0/C1이 실제로 정보를 갖는지 재는
    **단순 대조**로만 쓴다.
    """
    import numpy as np

    array = np.asarray(predictions, dtype=float)
    if array.size == 0:
        raise ReliabilityError("예측이 비어 있습니다.")
    return np.abs(array - float(reference_mean))


def rank_normalize(scores: Any) -> Any:
    """0~1의 순위 값으로 바꾼다. 축척이 다른 신호를 합치기 위해서다.

    값 자체가 아니라 **순서**만 남긴다. C0는 코사인 거리(0~2), C1은 잔반률
    단위의 표준편차라 축척이 완전히 다르다. 그대로 더하면 한쪽이 결합을
    지배한다.
    """
    import numpy as np

    array = np.asarray(scores, dtype=float)
    if array.size == 0:
        raise ReliabilityError("점수가 비어 있습니다.")
    if array.size == 1:
        return np.zeros(1, dtype=float)
    order = array.argsort(kind="stable")
    ranks = np.empty(len(array), dtype=float)
    ranks[order] = np.arange(len(array), dtype=float)
    return ranks / (len(array) - 1)


def combine(*score_arrays: Any) -> Any:
    """순위 정규화 후 평균.

    **메타 모델을 학습하지 않는다.** 가중치를 데이터로 맞추기 시작하면 그
    가중치를 어디서 골랐는지가 또 하나의 선택이 되고, 표본 400건에서 그것을
    정당화할 수 없다. 동일 가중 평균이면 설명할 것이 없다.
    """
    import numpy as np

    if len(score_arrays) < 2:
        raise ReliabilityError("결합하려면 신호가 둘 이상 필요합니다.")
    normalized = [rank_normalize(scores) for scores in score_arrays]
    lengths = {len(item) for item in normalized}
    if len(lengths) != 1:
        raise ReliabilityError(f"길이가 다릅니다: {sorted(lengths)}")
    return np.mean(np.vstack(normalized), axis=0)


# --- 선택적 예측 평가 -------------------------------------------------------
#
# W1의 `selective.py`는 confidence(높을수록 믿을 만함)를 받는다. 여기 점수는
# 반대 방향이므로 부호를 뒤집어 넘긴다. 그 변환을 한 곳에 모아 둔다.


def as_confidence(reliability_score: Any) -> Any:
    """신뢰 점수를 W1 유틸리티가 받는 confidence 방향으로 바꾼다."""
    import numpy as np

    return -np.asarray(reliability_score, dtype=float)


def risk_at_coverage_points(
    errors: Any, reliability_score: Any, coverages: tuple[float, ...]
) -> dict[float, float | None]:
    """정해진 coverage에서의 위험(승인 표본 MAE).

    **coverage를 함께 보고하지 않는 위험 값은 의미가 없다.** 5%만 승인하면
    거의 어떤 모델이든 MAE가 낮다. 그래서 이 함수는 coverage를 키로 돌려준다.

    coverage 0은 None이다. 승인이 하나도 없을 때 0을 돌려주면 아무것도 하지
    않은 정책이 완벽해 보인다.
    """
    import numpy as np

    error_array = np.asarray(errors, dtype=float)
    score_array = np.asarray(reliability_score, dtype=float)
    if len(error_array) != len(score_array):
        raise ReliabilityError(
            f"길이가 다릅니다: 오차 {len(error_array)}, 점수 {len(score_array)}"
        )
    if len(error_array) == 0:
        raise ReliabilityError("입력이 비어 있습니다.")

    order = score_array.argsort(kind="stable")  # 믿을 만한 것부터
    result: dict[float, float | None] = {}
    for coverage in coverages:
        if not 0.0 <= coverage <= 1.0:
            raise ReliabilityError(f"coverage는 0~1이어야 합니다: {coverage}")
        keep = int(round(coverage * len(error_array)))
        result[coverage] = (
            None if keep == 0 else float(error_array[order[:keep]].mean())
        )
    return result


def aurc(errors: Any, reliability_score: Any) -> float:
    """위험-coverage 곡선 아래 면적. **낮을수록 좋다.**

    정의를 여기 적어 둔다.

        신뢰 점수가 낮은 순(믿을 만한 순)으로 정렬한다.
        k = 1..n 각각에 대해 상위 k개의 MAE를 구한다.
        그 n개 값의 평균이 AURC다.

    즉 "모든 coverage 지점에서의 평균 위험"이다. coverage 격자를 임의로
    고르지 않으므로 격자 선택이 결과를 바꾸지 않는다.
    """
    import numpy as np

    error_array = np.asarray(errors, dtype=float)
    score_array = np.asarray(reliability_score, dtype=float)
    if len(error_array) != len(score_array):
        raise ReliabilityError("길이가 다릅니다.")
    if len(error_array) == 0:
        raise ReliabilityError("입력이 비어 있습니다.")

    order = score_array.argsort(kind="stable")
    sorted_errors = error_array[order]
    cumulative = np.cumsum(sorted_errors) / np.arange(1, len(sorted_errors) + 1)
    return float(cumulative.mean())


def random_rejection_risk(
    errors: Any, coverage: float, trials: int = 200, seed: int = 42
) -> float | None:
    """같은 비율을 **무작위로** 버렸을 때의 기대 위험.

    **이 대조가 없으면 아무 말도 할 수 없다.** 표본을 버리면 평균이 흔들리고,
    운이 좋으면 MAE가 내려간다. 신뢰 신호가 실제로 어려운 표본을 골랐다는
    주장은 무작위 제거보다 낫다는 것을 보여야 성립한다.

    무작위 제거의 기대 MAE는 사실 전체 MAE와 같다. 그래도 몬테카를로로
    분포까지 보는 이유는 **표본이 적을 때의 흔들림 폭**을 알기 위해서다.
    """
    import numpy as np

    error_array = np.asarray(errors, dtype=float)
    if not 0.0 <= coverage <= 1.0:
        raise ReliabilityError(f"coverage는 0~1이어야 합니다: {coverage}")
    keep = int(round(coverage * len(error_array)))
    if keep == 0:
        return None
    rng = np.random.default_rng(seed)
    samples = [
        error_array[rng.choice(len(error_array), size=keep, replace=False)].mean()
        for _ in range(trials)
    ]
    return float(np.mean(samples))


def oracle_rejection_risk(errors: Any, coverage: float) -> float | None:
    """**ORACLE UPPER BOUND.** 실제 오차가 큰 것부터 버린다.

    이것은 실행 가능한 방법이 **아니다.** 정답을 알아야 계산되고, 정답을
    알면 예측할 이유가 없다. 곡선의 도달 가능한 최선을 보여 주는 선으로만
    쓴다. 보고서에서 반드시 oracle이라고 적는다.
    """
    import numpy as np

    error_array = np.asarray(errors, dtype=float)
    if not 0.0 <= coverage <= 1.0:
        raise ReliabilityError(f"coverage는 0~1이어야 합니다: {coverage}")
    keep = int(round(coverage * len(error_array)))
    if keep == 0:
        return None
    return float(np.sort(error_array)[:keep].mean())


def spearman(first: Any, second: Any) -> float:
    """순위 상관.

    Pearson만 쓰지 않는 이유는 관계가 선형이라는 보장이 없기 때문이다.
    신뢰 점수가 오차와 **단조롭게** 움직이기만 하면 선택적 예측은 작동한다.
    """
    import numpy as np

    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if len(a) != len(b):
        raise ReliabilityError(f"길이가 다릅니다: {len(a)}, {len(b)}")
    if len(a) < 3:
        raise ReliabilityError("표본이 너무 적습니다.")

    def ranks(values: Any) -> Any:
        order = values.argsort(kind="stable")
        result = np.empty(len(values), dtype=float)
        result[order] = np.arange(1, len(values) + 1, dtype=float)
        # 동점은 평균 순위로. 동점을 무시하면 상관이 부풀려진다.
        unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        for index, count in enumerate(counts):
            if count > 1:
                mask = inverse == index
                result[mask] = result[mask].mean()
        return result

    rank_a, rank_b = ranks(a), ranks(b)
    rank_a = rank_a - rank_a.mean()
    rank_b = rank_b - rank_b.mean()
    denominator = np.sqrt((rank_a**2).sum() * (rank_b**2).sum())
    if denominator == 0:
        raise ReliabilityError("한쪽 값이 전부 같아 상관을 정의할 수 없습니다.")
    return float((rank_a * rank_b).sum() / denominator)
