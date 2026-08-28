"""회귀 baseline.

**scikit-learn을 넣지 않는다.** 여기서 필요한 것은 ridge 회귀 하나이고,
정규방정식 세 줄이면 끝난다. 의존성을 하나 줄이는 것보다 중요한 이유는
따로 있다 — 이 코드가 무엇을 하는지 읽어서 확인할 수 있어야 한다. baseline이
블랙박스이면 그것과 비교한 모델의 우열도 블랙박스가 된다.

세 가지를 같은 인터페이스로 둔다.

    GlobalMean   학습 집합의 평균을 모든 표본에 낸다. **넘지 못하면 실패다**
    Ridge        표준화 + L2 정규화 선형 회귀
    (ordinal)    구간 예측은 Ridge 결과를 구간으로 옮겨 비교한다

**출력을 자르지 않는다.** 회귀는 0~1 밖으로 나갈 수 있고, 지표 계산 전에
몰래 clip하면 성능이 좋아 보인다. 자르는 것은 서비스 정책이므로 `clip()`을
따로 두고 raw와 나란히 보고한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class BaselineError(ValueError):
    """모델을 만들 수 없다."""


@dataclass
class GlobalMean:
    """학습 집합의 평균을 그대로 낸다.

    **이것을 넘지 못하는 모델은 아무것도 배우지 못한 것이다.** 딥러닝 결과가
    MAE 0.30을 냈을 때 이 baseline이 0.31이면 그 모델은 이미지를 거의 보고
    있지 않다. 그 사실은 비교 대상이 있어야 보인다.
    """

    value: float | None = field(default=None, init=False)

    def fit(self, features: Any, targets: Any) -> "GlobalMean":
        import numpy as np

        array = np.asarray(targets, dtype=float)
        if array.size == 0:
            raise BaselineError("학습 표본이 없습니다.")
        self.value = float(array.mean())
        return self

    def predict(self, features: Any) -> Any:
        import numpy as np

        if self.value is None:
            raise BaselineError("fit()을 먼저 부르세요.")
        rows = len(features) if features is not None else 0
        return np.full(rows, self.value, dtype=float)


@dataclass
class Ridge:
    """표준화 + L2 정규화 선형 회귀.

    정규방정식으로 푼다.

        w = (XᵀX + λI)⁻¹ Xᵀy

    특징 수가 임베딩 차원(수백~천)이고 표본이 수백이라 닫힌 해가 빠르고
    결정적이다. 반복 최적화의 초기값·학습률 같은 자유도가 없어 재현도 쉽다.

    **표준화는 학습 집합에서만 계산한다.** 평가 집합의 통계를 쓰면 그 순간
    누수다.
    """

    alpha: float = 1.0

    _mean: Any = field(default=None, init=False, repr=False)
    _scale: Any = field(default=None, init=False, repr=False)
    _weights: Any = field(default=None, init=False, repr=False)
    _intercept: float = field(default=0.0, init=False, repr=False)

    def fit(
        self, features: Any, targets: Any, sample_weight: Any = None
    ) -> "Ridge":
        """`sample_weight`를 주면 가중 정규방정식을 푼다.

            w = (XᵀWX + λI)⁻¹ XᵀW(y − ȳ_w)

        표준화는 **가중치 없이** 학습 집합 통계로 한다. 특징의 축척은 가중치와
        무관한 성질이고, 여기까지 가중치를 태우면 "가중치 효과"와 "축척 변화"가
        섞여 나중에 분리할 수 없다.

        기본값 `None`이면 W2·W3이 쓰던 계산과 완전히 같다.
        """
        import numpy as np

        matrix = np.asarray(features, dtype=float)
        target = np.asarray(targets, dtype=float)
        if matrix.ndim != 2:
            raise BaselineError(f"특징은 2차원이어야 합니다: {matrix.shape}")
        if len(matrix) != len(target):
            raise BaselineError(
                f"개수가 다릅니다: 특징 {len(matrix)}, 정답 {len(target)}"
            )
        if len(matrix) == 0:
            raise BaselineError("학습 표본이 없습니다.")
        if self.alpha < 0:
            raise BaselineError(f"alpha는 0 이상이어야 합니다: {self.alpha}")

        if sample_weight is None:
            weights = np.ones(len(matrix), dtype=float)
        else:
            weights = np.asarray(sample_weight, dtype=float)
            if len(weights) != len(matrix):
                raise BaselineError(
                    f"가중치 개수가 다릅니다: {len(weights)} vs {len(matrix)}"
                )
            if (weights < 0).any():
                raise BaselineError("가중치는 음수일 수 없습니다.")
            if weights.sum() <= 0:
                raise BaselineError("가중치 합이 0입니다.")

        self._mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        # 상수 특징은 나누면 폭발한다. 1로 두면 표준화 뒤 0이 된다.
        scale[scale < 1e-8] = 1.0
        self._scale = scale
        centered = (matrix - self._mean) / scale

        self._intercept = float((weights * target).sum() / weights.sum())
        residual = target - self._intercept
        weighted = centered * weights[:, None]
        gram = centered.T @ weighted
        gram[np.diag_indices_from(gram)] += self.alpha
        self._weights = np.linalg.solve(gram, weighted.T @ residual)
        return self

    def predict(self, features: Any) -> Any:
        import numpy as np

        if self._weights is None:
            raise BaselineError("fit()을 먼저 부르세요.")
        matrix = np.asarray(features, dtype=float)
        centered = (matrix - self._mean) / self._scale
        return centered @ self._weights + self._intercept


@dataclass
class LogisticRegression:
    """L2 정규화 로지스틱 회귀. IRLS(뉴턴법)로 푼다.

    **여기도 scikit-learn을 넣지 않는다.** 이 트랙은 W1부터 numpy 닫힌 해로만
    baseline을 만들어 왔고, 보조 분류기 하나 때문에 무거운 의존성을 연구
    환경에 들이면 그 경계가 무너진다. 필요한 것은 선형 분류기 하나다.

    **정규화가 선택이 아니라 필수인 이유.** 특징이 512차원이고 표본이 수백
    건이면 두 클래스는 거의 항상 완전 분리된다. 정규화가 없으면 가중치가
    발산하고 예측 확률이 0과 1로 붙어 버린다 — 그러면 stacking 입력으로서
    아무 정보도 남지 않는다.

    `class_weight="balanced"`는 표준적인 처리다. 고잔반이 드무니 그대로 두면
    분류기가 "전부 저잔반"이라고 답하는 쪽이 손실이 낮다.
    """

    alpha: float = 100.0
    class_weight: str | None = None
    max_iterations: int = 50
    tolerance: float = 1e-6

    _mean: Any = field(default=None, init=False, repr=False)
    _scale: Any = field(default=None, init=False, repr=False)
    _weights: Any = field(default=None, init=False, repr=False)
    _intercept: float = field(default=0.0, init=False, repr=False)
    iterations: int = field(default=0, init=False)

    def fit(self, features: Any, labels: Any) -> "LogisticRegression":
        import numpy as np

        matrix = np.asarray(features, dtype=float)
        target = np.asarray(labels, dtype=float)
        if matrix.ndim != 2:
            raise BaselineError(f"특징은 2차원이어야 합니다: {matrix.shape}")
        if len(matrix) != len(target):
            raise BaselineError(
                f"개수가 다릅니다: 특징 {len(matrix)}, 라벨 {len(target)}"
            )
        if not set(np.unique(target)).issubset({0.0, 1.0}):
            raise BaselineError("라벨은 0 또는 1이어야 합니다.")
        if len(np.unique(target)) < 2:
            raise BaselineError("한 클래스만 있어 분류기를 만들 수 없습니다.")
        if self.alpha <= 0:
            raise BaselineError(f"alpha는 양수여야 합니다: {self.alpha}")
        if self.class_weight not in (None, "balanced"):
            raise BaselineError(f"알 수 없는 class_weight: {self.class_weight}")

        self._mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < 1e-8] = 1.0
        self._scale = scale
        design = np.hstack(
            [np.ones((len(matrix), 1)), (matrix - self._mean) / scale]
        )

        if self.class_weight == "balanced":
            counts = np.array([(target == 0).sum(), (target == 1).sum()], dtype=float)
            per_class = len(target) / (2.0 * counts)
            weights = np.where(target == 1, per_class[1], per_class[0])
        else:
            weights = np.ones(len(target), dtype=float)

        beta = np.zeros(design.shape[1], dtype=float)
        # 절편은 정규화하지 않는다. 정규화하면 기저 비율이 0.5 쪽으로 끌린다.
        penalty = np.full(design.shape[1], self.alpha, dtype=float)
        penalty[0] = 0.0

        for step in range(self.max_iterations):
            eta = design @ beta
            probability = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
            variance = np.clip(probability * (1.0 - probability), 1e-8, None) * weights
            gradient = design.T @ (weights * (target - probability)) - penalty * beta
            hessian = (design * variance[:, None]).T @ design
            hessian[np.diag_indices_from(hessian)] += penalty
            step_vector = np.linalg.solve(hessian, gradient)
            beta = beta + step_vector
            self.iterations = step + 1
            if np.abs(step_vector).max() < self.tolerance:
                break

        self._intercept = float(beta[0])
        self._weights = beta[1:]
        return self

    def predict_proba(self, features: Any) -> Any:
        """양성 클래스(고잔반)의 확률. **보정되지 않은 점수다.**"""
        import numpy as np

        if self._weights is None:
            raise BaselineError("fit()을 먼저 부르세요.")
        matrix = np.asarray(features, dtype=float)
        eta = ((matrix - self._mean) / self._scale) @ self._weights + self._intercept
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))

    def predict(self, features: Any, threshold: float = 0.5) -> Any:
        import numpy as np

        return (self.predict_proba(features) >= threshold).astype(int)


def clip(predictions: Any, low: float = 0.0, high: float = 1.0) -> Any:
    """서비스가 요구하는 범위로 자른다.

    **지표를 재기 전에 몰래 부르지 않는다.** raw 회귀 출력의 지표와 자른
    뒤의 지표를 나란히 보고한다. 자르는 것으로 좋아지는 부분은 모델이 배운
    것이 아니라 정답의 범위를 알고 있다는 사실에서 온다.
    """
    import numpy as np

    if low > high:
        raise BaselineError(f"범위가 뒤집혔습니다: {low} > {high}")
    return np.clip(np.asarray(predictions, dtype=float), low, high)
