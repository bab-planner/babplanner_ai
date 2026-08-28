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

    def fit(self, features: Any, targets: Any) -> "Ridge":
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

        self._mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        # 상수 특징은 나누면 폭발한다. 1로 두면 표준화 뒤 0이 된다.
        scale[scale < 1e-8] = 1.0
        self._scale = scale
        centered = (matrix - self._mean) / scale

        self._intercept = float(target.mean())
        residual = target - self._intercept
        gram = centered.T @ centered
        gram[np.diag_indices_from(gram)] += self.alpha
        self._weights = np.linalg.solve(gram, centered.T @ residual)
        return self

    def predict(self, features: Any) -> Any:
        import numpy as np

        if self._weights is None:
            raise BaselineError("fit()을 먼저 부르세요.")
        matrix = np.asarray(features, dtype=float)
        centered = (matrix - self._mean) / self._scale
        return centered @ self._weights + self._intercept


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
