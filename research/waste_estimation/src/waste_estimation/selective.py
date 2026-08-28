"""선택적 예측(selective prediction) 지표.

이 트랙의 핵심 질문은 "얼마나 맞히는가"가 아니라 **"틀릴 때를 미리 아는가"**
다. 잔반률로 포인트가 나가므로, 틀린 값을 자신 있게 내보내는 것이 값을 안
내보내는 것보다 나쁘다.

그래서 축이 둘이다.

    coverage         전체 중 몇 %를 APPROVE 했는가
    selective risk   APPROVE 한 것들의 오차 (여기서는 MAE)

전체 MAE 하나로는 모델의 좋고 나쁨을 말할 수 없다. 90%를 거절하고 남은
10%에서 MAE 0.02를 내는 모델과, 전부 받아 MAE 0.12를 내는 모델 중 무엇이
나은지는 **운영 정책**이 정한다. 연구가 낼 수 있는 것은 그 둘의 관계
곡선이다.

**confidence를 정확도 확률로 부르지 않는다.** 여기 들어오는 confidence는
모델이 낸 점수일 뿐이고, "0.9라고 말한 예측의 90%가 맞다"는 보정
(calibration)은 별개의 검증이다. CCTV 트랙에서 배운 것과 같다 — raw model
confidence는 calibrated correctness probability가 아니다.

**임계값을 여기서 정하지 않는다.** 이 모듈은 곡선을 그릴 뿐이고, 어디서
자를지는 운영 요구를 보고 나중에 정한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from waste_estimation.metrics import MetricError, mean_absolute_error


@dataclass(frozen=True)
class SelectivePoint:
    """임계값 하나에서의 (coverage, risk).

    threshold:  이 값 이상인 confidence만 APPROVE 했다.
    coverage:   APPROVE 비율. 0~1.
    risk:       APPROVE 한 표본의 MAE. 하나도 없으면 None.
    accepted:   APPROVE 한 표본 수.
    """

    threshold: float
    coverage: float
    risk: float | None
    accepted: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "threshold": self.threshold,
            "coverage": self.coverage,
            "risk": self.risk,
            "accepted": self.accepted,
        }


def _checked(
    actual: Sequence[float],
    predicted: Sequence[float],
    confidence: Sequence[float],
) -> None:
    if not (len(actual) == len(predicted) == len(confidence)):
        raise MetricError(
            "길이가 다릅니다: "
            f"정답 {len(actual)}, 예측 {len(predicted)}, confidence {len(confidence)}"
        )
    if not actual:
        raise MetricError("입력이 비어 있습니다.")
    for index, value in enumerate(confidence):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MetricError(f"{index}번째 confidence가 숫자가 아닙니다: {value!r}")
        if value != value:
            raise MetricError(f"{index}번째 confidence가 NaN입니다.")


def coverage_at(confidence: Sequence[float], threshold: float) -> float:
    """이 임계값에서 몇 %를 APPROVE 하는가."""
    if not confidence:
        raise MetricError("입력이 비어 있습니다.")
    accepted = sum(1 for value in confidence if value >= threshold)
    return accepted / len(confidence)


def selective_risk(
    actual: Sequence[float],
    predicted: Sequence[float],
    confidence: Sequence[float],
    threshold: float,
) -> float | None:
    """APPROVE 한 표본의 MAE.

    **하나도 APPROVE 하지 않았으면 None이다.** 0을 돌려주면 아무것도 내지
    않은 모델이 완벽해 보인다. 그것은 이 지표가 가장 하면 안 되는 거짓말이다.
    """
    _checked(actual, predicted, confidence)
    pairs = [
        (truth, guess)
        for truth, guess, score in zip(actual, predicted, confidence)
        if score >= threshold
    ]
    if not pairs:
        return None
    return mean_absolute_error([t for t, _ in pairs], [g for _, g in pairs])


def coverage_risk_curve(
    actual: Sequence[float],
    predicted: Sequence[float],
    confidence: Sequence[float],
    thresholds: Sequence[float] | None = None,
) -> list[SelectivePoint]:
    """임계값을 훑으며 (coverage, risk)를 낸다.

    thresholds를 주지 않으면 **관측된 confidence 값 자체**를 임계값으로 쓴다.
    임의의 격자를 만들면 데이터에 없는 지점이 곡선에 들어가고, 표본이 적을
    때 그 지점들이 곡선 모양을 정한다.

    coverage 내림차순으로 돌려준다. 곡선을 그대로 읽으면 "얼마나 버리면
    오차가 얼마까지 내려가는가"가 된다.
    """
    _checked(actual, predicted, confidence)
    if thresholds is None:
        candidates = sorted(set(float(value) for value in confidence))
    else:
        candidates = sorted(set(float(value) for value in thresholds))
    if not candidates:
        raise MetricError("임계값이 비어 있습니다.")

    points = [
        SelectivePoint(
            threshold=threshold,
            coverage=coverage_at(confidence, threshold),
            risk=selective_risk(actual, predicted, confidence, threshold),
            accepted=sum(1 for value in confidence if value >= threshold),
        )
        for threshold in candidates
    ]
    points.sort(key=lambda point: (-point.coverage, point.threshold))
    return points


def risk_at_coverage(
    points: Sequence[SelectivePoint], target_coverage: float
) -> SelectivePoint | None:
    """"최소 이만큼은 APPROVE 한다"는 조건에서의 위험.

    운영은 보통 이렇게 묻는다 — "80%는 자동 처리하고 싶다. 그때 오차는?"
    조건을 만족하는 점 중 위험이 가장 낮은 것을 돌려준다. 만족하는 점이
    없으면 None이다. **가장 가까운 점으로 대신하지 않는다.**
    """
    if not 0.0 < target_coverage <= 1.0:
        raise MetricError(f"target_coverage는 0 초과 1 이하여야 합니다: {target_coverage}")
    usable = [
        point
        for point in points
        if point.coverage >= target_coverage and point.risk is not None
    ]
    if not usable:
        return None
    return min(usable, key=lambda point: (point.risk, -point.coverage))
