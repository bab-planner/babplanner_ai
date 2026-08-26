"""저울에서 읽은 숫자를 정답으로 바꾸는 계산.

식은 dataset 계약 그대로다::

    served   = pre_meal_total  - tare
    leftover = post_meal_total - tare
    waste_rate = leftover / served

앞뒤가 맞지 않는 값은 조용히 고치지 않는다. -0.02를 0으로, 1.04를 1로
깎으면 그 표본은 정상처럼 보이지만 안에는 잘못된 측정이 그대로 남는다.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from babplanner_ai.collection.models import (
    MeasurementError,
    RawMeasurementRecord,
    TareProfile,
)
from babplanner_ai.dataset.schema import AnalysisLabel, ServedMassSource


@dataclass(frozen=True)
class ComputedMass:
    served_edible_mass_g: float
    leftover_edible_mass_g: float
    waste_rate: float


def compute_mass(
    record: RawMeasurementRecord, tare: TareProfile | None
) -> ComputedMass:
    """한 기록의 가식부 중량과 잔반률.

    거절된 기록에는 쓰지 않는다. 추정할 수 없다고 판단한 사진에 정답을
    붙이지 않기 위해서다.
    """
    if record.analysis_label is not AnalysisLabel.APPROVED:
        raise MeasurementError("거절된 기록에는 잔반률을 계산하지 않습니다.")

    source = record.served_mass_source
    if source is ServedMassSource.MEASURED:
        served, leftover = _from_scale(record, tare)
    elif source is ServedMassSource.STANDARD_PORTION:
        served, leftover = _from_standard_portion(record, tare)
    else:
        raise MeasurementError(f"정답으로 쓸 수 없는 제공량 출처입니다: {source}")

    _check(served, leftover)
    return ComputedMass(served, leftover, leftover / served)


def _require_tare(tare: TareProfile | None) -> float:
    if tare is None:
        raise MeasurementError("tare 프로파일이 필요합니다.")
    return tare.tare_weight_g


def _from_scale(
    record: RawMeasurementRecord, tare: TareProfile | None
) -> tuple[float, float]:
    tare_weight = _require_tare(tare)
    if record.pre_meal_total_weight_g is None:
        raise MeasurementError("배식 후 전체 중량이 없습니다.")
    if record.post_meal_total_weight_g is None:
        raise MeasurementError("식사 후 전체 중량이 없습니다.")

    if record.pre_meal_total_weight_g <= tare_weight:
        raise MeasurementError(
            "배식 후 중량이 식기 무게보다 크지 않습니다. "
            "tare 프로파일이 맞는지, 저울을 다시 읽었는지 확인하세요."
        )
    if record.post_meal_total_weight_g < tare_weight:
        raise MeasurementError(
            "식사 후 중량이 식기 무게보다 작습니다. "
            "다른 식기를 쓴 것은 아닌지 확인하세요."
        )

    return (
        record.pre_meal_total_weight_g - tare_weight,
        record.post_meal_total_weight_g - tare_weight,
    )


def _from_standard_portion(
    record: RawMeasurementRecord, tare: TareProfile | None
) -> tuple[float, float]:
    """제공량은 표준 배식량, 잔반량은 실제 계량.

    제공량이 개인별 실제 배식량과 다를 수 있으므로 이 표본은 주 평가
    집합에 들어가지 않는다. 그 판단은 dataset 계약이 한다.
    """
    if record.standard_served_edible_mass_g is None:
        raise MeasurementError("표준 배식량이 없습니다.")
    if record.post_meal_total_weight_g is None:
        raise MeasurementError("식사 후 전체 중량이 없습니다.")

    tare_weight = _require_tare(tare)
    if record.post_meal_total_weight_g < tare_weight:
        raise MeasurementError("식사 후 중량이 식기 무게보다 작습니다.")

    return (
        record.standard_served_edible_mass_g,
        record.post_meal_total_weight_g - tare_weight,
    )


def _check(served: float, leftover: float) -> None:
    if served <= 0:
        raise MeasurementError("제공량이 0 이하입니다.")
    if leftover < 0:
        raise MeasurementError("잔반량이 음수입니다.")
    if leftover > served:
        raise MeasurementError(
            "잔반량이 제공량보다 많습니다. 두 계량값과 tare를 다시 확인하세요."
        )


@dataclass(frozen=True)
class RepeatStats:
    """같은 대상을 여러 번 잰 결과.

    허용 오차를 여기서 정하지 않는다. 실제 산포를 보기 전에 정하면 그 숫자가
    근거 없이 굳는다.
    """

    count: int
    minimum: float
    maximum: float
    mean: float
    median: float

    @property
    def spread(self) -> float:
        return self.maximum - self.minimum


def repeat_stats(values: list[float]) -> RepeatStats | None:
    if not values:
        return None
    return RepeatStats(
        count=len(values),
        minimum=min(values),
        maximum=max(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
    )
