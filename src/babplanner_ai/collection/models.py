"""현장 기록의 모양.

여기 있는 것은 **원시 기록**이다. 저울에서 읽은 숫자와 그때의 사정을 그대로
적는다. 학습용 정답은 이것을 계산해서 따로 내보낸다. 둘을 같은 파일에 적으면
계산이 틀렸을 때 원래 숫자가 무엇이었는지 알 수 없다.

정답의 뜻과 열 이름은 dataset 쪽 계약을 그대로 쓴다. 여기서 다시 정의하면
두 곳이 갈라진다.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from babplanner_ai.dataset.schema import (
    AnalysisLabel,
    GtExclusionReason,
    MealType,
    RejectionReason,
    ServedMassSource,
    check_gt_exclusion,
)

#: 기록에 둘 수 없는 이름. 개인정보가 새는 가장 흔한 경로가 "일단 넣어 둔
#: 칸"이라, 값을 뒤지기 전에 이름부터 막는다.
FORBIDDEN_FIELDS = frozenset(
    {
        "name",
        "phone",
        "phone_number",
        "email",
        "student_id",
        "studentid",
        "user_id",
        "userid",
        "access_token",
        "accesstoken",
        "refresh_token",
        "refreshtoken",
        "verification_token",
        "verificationtoken",
        "password",
        "operator",
        "operator_name",
    }
)


class MeasurementError(ValueError):
    """측정값이 앞뒤가 맞지 않는다."""


class CollectionSession(BaseModel):
    """파일럿 수집 회차 하나.

    한 번의 파일럿(예: 어느 급식소에서 2주)에 해당한다. 여기 적힌 버전들이
    나중에 내보내는 데이터셋의 버전이 된다.
    """

    model_config = ConfigDict(extra="forbid")

    collection_session_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    label_version: str = Field(min_length=1)
    protocol_version: str = Field(min_length=1)
    started_at: str = Field(min_length=1)
    closed_at: str | None = None

    #: 급식소를 가리키는 짧은 코드. 기관 상세 정보를 적는 칸이 아니다.
    site_code: str | None = None

    #: 저울 눈금. 정답이 가질 수 있는 정밀도의 한계다.
    scale_resolution_g: float = Field(gt=0)

    notes: str | None = None

    @property
    def is_open(self) -> bool:
        return self.closed_at is None


class TareProfile(BaseModel):
    """식기 조합 하나의 빈 무게.

    끼니마다 손으로 다시 적으면 그만큼 잘못 적을 기회가 늘어난다. 한 번 재서
    이름을 붙여 두고 표본에서는 그 이름만 참조한다.
    """

    model_config = ConfigDict(extra="forbid")

    tare_profile_id: str = Field(min_length=1)

    #: 식기 구성. "4칸 스테인리스 식판 + 국그릇" 처럼 적는다. 사람 정보는 넣지 않는다.
    description: str = Field(min_length=1)

    #: 실제로 잰 값들. 대표값만 남기지 않는다. 얼마나 흔들리는지가
    #: 정답의 오차 하한을 말해 준다.
    tare_measurements_g: list[float] = Field(min_length=1)

    measured_at: str = Field(min_length=1)
    notes: str | None = None

    @model_validator(mode="after")
    def _weights_must_be_positive(self) -> "TareProfile":
        for weight in self.tare_measurements_g:
            if weight <= 0:
                raise ValueError("tare 무게는 0보다 커야 합니다.")
        return self

    @property
    def measurement_count(self) -> int:
        return len(self.tare_measurements_g)

    @property
    def tare_weight_g(self) -> float:
        """대표 tare.

        중앙값을 쓴다. 한 번 크게 잘못 읽은 값이 대표값을 끌고 가지 않는다.
        """
        ordered = sorted(self.tare_measurements_g)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2

    @property
    def spread_g(self) -> float:
        """가장 큰 값과 가장 작은 값의 차이."""
        return max(self.tare_measurements_g) - min(self.tare_measurements_g)


class MenuExclusion(BaseModel):
    """메뉴 하나를 정답 데이터에서 뺄지 말지.

    **표본이 아니라 메뉴에 붙는 값이다.** 수집자는 그날 그 메뉴에 대해 한 번만
    정하고, 표본마다 다시 판단하지 않는다. 매번 물으면 같은 메뉴가 어느 날은
    빠지고 어느 날은 들어가서, 나중에 왜 그런지 알 수 없게 된다.

    기록은 덧붙이기만 한다. 같은 메뉴를 다시 선언하면 나중 것이 이기고,
    바꾼 이력은 그대로 남는다.
    """

    model_config = ConfigDict(extra="forbid")

    menu_id: int

    #: 없으면 제외를 해제한다는 뜻이다. 잘못 걸어 둔 것을 되돌리는 길이다.
    gt_exclusion_reason: GtExclusionReason | None = None
    gt_exclusion_note: str | None = None

    declared_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_exclusion(self) -> "MenuExclusion":
        check_gt_exclusion(self.gt_exclusion_reason, self.gt_exclusion_note)
        return self

    @property
    def is_excluded(self) -> bool:
        return self.gt_exclusion_reason is not None


class EntryType(str, Enum):
    SAMPLE = "SAMPLE"
    CORRECTION = "CORRECTION"


class RawMeasurementRecord(BaseModel):
    """표본 한 건의 원시 기록.

    저울에 찍힌 숫자를 그대로 적는다. tare를 뺀 값이나 잔반률은 여기 적지
    않는다. 그것들은 이 값에서 계산되는 것이고, 계산식이 바뀌면 다시
    계산해야 하기 때문이다.
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    collection_session_id: str = Field(min_length=1)
    capture_session_id: str = Field(min_length=1)
    participant_group_id: str | None = None

    menu_id: int
    site_id: int | None = None
    service_date: str | None = None
    meal_type: MealType | None = None

    tare_profile_id: str | None = None

    #: 저울에 올린 그대로의 값. 식기 무게가 포함돼 있다.
    pre_meal_total_weight_g: float | None = None
    post_meal_total_weight_g: float | None = None

    #: 같은 대상을 여러 번 잰 경우. 저울과 기록이 얼마나 흔들리는지 본다.
    pre_meal_measurements_g: list[float] = Field(default_factory=list)
    post_meal_measurements_g: list[float] = Field(default_factory=list)

    #: 표준 배식량을 쓴 경우의 제공량. 개별 계량과 섞이지 않게 칸을 나눈다.
    standard_served_edible_mass_g: float | None = None

    captured_at: str | None = None
    served_mass_source: ServedMassSource | None = None

    #: 정답 데이터로 쓰지 않기로 한 이유. 표본을 기록할 때 그 메뉴의
    #: 선언(`MenuExclusion`)에서 자동으로 옮겨 적는다. 수집자가 표본마다
    #: 고르는 값이 아니다.
    gt_exclusion_reason: GtExclusionReason | None = None
    gt_exclusion_note: str | None = None

    #: 수집 루트 기준 상대 경로.
    raw_image_path: str | None = None
    sanitized_image_path: str | None = None
    sanitized_image_sha256: str | None = None
    raw_image_sha256: str | None = None

    analysis_label: AnalysisLabel
    rejection_reason: RejectionReason | None = None

    revision: int = Field(default=1, ge=1)
    entry_type: EntryType = EntryType.SAMPLE
    correction_note: str | None = None
    created_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_shape(self) -> "RawMeasurementRecord":
        check_gt_exclusion(self.gt_exclusion_reason, self.gt_exclusion_note)
        if self.analysis_label is AnalysisLabel.APPROVED:
            if self.rejection_reason is not None:
                raise ValueError("승인된 기록에 거절 사유가 있습니다.")
            if self.served_mass_source is None:
                raise ValueError("승인된 기록에는 제공량 출처가 필요합니다.")
            if self.served_mass_source is ServedMassSource.VISUAL_ESTIMATE:
                # 사람이 눈으로 매긴 값은 정답이 아니다. 수집 도구의 정답
                # 경로로 들어오지 못하게 여기서 막는다.
                raise ValueError(
                    "눈대중 추정치는 정답 경로로 수집하지 않습니다. "
                    "사람 기준선을 모으려면 별도 도구를 씁니다."
                )
        elif self.rejection_reason is None:
            raise ValueError("거절된 기록에는 사유가 필요합니다.")
        return self


def check_no_forbidden_fields(model: type[BaseModel]) -> None:
    """개인정보처럼 보이는 칸이 생기지 않았는지 본다."""
    present = {name.lower() for name in model.model_fields} & FORBIDDEN_FIELDS
    if present:
        raise ValueError(
            f"{model.__name__} 에 두면 안 되는 칸이 있습니다: "
            + ", ".join(sorted(present))
        )
