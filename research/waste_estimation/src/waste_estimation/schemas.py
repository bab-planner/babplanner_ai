"""연구용 표본 표현.

**서비스 스키마를 대체하지 않는다.** 정답의 정의와 매니페스트 열은 이미
`src/babplanner_ai/dataset/schema.py`와 `docs/waste_dataset_spec.md`에서
닫혀 있다. 여기 있는 것은 그것을 연구 코드가 다루기 쉬운 모양으로 옮긴
얇은 표현이고, 공개 데이터처럼 서비스 매니페스트에 없는 출처도 같은 자리에
놓을 수 있게 하려는 것이다.

**정답은 질량비다.**

    waste_rate = leftover_edible_mass_g / served_edible_mass_g

이 정의는 `docs/waste_ground_truth_protocol.md`에 있다. 여기서 새로 만들지
않고 그대로 따른다. 사진에서 음식이 차지하는 면적 비율도, 사람이 눈으로
매긴 추정치도 이 값이 아니다.

**개인정보를 담지 않는다.** 이름, 학번, 전화번호, 계정 식별자는 이 자료
구조에 들어올 자리가 없고, 그 사실을 테스트가 필드 이름으로 확인한다.
분할 단위로 쓰는 `group_id`는 익명 키다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WasteSchemaError(ValueError):
    """연구 표본을 만들 수 없다."""


#: 이 자료 구조에 있어서는 안 되는 필드 이름. 테스트가 이 목록으로 검사한다.
FORBIDDEN_FIELD_WORDS = (
    "name",
    "phone",
    "student",
    "email",
    "account",
    "user_id",
    "face",
    "address",
    "birth",
)

#: 정답 정의의 버전. 서비스 매니페스트의 `label_version`과 같은 값을 쓴다.
LABEL_VERSION = "waste-mass-ratio-v1"


class SampleSource(str, Enum):
    """이 표본이 어디서 왔는가.

    출처를 섞으면 나중에 어떤 숫자가 어느 데이터에서 나온 것인지 알 수 없다.
    공개 데이터로 만든 결과를 급식소 성능으로 읽는 일을 막는다.
    """

    #: 밥플래너가 직접 수집한 파일럿 데이터. 저울 정답이 있다.
    #: **사람이 실제로 먹고 남긴 것**을 관찰한 표본이다.
    PILOT = "pilot"

    #: 사람이 먹지 않았다. 준비한 음식의 일부를 정해진 비율로 덜어 내
    #: 잔반 상태를 **인위적으로 만든** 표본이다. 저울 정답은 있다.
    #:
    #: `PILOT`과 반드시 구분한다. 둘을 같은 이름으로 부르면 "실제 이용자의
    #: 식습관"이라고 잘못 읽히고, 그건 하지 않은 관찰을 했다고 말하는 것이다.
    #: 촬영 조건 전이(domain gap)를 재는 데는 쓸 수 있지만, 사람들이 실제로
    #: 얼마나 남기는지에 대해서는 **아무것도 말하지 못한다.**
    CONTROLLED_STAGED = "controlled_staged_pilot"

    #: 공개 데이터셋. 정답의 정의가 우리와 다를 수 있다.
    PUBLIC = "public"

    #: 도구가 도는지 확인하려고 만든 합성 표본. **학습 데이터가 아니다.**
    SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class WasteSample:
    """표본 하나.

    sample_id:  비식별 식별자.
    source:     출처. 학습/평가에서 섞였는지 확인할 수 있어야 한다.
    group_id:   분할 단위. 같은 사람/회차가 학습과 평가에 나뉘지 않게 한다.
    waste_rate: 질량비 정답. 없을 수 있다(라벨이 아직 없거나 거절된 표본).
    image_reference: 이미지 위치. 저장소 밖 경로일 수 있다.
    """

    sample_id: str
    source: SampleSource
    group_id: str
    waste_rate: float | None = None
    image_reference: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise WasteSchemaError("sample_id가 비어 있습니다.")
        if not self.group_id.strip():
            raise WasteSchemaError(
                "group_id가 비어 있습니다. 분할 단위가 없으면 누수를 막을 수 "
                "없습니다."
            )
        if self.waste_rate is not None:
            validate_waste_rate(self.waste_rate)

    @property
    def is_labeled(self) -> bool:
        return self.waste_rate is not None


def validate_waste_rate(value: float) -> float:
    """질량비가 정의를 만족하는지 본다.

    범위 밖 값을 잘라서 통과시키지 않는다. 1.2가 들어왔다면 측정이나 기록이
    틀린 것이고, 1.0으로 고쳐 학습에 넣으면 그 오류가 조용히 정답이 된다.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WasteSchemaError(f"waste_rate는 숫자여야 합니다: {value!r}")
    if value != value:  # NaN
        raise WasteSchemaError("waste_rate가 NaN입니다.")
    if not 0.0 <= float(value) <= 1.0:
        raise WasteSchemaError(f"waste_rate는 0~1이어야 합니다: {value}")
    return float(value)


def waste_rate_from_mass(
    served_edible_mass_g: float, leftover_edible_mass_g: float
) -> float:
    """두 무게에서 정답을 만든다.

    `docs/waste_ground_truth_protocol.md`의 정의 그대로다. 여기서 규칙을
    새로 만들지 않는다.

    거부하는 경우도 그 문서에 있다.

        served <= 0          분모가 없다
        leftover < 0         음수 질량은 측정 오류다
        leftover > served    받은 것보다 많이 남길 수 없다
    """
    for name, value in (
        ("served_edible_mass_g", served_edible_mass_g),
        ("leftover_edible_mass_g", leftover_edible_mass_g),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WasteSchemaError(f"{name}은(는) 숫자여야 합니다: {value!r}")
        if value != value:
            raise WasteSchemaError(f"{name}이(가) NaN입니다.")

    if served_edible_mass_g <= 0:
        raise WasteSchemaError(
            f"served_edible_mass_g는 0보다 커야 합니다: {served_edible_mass_g}"
        )
    if leftover_edible_mass_g < 0:
        raise WasteSchemaError(
            f"leftover_edible_mass_g는 0 이상이어야 합니다: {leftover_edible_mass_g}"
        )
    if leftover_edible_mass_g > served_edible_mass_g:
        raise WasteSchemaError(
            "leftover가 served보다 클 수 없습니다: "
            f"{leftover_edible_mass_g} > {served_edible_mass_g}"
        )
    return leftover_edible_mass_g / served_edible_mass_g
