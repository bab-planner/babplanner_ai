"""정답 라벨의 모양과 지켜야 할 조건.

여기서 정하는 것은 "잔반률이 무엇인가"이다. 추론 모델은 아직 없고, 먼저
무엇을 맞히려는 것인지부터 닫는다.

정답은 **질량비**다::

    waste_rate = leftover_edible_mass_g / served_edible_mass_g

사진에서 음식이 차지하는 면적 비율도, 사람이 눈으로 매긴 추정치도 정답으로
쓰지 않는다. 둘 다 측정하기는 쉽지만 서로 다른 값이고, 섞이는 순간 모델이
무엇을 배웠는지 알 수 없게 된다.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: 직렬화 오차만 허용한다. 측정 오차를 여기서 흡수하지 않는다.
RATE_TOLERANCE = 1e-6

#: 정답 정의의 버전. 정의가 바뀌면 이 값이 바뀐다. 학습된 모델의 버전
#: (model_version)과는 다른 축이다.
CURRENT_LABEL_VERSION = "waste-mass-ratio-v1"


class ServedMassSource(str, Enum):
    """제공량을 어떻게 알아냈는지.

    같은 숫자라도 출처에 따라 믿을 수 있는 정도가 다르다. 섞어 놓으면 나중에
    구분할 수 없으므로 행마다 남긴다.
    """

    #: 그 끼니의 제공량을 실제로 계량했다. 가장 믿을 수 있다.
    MEASURED = "MEASURED"

    #: 표준 배식량을 썼다. 개인별 실제 배식량과 다를 수 있다.
    STANDARD_PORTION = "STANDARD_PORTION"

    #: 사람이 사진을 보고 추정했다. 정답으로 쓰지 않는다.
    VISUAL_ESTIMATE = "VISUAL_ESTIMATE"


#: 주 평가 집합에 넣을 수 있는 출처. 눈대중은 여기 들어오지 않는다.
PRIMARY_BENCHMARK_SOURCES = frozenset({ServedMassSource.MEASURED})


class AnalysisLabel(str, Enum):
    """이미지가 잔반률 추정에 쓸 수 있는 상태인지.

    지급 여부와는 무관하다. 파일이 이미지가 아니거나 열 수 없는 경우는 이
    라벨이 아니라 요청 자체의 오류(HTTP 422)로 다룬다.
    """

    #: 잔반률을 추정할 수 있는 품질이다.
    APPROVED = "APPROVED"

    #: 정상 이미지 파일이지만 잔반량을 판단할 시각 정보가 부족하다.
    REJECTED = "REJECTED"


class RejectionReason(str, Enum):
    """왜 추정할 수 없었는지.

    실제 데이터를 보기 전이라 굵게만 나눈다. 어디에 몰리는지 보고 나서
    쪼개는 편이 낫다.
    """

    NO_RELEVANT_FOOD = "NO_RELEVANT_FOOD"
    SEVERE_BLUR = "SEVERE_BLUR"
    SEVERE_OCCLUSION = "SEVERE_OCCLUSION"
    INVALID_VIEW = "INVALID_VIEW"
    OTHER = "OTHER"


class MealType(str, Enum):
    BREAKFAST = "BREAKFAST"
    LUNCH = "LUNCH"
    DINNER = "DINNER"


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DatasetSample(BaseModel):
    """정답 한 건.

    한 사람이 한 끼에 받은 음식과 남긴 음식을 잰 결과, 그리고 그때 찍은
    사진 하나가 짝을 이룬다.
    """

    model_config = ConfigDict(extra="forbid")

    #: 비식별 식별자. 이름·학번·전화번호·앱 사용자 id를 쓰지 않는다.
    sample_id: str = Field(min_length=1)

    #: 이미지 루트로부터의 상대 경로.
    image_path: str = Field(min_length=1)

    menu_id: int
    site_id: int | None = None
    service_date: str | None = None
    meal_type: MealType | None = None
    captured_at: str | None = None

    #: 분할이 새지 않게 묶는 단위. 같은 사람이 찍은 사진이 학습과 평가에
    #: 나눠 들어가면 성능이 실제보다 좋아 보인다.
    participant_group_id: str | None = None

    #: 한 번의 촬영 회차. 같은 날 같은 자리에서 찍힌 사진은 배경과 조명이
    #: 거의 같아 사실상 같은 문제다.
    capture_session_id: str = Field(min_length=1)

    #: 그램. 문자열이 아니라 숫자로 적는다.
    served_edible_mass_g: float | None = None
    leftover_edible_mass_g: float | None = None
    waste_rate: float | None = None

    served_mass_source: ServedMassSource | None = None

    analysis_label: AnalysisLabel
    rejection_reason: RejectionReason | None = None

    image_sha256: str | None = None
    split: Split | None = None

    #: 정답 정의의 버전.
    label_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_label_invariants(self) -> "DatasetSample":
        if self.analysis_label is AnalysisLabel.APPROVED:
            self._check_approved()
        else:
            self._check_rejected()
        return self

    def _check_approved(self) -> None:
        if self.rejection_reason is not None:
            raise ValueError("승인된 표본에 거절 사유가 있습니다.")
        if self.served_mass_source is None:
            raise ValueError("승인된 표본에는 제공량 출처가 필요합니다.")
        if self.served_edible_mass_g is None or self.leftover_edible_mass_g is None:
            raise ValueError("승인된 표본에는 제공량과 잔반량이 모두 필요합니다.")
        if self.waste_rate is None:
            raise ValueError("승인된 표본에는 잔반률이 필요합니다.")

        served = self.served_edible_mass_g
        leftover = self.leftover_edible_mass_g

        if served <= 0:
            # 0으로 나눌 수 없고, 아무것도 받지 않은 끼니는 잔반률을 말할 수 없다.
            raise ValueError("제공량은 0보다 커야 합니다.")
        if leftover < 0:
            # 음수 잔반량은 측정이나 기록이 틀렸다는 뜻이다. 0으로 깎지 않는다.
            raise ValueError("잔반량은 음수일 수 없습니다.")
        if leftover > served:
            # 받은 것보다 많이 남길 수 없다. 저울 tare나 기록을 다시 봐야 한다.
            raise ValueError("잔반량이 제공량보다 많습니다.")
        if not 0.0 <= self.waste_rate <= 1.0:
            # 범위를 벗어난 값을 잘라서 정상 라벨로 만들지 않는다.
            raise ValueError("잔반률은 0과 1 사이여야 합니다.")

        expected = leftover / served
        if abs(self.waste_rate - expected) > RATE_TOLERANCE:
            raise ValueError("잔반률이 측정한 질량비와 맞지 않습니다.")

    def _check_rejected(self) -> None:
        if self.rejection_reason is None:
            # 사유가 없으면 나중에 왜 버렸는지 되짚을 수 없다.
            raise ValueError("거절된 표본에는 사유가 필요합니다.")
        if self.waste_rate is not None:
            # 추정할 수 없다고 판단한 사진에 정답을 붙이지 않는다. 특히
            # 0으로 채우면 "다 먹었다"는 뜻이 되어 정답이 오염된다.
            raise ValueError("거절된 표본에는 잔반률을 넣지 않습니다.")

    @property
    def group_key_candidates(self) -> tuple[str | None, str]:
        """분할을 묶는 기준. 참가자를 우선하고 없으면 촬영 회차를 쓴다."""
        return self.participant_group_id, self.capture_session_id

    @property
    def is_primary_benchmark(self) -> bool:
        """주 평가 집합에 넣을 수 있는지.

        승인됐고 제공량을 실제로 잰 표본만 해당한다.
        """
        return (
            self.analysis_label is AnalysisLabel.APPROVED
            and self.served_mass_source in PRIMARY_BENCHMARK_SOURCES
        )


class DatasetMetadata(BaseModel):
    """데이터셋 전체에 한 번만 적는 것.

    행마다 반복할 필요가 없는 수집 조건이 여기 온다. 개인정보는 넣지 않는다.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_version: str = Field(min_length=1)
    label_version: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    collection_site: str | None = None

    #: 저울 눈금. 정답의 정밀도 한계를 정하는 값이다.
    scale_resolution_g: float | None = Field(default=None, gt=0)

    protocol_version: str | None = None
    notes: str | None = None
