"""공개 데이터 후보 등록부.

CCTV 트랙에서 배운 것을 그대로 가져온다.

    공식 페이지에서 조건 원문을 읽지 못한 데이터는 쓰지 않는다.
    GitHub 저장소 코드의 라이선스를 데이터셋 라이선스로 읽지 않는다.
    2차 출처(블로그, 재업로드, 요약 페이지)를 법적 근거로 쓰지 않는다.

**그리고 이 트랙에만 있는 위험이 하나 더 있다.** 음식 분할 데이터셋의
픽셀 비율을 우리 정답으로 바꾸면 안 된다.

    남은 픽셀 비율  ≠  leftover_edible_mass_g / served_edible_mass_g

같은 부피라도 밀도가 다르고, 같은 면적이라도 두께가 다르다. 두 값을 이어
붙이는 순간 모델이 무엇을 배웠는지 알 수 없게 된다. 그래서 이 등록부는
데이터마다 **질량 기반 정답이 있는지**를 따로 기록하고, 없으면 회귀 정답으로
쓸 수 없다고 표시한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LicenseStatus(str, Enum):
    """조건을 어디까지 확인했는가."""

    #: 공식 배포 페이지에서 조건 원문을 직접 읽었다.
    VERIFIED = "VERIFIED"

    #: 일부만 확인했다. 예를 들어 논문은 공개 라이선스인데 데이터 자체의
    #: 조건은 읽지 못한 경우.
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"

    #: 공식 조건을 읽지 못했다. **쓰지 않는다.**
    UNVERIFIED = "UNVERIFIED"


class DatasetRole(str, Enum):
    """이 데이터를 무엇에 쓸 수 있는가."""

    #: 표현 학습 / 사전학습 / 분할 타당성 확인.
    REPRESENTATION = "REPRESENTATION"

    #: 같은 과제(잔반 정도)의 외부 벤치마크.
    BENCHMARK = "BENCHMARK"

    #: 시연 파이프라인 입력.
    DEMO = "DEMO"

    #: 쓸 수 없다.
    NONE = "NONE"


@dataclass(frozen=True)
class PublicDataset:
    """후보 하나.

    name:            데이터셋 이름
    official_source: 공식 배포/설명 페이지
    license_name:    확인한 라이선스 이름. 못 읽었으면 None
    license_status:  위 확인 수준
    images:          공개된 규모(대략). 모르면 None
    has_mass_ground_truth:  그램 단위 질량 정답이 있는가
    has_before_after:       식전/식후 짝이 있는가
    has_segmentation:       분할 마스크가 있는가
    roles:           허용하는 역할
    domain_gap:      우리 급식소와 무엇이 다른가
    """

    name: str
    official_source: str
    license_name: str | None
    license_status: LicenseStatus
    images: int | None
    has_mass_ground_truth: bool
    has_before_after: bool
    has_segmentation: bool
    roles: tuple[DatasetRole, ...]
    domain_gap: str

    def __post_init__(self) -> None:
        if self.license_status is LicenseStatus.UNVERIFIED and self.roles != (
            DatasetRole.NONE,
        ):
            raise ValueError(
                f"{self.name}: 조건을 확인하지 못한 데이터에 역할을 줄 수 없습니다."
            )
        if self.license_status is not LicenseStatus.UNVERIFIED and not self.license_name:
            raise ValueError(f"{self.name}: 확인했다면 라이선스 이름이 있어야 합니다.")

    @property
    def usable(self) -> bool:
        return self.license_status is not LicenseStatus.UNVERIFIED and bool(
            set(self.roles) - {DatasetRole.NONE}
        )

    @property
    def usable_as_regression_ground_truth(self) -> bool:
        """우리 회귀 정답으로 바로 쓸 수 있는가.

        질량 기반 정답이 없으면 **불가**다. 분할 마스크에서 면적 비율을
        계산해 정답으로 삼는 변환은 하지 않는다.
        """
        return self.usable and self.has_mass_ground_truth


#: 2026-08-28 조사. 각 항목의 근거는 research/waste_estimation/datasets/README.md.
CANDIDATES: tuple[PublicDataset, ...] = (
    PublicDataset(
        name="Nutrition5k",
        official_source="https://github.com/google-research-datasets/Nutrition5k",
        license_name="Creative Commons 4.0",
        license_status=LicenseStatus.VERIFIED,
        images=5006,
        has_mass_ground_truth=True,
        has_before_after=False,
        has_segmentation=False,
        roles=(DatasetRole.REPRESENTATION,),
        domain_gap=(
            "제공된 음식만 있고 먹고 난 뒤가 없다. 카페테리아 식판이 아니라 "
            "개별 접시이며 촬영 장비가 통제되어 있다."
        ),
    ),
    PublicDataset(
        name="LeFood-set",
        official_source="https://data.mendeley.com/datasets/cchsk79jkt/1",
        license_name="CC BY 4.0",
        license_status=LicenseStatus.VERIFIED,
        images=524,
        has_mass_ground_truth=True,
        has_before_after=True,
        has_segmentation=False,
        roles=(DatasetRole.BENCHMARK, DatasetRole.REPRESENTATION),
        domain_gap=(
            "인도네시아 병원 급식이고 34개 음식 범주다. 식판 구성과 메뉴가 "
            "한국 대학 학식과 다르다."
        ),
    ),
    PublicDataset(
        name="FLIC",
        official_source="https://doi.org/10.3390/app16115465",
        license_name=None,
        license_status=LicenseStatus.UNVERIFIED,
        images=401,
        has_mass_ground_truth=True,
        has_before_after=True,
        has_segmentation=True,
        roles=(DatasetRole.NONE,),
        domain_gap=(
            "대학 구내식당 식판이라 과제가 거의 같다. 다만 배포 조건을 공식 "
            "페이지에서 읽지 못했다."
        ),
    ),
    PublicDataset(
        name="UNIMIB2016",
        official_source="http://www.ivl.disco.unimib.it/activities/food-recognition/",
        license_name=None,
        license_status=LicenseStatus.UNVERIFIED,
        images=1027,
        has_mass_ground_truth=False,
        has_before_after=False,
        has_segmentation=True,
        roles=(DatasetRole.NONE,),
        domain_gap=(
            "이탈리아 셀프서비스 식당 식판. 질량 정답이 없고 배포 페이지에 "
            "접속하지 못했다."
        ),
    ),
    PublicDataset(
        name="FoodSeg103",
        official_source="https://xiongweiwu.github.io/foodseg103.html",
        license_name=None,
        license_status=LicenseStatus.UNVERIFIED,
        images=9490,
        has_mass_ground_truth=False,
        has_before_after=False,
        has_segmentation=True,
        roles=(DatasetRole.NONE,),
        domain_gap=(
            "웹 음식 사진의 재료 분할이다. 식판도 잔반도 아니고 공식 페이지에 "
            "라이선스 표기가 없다."
        ),
    ),
)


def usable_candidates() -> tuple[PublicDataset, ...]:
    """지금 쓸 수 있는 것만."""
    return tuple(dataset for dataset in CANDIDATES if dataset.usable)


def regression_ground_truth_candidates() -> tuple[PublicDataset, ...]:
    """우리 회귀 정답으로 쓸 수 있는 것만.

    **여기 있는 데이터도 그대로 우리 정답이 되는 것은 아니다.** 질량 정답이
    있다는 것과 우리 급식소의 잔반률이라는 것은 다른 말이다. 이 목록은
    "면적 비율을 질량비로 바꾸는 변환 없이 다룰 수 있는가"까지만 말한다.
    """
    return tuple(
        dataset for dataset in CANDIDATES if dataset.usable_as_regression_ground_truth
    )
