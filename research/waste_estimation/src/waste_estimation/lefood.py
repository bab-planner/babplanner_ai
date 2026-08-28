"""LeFood-Set 어댑터.

공식 출처: <https://data.mendeley.com/datasets/cchsk79jkt/1>
라이선스: CC BY 4.0 (Mendeley Data 공개 API의 `data_licence` 필드에서 확인)

    Bahar, R. et al. "LeFood-Set: Leftovers Food Dataset", Mendeley Data,
    V1, doi: 10.17632/cchsk79jkt.1

**이것은 밥플래너 현장 데이터가 아니다.** 인도네시아 병원 급식이고, 식판
구성·메뉴·촬영 조건이 한국 대학 학식과 다르다. 그래서 여기서 만드는 라벨을
`waste_rate`라고 부르지 않고 **`external_waste_ratio`** 라고 부른다. 계산식이
같아도 다른 데이터의 값이다.

    external_waste_ratio = after_mass_g / before_mass_g

**질량이 식기를 포함하지 않는다는 근거.** 관찰자 등급 7("전부 먹음")인 표본의
식후 질량이 1g 수준까지 내려간다. 식기 무게가 포함되어 있다면 그 값이 접시
무게(수백 g)로 남아 있어야 한다. 즉 이미 tare가 제거된 음식 무게로 보인다.
이것은 관측에 근거한 판단이고, 공식 문서가 tare를 명시하지는 않는다.

**xlsx를 표준 라이브러리로 읽는다.** openpyxl을 새로 넣지 않는다. 읽는 것은
문자열과 숫자뿐이라 서식·수식 처리가 필요 없다.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

#: 공식 배포 정보. 보고서와 동결 기록이 같은 값을 참조한다.
SOURCE_URL = "https://data.mendeley.com/datasets/cchsk79jkt/1"
DOI = "10.17632/cchsk79jkt.1"
VERSION = "1"
LICENSE = "CC BY 4.0"
LICENSE_URL = "http://creativecommons.org/licenses/by/4.0"
CITATION = (
    "LeFood-Set: Leftovers Food Dataset, Mendeley Data V1, "
    "doi:10.17632/cchsk79jkt.1 (CC BY 4.0)"
)

#: 표의 열 순서. 실제 파일에서 읽은 그대로다.
EXPECTED_HEADER = (
    "ID",
    "Name of the food",
    "Image Before Eaten",
    "Weight Before Eaten (g)",
    "Image After Eaten",
    "Weight After Eaten (g)",
    "Visual Estimation by Observer (1-7)",
)

#: 파일명 규약. `{음식범주}_{기록번호}_{원본사진번호}_{bef|aft}.JPG`
FILENAME = re.compile(r"^(?P<category>\d+)_(?P<record>\d+)_(?P<capture>.+)_(?:bef|aft)\.JPG$")


class LeFoodError(ValueError):
    """LeFood-Set을 읽을 수 없다."""


class RejectionReason(str, Enum):
    """표본을 쓰지 않는 이유. 조용히 버리지 않고 세어서 보고한다."""

    #: 표에는 있는데 공개된 이미지에 없다. 524쌍만 배포되었다.
    MISSING_IMAGE = "MISSING_IMAGE"
    #: 분모가 없다.
    NON_POSITIVE_BEFORE = "NON_POSITIVE_BEFORE"
    #: 음수 질량은 측정 오류다.
    NEGATIVE_AFTER = "NEGATIVE_AFTER"
    #: 받은 것보다 많이 남을 수 없다.
    AFTER_EXCEEDS_BEFORE = "AFTER_EXCEEDS_BEFORE"
    #: 숫자로 읽히지 않는다.
    PARSE_ERROR = "PARSE_ERROR"
    #: before/after가 같은 식사가 아니다.
    PAIR_MISMATCH = "PAIR_MISMATCH"


@dataclass(frozen=True)
class LeFoodRecord:
    """유효한 표본 하나.

    external_waste_ratio는 **밥플래너의 waste_rate가 아니다.** 8절의 이름
    분리를 코드에서도 지킨다.
    """

    row_id: str
    food_category_id: str
    food_name: str
    before_image: str
    before_mass_g: float
    after_image: str
    after_mass_g: float
    observer_level: int
    external_waste_ratio: float

    @property
    def group_id(self) -> str:
        """분할 단위.

        **참가자나 회차 식별자가 데이터에 없다.** 파일명의 첫 필드가 음식
        범주이고, 그것이 지금 쓸 수 있는 가장 강한 묶음이다. 없는 참가자를
        발명하지 않는다.

        음식 범주로 묶으면 "한 번도 본 적 없는 음식에서도 되는가"를 평가하게
        된다. RQ-W4가 묻는 것과 같은 축이다.
        """
        return f"food:{self.food_category_id}"


@dataclass(frozen=True)
class LoadReport:
    """읽은 결과와 버린 것들."""

    records: tuple[LeFoodRecord, ...]
    rejections: tuple[tuple[str, RejectionReason], ...]
    total_rows: int

    @property
    def rejection_counts(self) -> dict[str, int]:
        counts = {reason.value: 0 for reason in RejectionReason}
        for _, reason in self.rejections:
            counts[reason.value] += 1
        return counts

    @property
    def food_categories(self) -> int:
        return len({record.food_category_id for record in self.records})


def _read_sheet(xlsx_path: Path) -> list[list[str]]:
    """xlsx의 첫 시트를 문자열 표로 읽는다."""
    if not xlsx_path.exists():
        raise LeFoodError(f"파일이 없습니다: {xlsx_path}")
    with zipfile.ZipFile(xlsx_path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in si.iter(f"{NS}t"))
                for si in root.findall(f"{NS}si")
            ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: list[list[str]] = []
    for row in sheet.iter(f"{NS}row"):
        values: list[str] = []
        for cell in row.findall(f"{NS}c"):
            value = cell.find(f"{NS}v")
            if value is None:
                values.append("")
            elif cell.get("t") == "s":
                values.append(shared[int(value.text)])
            else:
                values.append(value.text or "")
        rows.append(values)
    return rows


def _pair_key(name: str) -> tuple[str, str] | None:
    match = FILENAME.match(name)
    if match is None:
        return None
    return match.group("category"), match.group("record")


def load(root: Path) -> LoadReport:
    """공식 배포본을 읽어 유효한 표본만 돌려준다.

    라벨 생성 관문(W2 §7)을 여기서 통과시킨다.

        before_mass_g > 0
        after_mass_g >= 0
        after_mass_g <= before_mass_g

    그리고 **두 이미지가 실제로 디스크에 있어야 한다.** 표에는 678행이
    있는데 공개된 이미지는 524쌍뿐이다. 표만 믿고 라벨을 만들면 이미지 없는
    표본이 학습 집합에 들어간다.
    """
    rows = _read_sheet(root / "data_original.xlsx")
    if not rows:
        raise LeFoodError("표가 비어 있습니다.")
    header = tuple(value.strip() for value in rows[0][: len(EXPECTED_HEADER)])
    if header != EXPECTED_HEADER:
        raise LeFoodError(
            "열 구성이 예상과 다릅니다. 공식 배포본이 바뀌었는지 확인하세요.\n"
            f"  기대: {EXPECTED_HEADER}\n  실제: {header}"
        )

    on_disk = {path.name for path in root.glob("*.JPG")}
    records: list[LeFoodRecord] = []
    rejections: list[tuple[str, RejectionReason]] = []

    data_rows = [row for row in rows[1:] if any(value.strip() for value in row)]
    for row in data_rows:
        padded = list(row) + [""] * (len(EXPECTED_HEADER) - len(row))
        row_id = padded[0].strip()
        food_name = padded[1].strip()
        before_image = padded[2].strip()
        after_image = padded[4].strip()

        try:
            before_mass = float(padded[3])
            after_mass = float(padded[5])
            observer_level = int(float(padded[6]))
        except ValueError:
            rejections.append((row_id, RejectionReason.PARSE_ERROR))
            continue

        before_key = _pair_key(before_image)
        after_key = _pair_key(after_image)
        if before_key is None or after_key is None or before_key != after_key:
            # 파일명만 보고 짝을 만들지 않는다. 표가 같은 행에 둔 두 파일이
            # 같은 식사의 것인지 접두사로 한 번 더 확인한다.
            rejections.append((row_id, RejectionReason.PAIR_MISMATCH))
            continue

        if before_image not in on_disk or after_image not in on_disk:
            rejections.append((row_id, RejectionReason.MISSING_IMAGE))
            continue
        if before_mass <= 0:
            rejections.append((row_id, RejectionReason.NON_POSITIVE_BEFORE))
            continue
        if after_mass < 0:
            rejections.append((row_id, RejectionReason.NEGATIVE_AFTER))
            continue
        if after_mass > before_mass:
            rejections.append((row_id, RejectionReason.AFTER_EXCEEDS_BEFORE))
            continue

        records.append(
            LeFoodRecord(
                row_id=row_id,
                food_category_id=before_key[0],
                food_name=food_name,
                before_image=before_image,
                before_mass_g=before_mass,
                after_image=after_image,
                after_mass_g=after_mass,
                observer_level=observer_level,
                external_waste_ratio=after_mass / before_mass,
            )
        )

    return LoadReport(
        records=tuple(records),
        rejections=tuple(rejections),
        total_rows=len(data_rows),
    )


def describe(report: LoadReport) -> dict[str, object]:
    """기록에 남길 요약. 원본 이미지는 들어가지 않는다."""
    ratios = sorted(record.external_waste_ratio for record in report.records)
    levels = Counter(record.observer_level for record in report.records)
    categories = Counter(record.food_category_id for record in report.records)

    def quantile(fraction: float) -> float | None:
        if not ratios:
            return None
        return ratios[int(fraction * (len(ratios) - 1))]

    return {
        "source": SOURCE_URL,
        "doi": DOI,
        "version": VERSION,
        "license": LICENSE,
        "totalRows": report.total_rows,
        "usableRecords": len(report.records),
        "rejections": report.rejection_counts,
        "foodCategories": report.food_categories,
        "largestCategory": max(categories.values(), default=0),
        "observerLevels": dict(sorted(levels.items())),
        "externalWasteRatio": {
            "min": ratios[0] if ratios else None,
            "q25": quantile(0.25),
            "median": quantile(0.5),
            "q75": quantile(0.75),
            "max": ratios[-1] if ratios else None,
            "mean": sum(ratios) / len(ratios) if ratios else None,
            "exactlyZero": sum(1 for value in ratios if value == 0.0),
            "atLeast099": sum(1 for value in ratios if value >= 0.99),
        },
    }
