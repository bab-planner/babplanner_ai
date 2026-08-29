"""통제된 연출 파일럿 — 매니페스트, 관문, 반복 계량.

**이 데이터는 사람이 먹고 남긴 것이 아니다.**

준비한 음식의 일부를 정해진 비율로 덜어 내 잔반 상태를 인위적으로 만든다.
참가자를 모으지 않고, 식사 행동을 관찰하지 않고, 얼굴을 찍지 않는다.

그래서 이 표본으로 답할 수 있는 질문은 하나뿐이다.

    LeFood에서 만든 모델이 **밥플래너 시연과 비슷한 촬영 조건**에서도
    최소한의 신호를 보이는가?

답할 수 **없는** 질문은 이것이다.

    사람들이 실제로 얼마나 남기는가?

두 질문을 섞으면 하지 않은 관찰을 했다고 말하게 된다. `SampleSource`에
`CONTROLLED_STAGED`를 따로 둔 이유이고, 이 모듈의 함수들이 그 값만 받는
이유다.

정답 정의는 W1 그대로다. 여기서 새로 만들지 않는다.

    waste_rate = leftover_edible_mass_g / served_edible_mass_g
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .schemas import (
    FORBIDDEN_FIELD_WORDS,
    SampleSource,
    WasteSchemaError,
    waste_rate_from_mass,
)


class PilotError(ValueError):
    """파일럿 매니페스트를 읽을 수 없다."""


#: 매니페스트에 반드시 있어야 하는 열.
REQUIRED_COLUMNS = (
    "sample_id",
    "source",
    "food_group",
    "capture_session_id",
    "image_file",
    "served_edible_mass_g",
    "leftover_edible_mass_g",
)

#: 있으면 쓰고 없어도 되는 열. 개인 식별과 무관한 촬영 조건만.
OPTIONAL_COLUMNS = (
    "waste_rate",
    "device_model",
    "resolution",
    "orientation",
    "lighting",
    "repeat_index",
    "note",
)

MANIFEST_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

#: 이 파일럿에서 유일하게 허용하는 출처.
ALLOWED_SOURCE = SampleSource.CONTROLLED_STAGED


class PilotRejection(str, Enum):
    """표본을 뺀 이유. 조용히 버리지 않는다."""

    MISSING_COLUMN = "MISSING_COLUMN"
    FORBIDDEN_FIELD = "FORBIDDEN_FIELD"
    WRONG_SOURCE = "WRONG_SOURCE"
    PARSE_ERROR = "PARSE_ERROR"
    NON_POSITIVE_SERVED = "NON_POSITIVE_SERVED"
    NEGATIVE_LEFTOVER = "NEGATIVE_LEFTOVER"
    LEFTOVER_EXCEEDS_SERVED = "LEFTOVER_EXCEEDS_SERVED"
    WASTE_RATE_MISMATCH = "WASTE_RATE_MISMATCH"
    MISSING_IMAGE = "MISSING_IMAGE"
    DUPLICATE_SAMPLE_ID = "DUPLICATE_SAMPLE_ID"
    DUPLICATE_IMAGE = "DUPLICATE_IMAGE"
    ABSOLUTE_PATH = "ABSOLUTE_PATH"


#: 기록된 waste_rate와 질량에서 계산한 값이 이보다 더 벌어지면 거절한다.
#: 반올림 오차는 통과시키되, 다른 표본의 값을 잘못 옮겨 적은 것은 잡는다.
WASTE_RATE_TOLERANCE = 5e-3


@dataclass(frozen=True)
class PilotSample:
    """연출 표본 하나. **사람 정보가 들어갈 자리가 없다.**"""

    sample_id: str
    food_group: str
    capture_session_id: str
    image_file: str
    served_edible_mass_g: float
    leftover_edible_mass_g: float
    repeat_index: int = 0

    @property
    def waste_rate(self) -> float:
        return waste_rate_from_mass(
            self.served_edible_mass_g, self.leftover_edible_mass_g
        )

    @property
    def source(self) -> SampleSource:
        return ALLOWED_SOURCE


def is_absolute_reference(image_file: str) -> bool:
    """POSIX·Windows 어느 쪽 절대 경로든 잡는다.

    매니페스트는 어느 기계에서 쓰였는지 모른다. 한쪽 규칙만 보면 다른 쪽
    절대 경로가 그대로 저장소에 들어간다.
    """
    value = image_file.strip()
    if not value:
        return False
    if value[0] in "/\\":
        return True
    if len(value) >= 2 and value[1] == ":" and value[0].isalpha():
        return True
    return PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()


def _forbidden_columns(columns: list[str]) -> list[str]:
    """개인 식별 냄새가 나는 열 이름을 찾는다."""
    return [
        column
        for column in columns
        if any(word in column.lower() for word in FORBIDDEN_FIELD_WORDS)
    ]


def load_manifest(
    path: Path, images_root: Path | None = None
) -> tuple[list[PilotSample], list[dict]]:
    """매니페스트를 읽고 **거절 사유와 함께** 돌려준다.

    거절을 조용히 하지 않는 이유는 W2에서 배웠다 — 라벨 관문이 154행을
    걸렀는데, 그 수를 세지 않았다면 데이터가 그만큼 있다고 착각했을 것이다.
    """
    if not path.exists():
        raise PilotError(f"매니페스트가 없습니다: {path}")

    rejections: list[dict] = []
    samples: list[PilotSample] = []
    # 반복 계량 행은 같은 sample_id를 **의도적으로** 공유한다. 그래서 중복
    # 판정 키는 (sample_id, repeat_index)다. sample_id만 보면 §5의 반복
    # 계량 절차가 통째로 거절된다.
    seen_keys: set[tuple[str, int]] = set()
    seen_images: dict[str, str] = {}

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        missing = [name for name in REQUIRED_COLUMNS if name not in columns]
        if missing:
            raise PilotError(f"필수 열이 없습니다: {missing}")
        forbidden = _forbidden_columns(columns)
        if forbidden:
            raise PilotError(
                f"개인 식별 가능성이 있는 열은 둘 수 없습니다: {forbidden}"
            )

        for row in reader:
            sample_id = (row.get("sample_id") or "").strip()

            def reject(reason: PilotRejection, detail: str = "") -> None:
                rejections.append(
                    {
                        "sampleId": sample_id,
                        "reason": reason.value,
                        "detail": detail,
                    }
                )

            if not sample_id:
                reject(PilotRejection.PARSE_ERROR, "sample_id가 비어 있습니다")
                continue
            if (row.get("source") or "").strip() != ALLOWED_SOURCE.value:
                reject(
                    PilotRejection.WRONG_SOURCE,
                    f"source는 {ALLOWED_SOURCE.value}여야 합니다",
                )
                continue

            image_file = (row.get("image_file") or "").strip()
            if not image_file:
                reject(PilotRejection.PARSE_ERROR, "image_file이 비어 있습니다")
                continue
            # 절대 경로는 개인 폴더 구조를 그대로 노출한다.
            # **플랫폼에 의존하지 않게 직접 본다.** Windows에서
            # `Path("/home/me/x.jpg").is_absolute()`는 False라서, 그것만
            # 믿으면 POSIX 절대 경로가 그대로 통과한다.
            if is_absolute_reference(image_file):
                reject(PilotRejection.ABSOLUTE_PATH, image_file)
                continue

            try:
                served = float(row["served_edible_mass_g"])
                leftover = float(row["leftover_edible_mass_g"])
                repeat_index = int(row.get("repeat_index") or 0)
            except (TypeError, ValueError) as error:
                reject(PilotRejection.PARSE_ERROR, str(error))
                continue

            if repeat_index < 0:
                reject(PilotRejection.PARSE_ERROR, f"repeat_index {repeat_index}")
                continue
            if (sample_id, repeat_index) in seen_keys:
                reject(
                    PilotRejection.DUPLICATE_SAMPLE_ID,
                    f"repeat_index {repeat_index}",
                )
                continue

            if served <= 0:
                reject(PilotRejection.NON_POSITIVE_SERVED, f"{served}")
                continue
            if leftover < 0:
                reject(PilotRejection.NEGATIVE_LEFTOVER, f"{leftover}")
                continue
            if leftover > served:
                reject(
                    PilotRejection.LEFTOVER_EXCEEDS_SERVED,
                    f"{leftover} > {served}",
                )
                continue

            computed = leftover / served
            recorded = (row.get("waste_rate") or "").strip()
            if recorded:
                try:
                    if abs(float(recorded) - computed) > WASTE_RATE_TOLERANCE:
                        reject(
                            PilotRejection.WASTE_RATE_MISMATCH,
                            f"기록 {recorded} vs 계산 {computed:.4f}",
                        )
                        continue
                except ValueError as error:
                    reject(PilotRejection.PARSE_ERROR, str(error))
                    continue

            if images_root is not None:
                image_path = images_root / image_file
                if not image_path.exists():
                    reject(PilotRejection.MISSING_IMAGE, image_file)
                    continue
                # 같은 사진의 재사용은 **반복 계량 행에서만** 정상이다.
                # 그 외의 중복은 같은 접시를 두 표본으로 등록한 것이다.
                if repeat_index == 0:
                    digest = file_digest(image_path)
                    if digest in seen_images:
                        reject(
                            PilotRejection.DUPLICATE_IMAGE,
                            f"{image_file} == {seen_images[digest]}",
                        )
                        continue
                    seen_images[digest] = image_file

            seen_keys.add((sample_id, repeat_index))
            samples.append(
                PilotSample(
                    sample_id=sample_id,
                    food_group=(row.get("food_group") or "").strip(),
                    capture_session_id=(row.get("capture_session_id") or "").strip(),
                    image_file=image_file,
                    served_edible_mass_g=served,
                    leftover_edible_mass_g=leftover,
                    repeat_index=repeat_index,
                )
            )

    for sample in samples:
        if not sample.food_group:
            raise WasteSchemaError(
                f"{sample.sample_id}: food_group이 비어 있습니다. 범주별 분석을 "
                "할 수 없습니다."
            )
    return samples, rejections


def file_digest(path: Path) -> str:
    """파일 내용의 SHA-256. 같은 사진을 두 번 등록했는지 본다."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- 수용 관문 ---------------------------------------------------------------
#
# **결과를 보기 전에 정한 값이다.** 표본이 모자란 채로 평가하면 작은 숫자가
# 성능처럼 인용되기 시작한다. 그럴 바에는 평가를 안 하는 게 낫다.

MIN_VALID_SAMPLES = 30
MIN_FOOD_GROUPS = 3
MIN_HIGH_WASTE = 6
MAX_INVALID = 0
MAX_DUPLICATES = 0

#: 고잔반 기준은 W4/W5와 같다. 여기서 새로 만들지 않는다.
HIGH_WASTE_THRESHOLD = 0.8

#: 구간도 W2가 고정한 등간격 5구간 그대로.
BIN_EDGES = (0.2, 0.4, 0.6, 0.8)


def bin_counts(samples: list[PilotSample]) -> list[int]:
    """구간별 표본 수. 왼쪽 닫힘(W5 규약)."""
    counts = [0] * (len(BIN_EDGES) + 1)
    for sample in samples:
        index = sum(1 for edge in BIN_EDGES if sample.waste_rate >= edge)
        counts[index] += 1
    return counts


def evaluate_gate(samples: list[PilotSample], rejections: list[dict]) -> dict:
    """평가를 시작해도 되는지 판정한다.

    통과하지 못하면 **모델 평가를 하지 않는다.** 표본 12개로 잰 MAE를
    보고서에 적으면, 그 숫자는 반드시 성능으로 인용된다.
    """
    unique = {sample.sample_id for sample in samples}
    groups = {sample.food_group for sample in samples}
    high = [s for s in samples if s.waste_rate >= HIGH_WASTE_THRESHOLD]
    duplicates = [
        item
        for item in rejections
        if item["reason"]
        in (PilotRejection.DUPLICATE_IMAGE.value, PilotRejection.DUPLICATE_SAMPLE_ID.value)
    ]
    invalid = [item for item in rejections if item not in duplicates]

    checks = {
        "validSamples": (len(unique), MIN_VALID_SAMPLES, len(unique) >= MIN_VALID_SAMPLES),
        "foodGroups": (len(groups), MIN_FOOD_GROUPS, len(groups) >= MIN_FOOD_GROUPS),
        "highWasteSamples": (len(high), MIN_HIGH_WASTE, len(high) >= MIN_HIGH_WASTE),
        "invalidAfterExclusion": (len(invalid), MAX_INVALID, len(invalid) <= MAX_INVALID),
        "duplicates": (len(duplicates), MAX_DUPLICATES, len(duplicates) <= MAX_DUPLICATES),
    }
    passed = all(result for _, _, result in checks.values())
    return {
        "passed": passed,
        "status": "PILOT_READY" if passed else "PILOT_INSUFFICIENT",
        "checks": {
            name: {"actual": actual, "required": required, "passed": result}
            for name, (actual, required, result) in checks.items()
        },
        "binCounts": bin_counts(samples),
        "note": (
            "통과하지 못하면 모델을 평가하지 않는다. 작은 표본의 숫자를 "
            "성능처럼 인용하지 않기 위해서다."
        ),
    }


# --- 반복 계량 ---------------------------------------------------------------


def repeatability(measurements_g: Any, served_edible_mass_g: float) -> dict:
    """같은 상태를 여러 번 잰 값의 흩어짐.

    **모델 MAE와 반드시 나란히 읽는다.** 저울과 절차가 만드는 흔들림보다
    모델 오차가 작을 수는 없다. 그 바닥을 모르면 "MAE 0.05"가 좋은 값인지
    측정 잡음인지 알 수 없다.

    그램과 잔반률 두 단위로 돌려주는 이유도 같다 — 5 g의 흔들림은 300 g
    급식에서 0.017이지만 60 g 반찬에서는 0.083이다.
    """
    values = [float(value) for value in measurements_g]
    if len(values) < 2:
        raise PilotError("반복 계량은 2회 이상이어야 합니다.")
    if served_edible_mass_g <= 0:
        raise PilotError(
            f"served_edible_mass_g는 0보다 커야 합니다: {served_edible_mass_g}"
        )
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    deviation = variance**0.5
    spread = max(values) - min(values)
    return {
        "count": len(values),
        "meanG": mean,
        "stdG": deviation,
        "rangeG": spread,
        "meanWasteRate": mean / served_edible_mass_g,
        "stdWasteRate": deviation / served_edible_mass_g,
        "rangeWasteRate": spread / served_edible_mass_g,
    }


def summarize_repeatability(entries: list[dict]) -> dict:
    """여러 표본의 반복 계량을 한 줄로. 이것이 측정 오차의 바닥이다."""
    if not entries:
        raise PilotError("반복 계량 기록이 없습니다.")
    rates = [entry["stdWasteRate"] for entry in entries]
    grams = [entry["stdG"] for entry in entries]
    return {
        "samplesRepeated": len(entries),
        "meanStdG": sum(grams) / len(grams),
        "maxStdG": max(grams),
        "meanStdWasteRate": sum(rates) / len(rates),
        "maxStdWasteRate": max(rates),
        "interpretation": (
            "모델 MAE가 이 값과 비슷하면 그 차이는 측정 잡음과 구분되지 않는다."
        ),
    }
