"""데이터 상태 점검.

**읽기만 한다.** 원본을 고치거나 옮기지 않는다. 이 도구의 목적은 "지금
학습에 쓸 수 있는 표본이 몇 개인가"에 정확히 답하는 것이고, 그 답이 0이면
0이라고 말하는 것이다.

서비스 쪽 `src/babplanner_ai/dataset/validation.py`가 매니페스트의 정합성을
검사한다(열 이름, 정답 계산, 분할 누수). 여기 있는 것은 그 앞 단계다 —
**애초에 파일이 있는가, 몇 개인가, 중복은 없는가.**

`0`을 감추지 않는다. 데이터가 없다는 사실은 이 단계에서 가장 중요한 결과다.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from waste_estimation.schemas import WasteSample


class AuditError(ValueError):
    """점검할 수 없다."""


#: 세는 이미지 확장자. 목록에 없는 것은 이미지로 세지 않는다.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})


@dataclass(frozen=True)
class DistributionSummary:
    """라벨 분포. 표본이 적으면 적다고 말한다."""

    count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    median: float | None
    #: (구간 시작, 개수). 경계는 부르는 쪽이 정한다.
    histogram: list[tuple[float, int]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "median": self.median,
            "histogram": [
                {"binStart": start, "count": value} for start, value in self.histogram
            ],
        }


def summarize_rates(
    rates: Sequence[float], bin_width: float = 0.1
) -> DistributionSummary:
    """정답 분포를 요약한다.

    표본이 없으면 통계값이 전부 None이다. 0으로 채우면 "평균 잔반률 0%"처럼
    읽힌다.
    """
    if bin_width <= 0:
        raise AuditError(f"bin_width는 0보다 커야 합니다: {bin_width}")
    usable = sorted(float(value) for value in rates)
    if not usable:
        return DistributionSummary(count=0, minimum=None, maximum=None, mean=None, median=None)

    middle = len(usable) // 2
    median = (
        usable[middle]
        if len(usable) % 2
        else (usable[middle - 1] + usable[middle]) / 2
    )

    counts: dict[int, int] = defaultdict(int)
    for value in usable:
        index = min(int(value / bin_width), int(1.0 / bin_width) - 1)
        counts[index] += 1
    histogram = [
        (round(index * bin_width, 10), counts[index]) for index in sorted(counts)
    ]

    return DistributionSummary(
        count=len(usable),
        minimum=usable[0],
        maximum=usable[-1],
        mean=sum(usable) / len(usable),
        median=median,
        histogram=histogram,
    )


@dataclass(frozen=True)
class SampleAudit:
    """표본 묶음의 상태."""

    total: int
    labeled: int
    unlabeled: int
    groups: int
    largest_group: int
    duplicate_sample_ids: list[str]
    distribution: DistributionSummary

    @property
    def usable_for_training(self) -> bool:
        """지금 학습을 시작할 수 있는가.

        정답이 있는 표본이 하나도 없으면 무엇을 해도 학습이 아니다. 여기서
        "적지만 가능"처럼 흐리게 말하지 않는다. 몇 개가 필요한지는 이
        함수가 정할 문제가 아니고, **0인지 아닌지**는 정할 수 있다.
        """
        return self.labeled > 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "labeled": self.labeled,
            "unlabeled": self.unlabeled,
            "groups": self.groups,
            "largestGroup": self.largest_group,
            "duplicateSampleIds": self.duplicate_sample_ids,
            "distribution": self.distribution.as_dict(),
            "usableForTraining": self.usable_for_training,
        }


def audit_samples(
    samples: Sequence[WasteSample], bin_width: float = 0.1
) -> SampleAudit:
    """표본 목록의 상태를 센다."""
    seen: dict[str, int] = defaultdict(int)
    for sample in samples:
        seen[sample.sample_id] += 1
    duplicates = sorted(key for key, count in seen.items() if count > 1)

    group_counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        group_counts[sample.group_id] += 1

    labeled = [s.waste_rate for s in samples if s.waste_rate is not None]
    return SampleAudit(
        total=len(samples),
        labeled=len(labeled),
        unlabeled=len(samples) - len(labeled),
        groups=len(group_counts),
        largest_group=max(group_counts.values(), default=0),
        duplicate_sample_ids=duplicates,
        distribution=summarize_rates(labeled, bin_width),
    )


@dataclass(frozen=True)
class ImageAudit:
    """디렉터리 안의 이미지 파일 상태."""

    root: str
    exists: bool
    image_count: int
    #: 같은 내용의 파일 묶음. 하나라도 있으면 학습/평가가 오염될 수 있다.
    duplicate_groups: list[list[str]]
    unreadable: list[str]

    @property
    def unique_images(self) -> int:
        duplicated = sum(len(group) - 1 for group in self.duplicate_groups)
        return self.image_count - duplicated

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "exists": self.exists,
            "imageCount": self.image_count,
            "uniqueImages": self.unique_images,
            "duplicateGroups": self.duplicate_groups,
            "unreadable": self.unreadable,
        }


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_images(root: Path) -> ImageAudit:
    """디렉터리를 훑어 이미지 수와 정확 중복을 센다.

    **내용을 디코딩하지 않는다.** 흐림이나 밝기 같은 품질 판정은 임계값이
    필요하고, 그 임계값은 실제 데이터를 보기 전에 정할 수 없다. 여기서는
    파일이 열리는지와 해시만 본다.

    지각적 중복(비슷하지만 다른 파일)은 다루지 않는다. 정확 중복만으로도
    같은 사진이 학습과 평가에 함께 들어가는 가장 흔한 경우를 잡는다.
    """
    if not root.exists():
        return ImageAudit(
            root=str(root), exists=False, image_count=0, duplicate_groups=[], unreadable=[]
        )

    by_hash: dict[str, list[str]] = defaultdict(list)
    unreadable: list[str] = []
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        count += 1
        try:
            by_hash[sha256_of(path)].append(str(path.relative_to(root)))
        except OSError:
            unreadable.append(str(path.relative_to(root)))

    duplicates = sorted(
        (sorted(paths) for paths in by_hash.values() if len(paths) > 1),
        key=lambda group: group[0],
    )
    return ImageAudit(
        root=str(root),
        exists=True,
        image_count=count,
        duplicate_groups=duplicates,
        unreadable=unreadable,
    )
