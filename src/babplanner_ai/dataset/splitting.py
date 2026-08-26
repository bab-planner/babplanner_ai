"""학습/검증/평가 분할.

행을 무작위로 나누지 않는다. 같은 사람이 같은 회차에 찍은 사진들은 서로
거의 같은 문제이고, 그것들이 학습과 평가에 나뉘어 들어가면 평가 숫자가
실제보다 좋아진다. 그래서 그룹째로 나눈다.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from babplanner_ai.dataset.schema import DatasetSample, Split
from babplanner_ai.dataset.validation import group_key_of

DEFAULT_SEED = 42

#: 기본 비율. 제품 정책이 아니라 출발점이며 호출할 때 바꿀 수 있다.
DEFAULT_RATIOS = (0.8, 0.1, 0.1)


class SplitError(ValueError):
    """분할할 수 없다."""


@dataclass(frozen=True)
class SplitRatios:
    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        for name, value in (
            ("train", self.train),
            ("val", self.val),
            ("test", self.test),
        ):
            if value <= 0:
                raise SplitError(f"{name} 비율은 0보다 커야 합니다.")
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-9:
            raise SplitError(f"비율의 합이 1이 아닙니다: {total}")

    @classmethod
    def of(cls, ratios: tuple[float, float, float] = DEFAULT_RATIOS) -> "SplitRatios":
        return cls(*ratios)


def group_by_menu(sample: DatasetSample) -> str:
    """메뉴 단위로 묶는다.

    한 번도 본 적 없는 메뉴에서 얼마나 하는지 보고 싶을 때 쓴다. 표본이
    적을 때 억지로 이 기준을 쓰면 평가 집합이 한두 메뉴로 쏠린다.
    """
    return f"menu:{sample.menu_id}"


GROUPERS = {
    "auto": group_key_of,
    "menu": group_by_menu,
}


def assign_splits(
    samples: list[DatasetSample],
    *,
    ratios: SplitRatios | None = None,
    seed: int = DEFAULT_SEED,
    group_by: str = "auto",
) -> list[DatasetSample]:
    """분할을 붙인 새 목록을 돌려준다. 원본은 건드리지 않는다.

    같은 매니페스트와 같은 seed면 언제나 같은 결과가 나온다. 그러지 않으면
    어제의 평가 숫자와 오늘의 숫자를 비교할 수 없다.
    """
    if not samples:
        raise SplitError("표본이 없습니다.")

    grouper = GROUPERS.get(group_by)
    if grouper is None:
        raise SplitError(
            f"알 수 없는 그룹 기준입니다: {group_by} "
            f"(가능: {', '.join(sorted(GROUPERS))})"
        )

    ratios = ratios or SplitRatios.of()

    members: dict[str, list[DatasetSample]] = defaultdict(list)
    for sample in samples:
        members[grouper(sample)].append(sample)

    keys = sorted(members)
    if len(keys) < 3:
        # 그룹이 셋보다 적으면 세 갈래로 나눌 수 없다. 행을 쪼개서
        # 억지로 만들면 그게 바로 누수다.
        raise SplitError(
            f"그룹이 {len(keys)}개뿐이라 train/val/test로 나눌 수 없습니다. "
            "그룹을 더 모으거나 분할을 만들지 마세요."
        )

    # 정렬한 뒤 섞는다. 입력 순서가 결과를 바꾸지 않게 하려는 것이다.
    random.Random(seed).shuffle(keys)

    train_count = max(1, round(len(keys) * ratios.train))
    val_count = max(1, round(len(keys) * ratios.val))
    # 남은 것은 전부 평가로. 반올림 때문에 하나도 안 남는 일이 없게 한다.
    if train_count + val_count >= len(keys):
        train_count = len(keys) - 2
        val_count = 1

    boundaries = {
        Split.TRAIN: keys[:train_count],
        Split.VAL: keys[train_count : train_count + val_count],
        Split.TEST: keys[train_count + val_count :],
    }

    split_of_group = {
        key: split for split, group_keys in boundaries.items() for key in group_keys
    }

    return [
        sample.model_copy(update={"split": split_of_group[grouper(sample)]})
        for sample in samples
    ]


def split_counts(samples: list[DatasetSample]) -> dict[str, int]:
    counts: dict[str, int] = {split.value: 0 for split in Split}
    unassigned = 0
    for sample in samples:
        if sample.split is None:
            unassigned += 1
        else:
            counts[sample.split.value] += 1
    if unassigned:
        counts["unassigned"] = unassigned
    return counts
