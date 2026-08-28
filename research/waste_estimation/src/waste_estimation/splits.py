"""그룹 단위 분할.

**이미지 단위 무작위 분할을 쓰지 않는다.** 같은 사람이 같은 회차에 찍은
사진들은 배경도 식판도 습관도 닮아 있다. 그것들이 학습과 평가에 나뉘어
들어가면 모델이 사람을 외운 것을 일반화로 착각하게 된다. 서비스 쪽
`src/babplanner_ai/dataset/splitting.py`가 같은 이유로 그룹째 나눈다.

여기 있는 것은 **연구 표본(`WasteSample`)에 대한 같은 규칙**이다. 서비스
매니페스트에 없는 출처(공개 데이터 등)도 같은 방식으로 다루기 위해서다.

분할 단위는 `group_id`이고, 그 값을 여기서 만들어 내지 않는다. 서비스
스키마는 참가자 키가 있으면 그것을, 없으면 촬영 회차를 쓴다
(`dataset/validation.py`의 `group_key_of`). 없는 그룹 정보를 발명하면
누수를 막은 것처럼 보이지만 실제로는 막지 못한다.

**표본이 적을 때 한 번의 분할로 결론을 내지 않는다.** 그래서 holdout뿐
아니라 그룹 단위 k-fold도 둔다.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from waste_estimation.schemas import WasteSample


class SplitError(ValueError):
    """분할할 수 없다."""


def group_ids(samples: Sequence[WasteSample]) -> list[str]:
    """등장 순서를 유지한 그룹 목록. 결정적이어야 재현된다."""
    seen: dict[str, None] = {}
    for sample in samples:
        seen.setdefault(sample.group_id, None)
    return list(seen)


def group_holdout_split(
    samples: Sequence[WasteSample],
    holdout_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[WasteSample], list[WasteSample]]:
    """그룹을 섞어 나눈다. 같은 그룹은 한쪽에만 들어간다.

    비율은 **표본 수가 아니라 그룹 수** 기준이다. 그룹마다 표본 수가 다르면
    실제 표본 비율은 조금 달라진다. 그 대신 누수가 없다. 둘 중 하나만
    가질 수 있고, 이 문제에서는 누수가 없는 쪽이 훨씬 중요하다.
    """
    if not 0.0 < holdout_ratio < 1.0:
        raise SplitError(f"holdout_ratio는 0과 1 사이여야 합니다: {holdout_ratio}")
    groups = group_ids(samples)
    if len(groups) < 2:
        raise SplitError(
            f"그룹이 {len(groups)}개뿐이라 누수 없이 나눌 수 없습니다. "
            "표본을 더 모으거나 분할 단위를 다시 정해야 합니다."
        )

    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    holdout_count = max(1, round(len(shuffled) * holdout_ratio))
    if holdout_count >= len(shuffled):
        holdout_count = len(shuffled) - 1
    holdout_groups = set(shuffled[:holdout_count])

    train = [s for s in samples if s.group_id not in holdout_groups]
    holdout = [s for s in samples if s.group_id in holdout_groups]
    return train, holdout


def group_kfold(
    samples: Sequence[WasteSample], folds: int = 5, seed: int = 42
) -> list[tuple[list[WasteSample], list[WasteSample]]]:
    """그룹 단위 k-fold.

    표본이 적을 때 한 번의 분할은 어느 그룹이 평가에 걸렸는지에 따라 결과가
    크게 흔들린다. 폴드마다 결과를 내고 그 분산까지 함께 보고해야 한다.
    """
    if folds < 2:
        raise SplitError(f"folds는 2 이상이어야 합니다: {folds}")
    groups = group_ids(samples)
    if len(groups) < folds:
        raise SplitError(
            f"그룹이 {len(groups)}개인데 {folds}-fold를 만들 수 없습니다."
        )

    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    buckets: list[set[str]] = [set() for _ in range(folds)]
    for index, group in enumerate(shuffled):
        buckets[index % folds].add(group)

    result: list[tuple[list[WasteSample], list[WasteSample]]] = []
    for bucket in buckets:
        train = [s for s in samples if s.group_id not in bucket]
        test = [s for s in samples if s.group_id in bucket]
        result.append((train, test))
    return result


def leaked_groups(
    train: Sequence[WasteSample], test: Sequence[WasteSample]
) -> set[str]:
    """양쪽에 걸친 그룹. 비어 있어야 한다.

    분할 함수를 믿지 않고 결과를 검사하는 자리를 따로 둔다. 분할을 손으로
    고치는 일이 생기기 때문이다.
    """
    return {s.group_id for s in train} & {s.group_id for s in test}


def group_sizes(samples: Sequence[WasteSample]) -> dict[str, int]:
    """그룹별 표본 수. 한 그룹이 데이터를 지배하는지 본다."""
    counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        counts[sample.group_id] += 1
    return dict(counts)
