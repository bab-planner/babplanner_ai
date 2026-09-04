"""분할.

같은 그룹이 여러 분할에 걸치면 평가 숫자가 실제보다 좋아진다. 그 일이
일어나지 않는지만 본다.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from babplanner_ai.dataset.manifest import read_manifest
from babplanner_ai.dataset.schema import (
    CURRENT_LABEL_VERSION,
    AnalysisLabel,
    DatasetSample,
    GtExclusionReason,
    ServedMassSource,
    Split,
)
from babplanner_ai.dataset.splitting import (
    SplitError,
    SplitRatios,
    assign_splits,
    group_by_menu,
    split_counts,
)
from babplanner_ai.dataset.validation import group_key_of

FIXTURE = Path(__file__).parent / "fixtures" / "dataset"


@pytest.fixture
def samples() -> list[DatasetSample]:
    return read_manifest(FIXTURE / "manifest.csv")


def groups_per_split(
    assigned: list[DatasetSample], grouper=group_key_of
) -> dict[Split, set[str]]:
    result: dict[Split, set[str]] = defaultdict(set)
    for sample in assigned:
        assert sample.split is not None
        result[sample.split].add(grouper(sample))
    return result


def test_same_seed_gives_the_same_split(samples: list[DatasetSample]) -> None:
    first = assign_splits(samples, seed=42)
    second = assign_splits(samples, seed=42)

    assert [sample.split for sample in first] == [sample.split for sample in second]


def test_another_seed_can_change_the_split(samples: list[DatasetSample]) -> None:
    baseline = [sample.split for sample in assign_splits(samples, seed=42)]

    changed = any(
        [sample.split for sample in assign_splits(samples, seed=seed)] != baseline
        for seed in (1, 2, 3, 7, 11)
    )

    assert changed


def test_input_order_does_not_change_the_split(samples: list[DatasetSample]) -> None:
    # 매니페스트 줄 순서를 바꿨다고 결과가 달라지면 비교가 불가능해진다.
    forward = {s.sample_id: s.split for s in assign_splits(samples, seed=42)}
    backward = {
        s.sample_id: s.split for s in assign_splits(list(reversed(samples)), seed=42)
    }

    assert forward == backward


def test_participant_never_spans_splits(samples: list[DatasetSample]) -> None:
    per_split = groups_per_split(assign_splits(samples, seed=42))

    assert not per_split[Split.TRAIN] & per_split[Split.VAL]
    assert not per_split[Split.TRAIN] & per_split[Split.TEST]
    assert not per_split[Split.VAL] & per_split[Split.TEST]


def test_session_is_used_when_participant_is_missing(
    samples: list[DatasetSample],
) -> None:
    without_participant = [
        sample.model_copy(update={"participant_group_id": None}) for sample in samples
    ]

    per_split = groups_per_split(assign_splits(without_participant, seed=42))

    assert all(key.startswith("session:") for keys in per_split.values() for key in keys)
    assert not per_split[Split.TRAIN] & per_split[Split.TEST]


def test_every_sample_gets_exactly_one_split(samples: list[DatasetSample]) -> None:
    assigned = assign_splits(samples, seed=42)

    assert len(assigned) == len(samples)
    assert all(sample.split is not None for sample in assigned)
    assert len({sample.sample_id for sample in assigned}) == len(assigned)


def test_original_samples_are_untouched(samples: list[DatasetSample]) -> None:
    assign_splits(samples, seed=42)

    assert all(sample.split is None for sample in samples)


def test_menu_grouping_keeps_menus_apart(samples: list[DatasetSample]) -> None:
    # 처음 보는 메뉴에서 얼마나 하는지 보고 싶을 때 쓴다.
    per_split = groups_per_split(
        assign_splits(samples, seed=42, group_by="menu"), grouper=group_by_menu
    )

    assert not per_split[Split.TRAIN] & per_split[Split.TEST]


def test_unknown_grouping_is_refused(samples: list[DatasetSample]) -> None:
    with pytest.raises(SplitError, match="알 수 없는 그룹 기준"):
        assign_splits(samples, group_by="whatever")


def test_too_few_groups_fails_instead_of_splitting_rows(
    samples: list[DatasetSample],
) -> None:
    # 그룹이 둘뿐인데 셋으로 나누려 하면, 행을 쪼개는 순간 누수가 된다.
    two_groups = [
        sample.model_copy(
            update={"participant_group_id": "P01" if index % 2 else "P02"}
        )
        for index, sample in enumerate(samples)
    ]

    with pytest.raises(SplitError, match="나눌 수 없습니다"):
        assign_splits(two_groups, seed=42)


def test_empty_input_is_refused() -> None:
    with pytest.raises(SplitError, match="표본이 없습니다"):
        assign_splits([])


def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(SplitError, match="합이 1이 아닙니다"):
        SplitRatios(0.8, 0.1, 0.2)


def test_ratios_must_be_positive() -> None:
    with pytest.raises(SplitError, match="0보다 커야"):
        SplitRatios(0.9, 0.1, 0.0)


def test_custom_ratios_are_honoured(samples: list[DatasetSample]) -> None:
    assigned = assign_splits(
        samples, ratios=SplitRatios(0.5, 0.25, 0.25), seed=42
    )
    per_split = groups_per_split(assigned)

    assert per_split[Split.TRAIN]
    assert per_split[Split.VAL]
    assert per_split[Split.TEST]


def test_no_split_is_empty(samples: list[DatasetSample]) -> None:
    per_split = groups_per_split(assign_splits(samples, seed=42))

    for split in Split:
        assert per_split[split], f"{split.value} 가 비었습니다"


# --- 정답 제외 --------------------------------------------------------------


def excluded(sample: DatasetSample) -> DatasetSample:
    return sample.model_copy(
        update={"gt_exclusion_reason": GtExclusionReason.BONE_IN}
    )


def test_excluded_samples_get_no_split(samples: list[DatasetSample]) -> None:
    marked = [excluded(samples[0]), *samples[1:]]

    assigned = assign_splits(marked, seed=42)

    by_id = {sample.sample_id: sample for sample in assigned}
    assert by_id[samples[0].sample_id].split is None
    # 나머지는 그대로 나뉜다.
    assert all(
        sample.split is not None
        for sample in assigned
        if not sample.is_gt_excluded
    )


def test_excluded_samples_are_counted_separately(samples: list[DatasetSample]) -> None:
    marked = [excluded(samples[0]), *samples[1:]]

    counts = split_counts(assign_splits(marked, seed=42))

    # 아직 안 나눈 것과 일부러 뺀 것을 한 칸에 몰아넣지 않는다.
    assert counts["gt_excluded"] == 1
    assert "unassigned" not in counts


def test_excluded_samples_do_not_fill_the_group_minimum() -> None:
    """뼈 메뉴만 잔뜩 모아 놓고 분할이 만들어지면 안 된다.

    그룹은 셋인데 쓸 수 있는 표본은 하나뿐인 상태로 분할이 생기면, 학습에
    쓸 데이터가 없다는 사실이 분할 파일 뒤에 숨는다.
    """
    base = DatasetSample(
        sample_id="s0",
        image_path="images/s0.png",
        menu_id=7,
        capture_session_id="S00",
        served_edible_mass_g=300.0,
        leftover_edible_mass_g=60.0,
        waste_rate=0.2,
        served_mass_source=ServedMassSource.MEASURED,
        analysis_label=AnalysisLabel.APPROVED,
        label_version=CURRENT_LABEL_VERSION,
    )
    usable = base.model_copy(update={"sample_id": "s0", "capture_session_id": "S00"})
    marked = [
        excluded(base.model_copy(update={"sample_id": f"x{i}", "capture_session_id": f"X{i}"}))
        for i in range(3)
    ]

    with pytest.raises(SplitError, match="정답 제외 3건은 세지 않았습니다"):
        assign_splits([usable, *marked], seed=42)


def test_all_excluded_is_an_error(samples: list[DatasetSample]) -> None:
    marked = [excluded(sample) for sample in samples]

    with pytest.raises(SplitError, match="정답 제외"):
        assign_splits(marked, seed=42)
