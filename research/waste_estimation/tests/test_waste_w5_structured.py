"""W5 — 출력 구조의 규약.

여기서 지키는 것은 성능이 아니라 **경계**다.

    누적 확률은 비증가여야 한다
    구간 확률은 음수가 없고 합이 1이다
    구간 대표값은 학습 fold 정답에서만 나온다
    전문가는 학습 fold 표본으로만 맞춘다
    게이트 점수를 확률이라고 부르지 않는다
    딱딱한 전환을 쓰지 않는다
    후보 간 표현이 같다
    예측 정책이 후보마다 다르지 않다
    이미 본 집합에서 다시 고르지 않는다
    fixture를 갈아치우지 않는다

무거운 의존성 없이 도는 것만 여기 둔다. torch가 필요한 부분은 스크립트에
있고, 그 결과 파일을 읽어서 검사한다.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from waste_estimation.structured import (
    CUMULATIVE_THRESHOLDS,
    StructuredError,
    bin_probabilities,
    bin_representatives,
    cumulative_bin_index,
    cumulative_targets,
    expected_value,
    monotonic_repair,
    prediction_quantiles,
    soft_mixture,
)

requires_numpy = pytest.mark.skipif(
    importlib.util.find_spec("numpy") is None,
    reason="numpy가 필요합니다 (.venv-waste에서 돕니다)",
)

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "research" / "waste_estimation" / "experiments" / "w5_structured"
W4 = ROOT / "research" / "waste_estimation" / "experiments" / "w4_tail"
REPORT = ROOT / "reports" / "waste_w5_structured_outputs.md"
W4_REPORT = ROOT / "reports" / "waste_w4_tail_aware_regression.md"
SCRIPTS = ROOT / "research" / "waste_estimation" / "scripts"

NAMES = (
    "S0_STANDARD_RIDGE",
    "S1_CUMULATIVE_ORDINAL_EXPECTATION",
    "S2_SOFT_HIGH_LOW_MIXTURE",
)


def read_json(name: str, base: Path = EXP) -> dict:
    path = base / name
    if not path.exists():
        pytest.skip(f"{path.name}이 아직 없습니다. W5 스크립트를 먼저 돌리세요.")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(name: str, base: Path = EXP) -> list[dict]:
    path = base / name
    if not path.exists():
        pytest.skip(f"{path.name}이 아직 없습니다. W5 스크립트를 먼저 돌리세요.")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def report_text(path: Path = REPORT) -> str:
    if not path.exists():
        pytest.skip("보고서가 아직 없습니다.")
    return " ".join(path.read_text(encoding="utf-8").split())


def bounded(rows: list[dict]) -> dict[str, dict]:
    return {row["candidate"]: row for row in rows if row["output"] == "bounded"}


# --- W4 정정 ------------------------------------------------------------------


def test_w4_overclaim_was_retracted_not_silently_deleted():
    """문장을 지우면 무엇이 바뀌었는지 남지 않는다. 내렸다고 적는다."""
    text = report_text(W4_REPORT)
    retracted = "꼬리 실패는 정보 부족이 아니라 목적함수 문제"
    assert text.count(retracted) == 1
    assert "초판은" in text[: text.index(retracted)][-200:]


def test_w5_freeze_carries_the_corrected_w4_conclusion():
    note = read_json("freeze.json")["priorStages"]["w4CorrectedConclusion"]
    assert "판별하는 신호" in note
    assert "충분한 정보를 담고 있음을 보이지 않았다" in note


def test_w4_artifacts_now_carry_both_catastrophic_aggregations():
    for row in read_csv("candidate_summary.csv", base=W4):
        assert row["catastrophicRatePooled"] not in ("", None)
        assert row["catastrophicRateMacro"] not in ("", None)


def test_w5_report_explains_the_aggregation_correction():
    text = report_text()
    assert "32/90 = 31.4%" in text
    assert "두 집계를 섞은 분수다" in text
    assert "35.6%" in text


def test_w5_uses_pooled_as_the_primary_aggregation():
    assert read_json("freeze.json")["catastrophicPrimaryAggregation"] == "pooled"


# --- 누적 순서형 --------------------------------------------------------------


def test_cumulative_thresholds_are_the_w2_bin_edges():
    policy = read_json("freeze.json")
    assert tuple(policy["cumulativeThresholds"]) == CUMULATIVE_THRESHOLDS
    assert CUMULATIVE_THRESHOLDS == (0.2, 0.4, 0.6, 0.8)


@requires_numpy
def test_cumulative_targets_use_left_closed_intervals():
    """`y >= t` 이다. 그래야 마지막 칸이 고잔반 집합과 정확히 같다."""
    import numpy as np

    from waste_estimation.tail import is_high_waste

    targets = np.array([0.0, 0.2, 0.79, 0.8, 1.0])
    z = cumulative_targets(targets)
    assert z[:, 0].tolist() == [0, 1, 1, 1, 1]  # y >= 0.2
    assert z[:, 3].tolist() == [0, 0, 0, 1, 1]  # y >= 0.8
    assert z[:, 3].astype(bool).tolist() == is_high_waste(targets).tolist()


@requires_numpy
def test_cumulative_bin_index_counts_thresholds_passed():
    import numpy as np

    assert cumulative_bin_index(np.array([0.0, 0.25, 0.5, 0.7, 0.8, 1.0])).tolist() == [
        0,
        1,
        2,
        3,
        4,
        4,
    ]


@requires_numpy
def test_cumulative_targets_refuse_unsorted_thresholds():
    import numpy as np

    with pytest.raises(StructuredError):
        cumulative_targets(np.array([0.5]), (0.4, 0.2))


@requires_numpy
def test_monotonic_repair_makes_probabilities_non_increasing():
    import numpy as np

    repaired = monotonic_repair(np.array([[0.5, 0.9, 0.3, 0.4]]))
    assert repaired[0].tolist() == pytest.approx([0.5, 0.5, 0.3, 0.3])
    assert (np.diff(repaired, axis=1) <= 1e-12).all()


@requires_numpy
def test_monotonic_repair_is_the_running_minimum_not_the_running_maximum():
    """확률을 올리는 보정은 고잔반 확률을 낙관적으로 부풀린다.

    과소추정이 관심사이므로 보수적인 쪽(내리는 쪽)을 골랐다.
    """
    import numpy as np

    repaired = monotonic_repair(np.array([[0.2, 0.9]]))
    assert repaired[0, 1] == pytest.approx(0.2)  # 0.9로 올리지 않는다
    assert repaired[0, 0] == pytest.approx(0.2)


@requires_numpy
def test_monotonic_repair_leaves_valid_input_untouched():
    import numpy as np

    valid = np.array([[0.9, 0.7, 0.4, 0.1]])
    assert monotonic_repair(valid)[0].tolist() == pytest.approx(valid[0].tolist())


@requires_numpy
def test_monotonic_repair_refuses_values_outside_zero_one():
    import numpy as np

    with pytest.raises(StructuredError):
        monotonic_repair(np.array([[1.4, 0.2]]))


@requires_numpy
def test_bin_probabilities_are_non_negative_and_sum_to_one():
    import numpy as np

    rng = np.random.default_rng(0)
    # 일부러 위반이 섞인 누적 확률을 넣는다.
    cumulative = rng.random((200, len(CUMULATIVE_THRESHOLDS)))
    probabilities = bin_probabilities(cumulative)
    assert probabilities.shape == (200, len(CUMULATIVE_THRESHOLDS) + 1)
    assert (probabilities >= 0).all()
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(200))


@requires_numpy
def test_bin_probabilities_recover_a_one_hot_distribution():
    import numpy as np

    # y가 확실히 마지막 칸에 있다고 말하는 누적 확률
    probabilities = bin_probabilities(np.array([[1.0, 1.0, 1.0, 1.0]]))
    assert probabilities[0].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0])
    # 확실히 첫 칸
    probabilities = bin_probabilities(np.array([[0.0, 0.0, 0.0, 0.0]]))
    assert probabilities[0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0])


# --- 구간 대표값 ---------------------------------------------------------------


@requires_numpy
def test_bin_representatives_are_training_bin_means():
    import numpy as np

    targets = np.array([0.0, 0.1, 0.25, 0.45, 0.65, 0.9, 1.0])
    r = bin_representatives(targets)
    assert r[0] == pytest.approx(0.05)  # (0.0 + 0.1) / 2
    assert r[4] == pytest.approx(0.95)  # (0.9 + 1.0) / 2


@requires_numpy
def test_bin_representatives_beat_fixed_midpoints_on_this_data():
    """양 끝이 0과 1에 몰려 있어 중점을 쓰면 체계적으로 어긋난다."""
    rows = read_csv("structure_diagnostics.csv")
    for row in rows:
        values = [float(v) for v in row["representatives"].split(";")]
        assert values[0] < 0.10, "첫 칸 실제 평균이 중점 0.1보다 훨씬 낮다"
        assert values[-1] > 0.90, "마지막 칸 실제 평균이 중점 0.9보다 높다"


@requires_numpy
def test_empty_bin_falls_back_to_the_midpoint_without_looking_at_labels():
    import numpy as np

    # 중간 두 칸이 비어 있다.
    r = bin_representatives(np.array([0.05, 0.1, 0.9, 0.95]))
    assert r[1] == pytest.approx(0.3)  # [0.2,0.4) 중점
    assert r[2] == pytest.approx(0.5)  # [0.4,0.6) 중점


def test_no_fold_actually_had_an_empty_bin():
    """대체 규칙은 정해 뒀지만 발동하지 않았다는 사실을 기록한다."""
    for row in read_csv("structure_diagnostics.csv"):
        values = [float(v) for v in row["representatives"].split(";")]
        assert len(values) == len(CUMULATIVE_THRESHOLDS) + 1
        assert values == sorted(values)


@requires_numpy
def test_bin_representatives_refuse_a_wrong_fallback_length():
    import numpy as np

    with pytest.raises(StructuredError):
        bin_representatives(np.array([0.5]), fallback=(0.1, 0.2))


# --- 기대값 --------------------------------------------------------------------


@requires_numpy
def test_expected_value_is_the_probability_weighted_representative():
    import numpy as np

    probabilities = np.array([[0.5, 0.0, 0.0, 0.0, 0.5]])
    assert expected_value(probabilities, np.array([0.0, 0.3, 0.5, 0.7, 1.0]))[
        0
    ] == pytest.approx(0.5)


@requires_numpy
def test_expected_value_stays_inside_zero_one_by_construction():
    """S1이 [0,1] 안에서만 답하는 것은 자르기 때문이 아니라 구조 때문이다."""
    import numpy as np

    rng = np.random.default_rng(1)
    probabilities = bin_probabilities(rng.random((300, 4)))
    representatives = np.array([0.02, 0.30, 0.49, 0.70, 0.96])
    values = expected_value(probabilities, representatives)
    assert values.min() >= representatives.min() - 1e-9
    assert values.max() <= representatives.max() + 1e-9


@requires_numpy
def test_expected_value_refuses_a_bin_count_mismatch():
    import numpy as np

    with pytest.raises(StructuredError):
        expected_value(np.ones((2, 5)) / 5, np.array([0.1, 0.5]))


# --- 부드러운 혼합 --------------------------------------------------------------


@requires_numpy
def test_soft_mixture_is_the_convex_combination():
    import numpy as np

    values = soft_mixture(
        np.array([0.0, 0.5, 1.0]),
        np.array([0.1, 0.1, 0.1]),
        np.array([0.9, 0.9, 0.9]),
    )
    assert values.tolist() == pytest.approx([0.1, 0.5, 0.9])


@requires_numpy
def test_soft_mixture_is_continuous_not_a_hard_switch():
    """딱딱한 전환이면 0.499와 0.501 사이에서 예측이 튄다."""
    import numpy as np

    low, high = np.array([0.1, 0.1]), np.array([0.9, 0.9])
    values = soft_mixture(np.array([0.499, 0.501]), low, high)
    assert abs(values[1] - values[0]) < 0.01


@requires_numpy
def test_soft_mixture_refuses_a_gate_outside_zero_one():
    import numpy as np

    with pytest.raises(StructuredError):
        soft_mixture(np.array([1.5]), np.array([0.1]), np.array([0.9]))


def test_freeze_says_hard_switching_is_not_used():
    candidate = read_json("freeze.json")["candidates"][2]
    assert candidate["hardSwitching"] is False
    assert "경계에서 예측이 튄다" in candidate["hardSwitchingNote"]


def test_gate_score_is_not_called_a_calibrated_probability():
    candidate = read_json("freeze.json")["candidates"][2]
    assert "보정된 확률이라고 부르지 않는다" in candidate["gateNote"]
    assert "보정된 확률이 아니다" in report_text()


def test_experts_are_trained_on_disjoint_training_subsets():
    for row in read_csv("structure_diagnostics.csv"):
        low, high = int(row["lowExpertCount"]), int(row["highExpertCount"])
        assert low > 0 and high > 0
        assert low + high == int(row["trainCount"])


def test_expert_sample_sufficiency_was_checked_and_never_failed():
    minimum = read_json("freeze.json")["candidates"][2]["minExpertSamples"]
    for row in read_csv("structure_diagnostics.csv"):
        assert int(row["highExpertCount"]) >= minimum
        assert row["mixtureFailed"] == "False"
    assert read_json("metadata.json")["mixtureFailedFolds"] == 0


# --- 누수 ----------------------------------------------------------------------


def test_nothing_used_validation_labels():
    source = (SCRIPTS / "w5_experiment.py").read_text(encoding="utf-8")
    assert "train_targets" in source
    assert "평가 표본의 정답은 어디에도 들어가지 않는다" in source
    assert "학습 fold 정답만 봤다" in read_json("metadata.json")["note"]


def test_outer_split_reuses_the_w4_folds_with_no_leakage():
    policy = read_json("freeze.json")
    assert policy["outerFolds"] == 5
    assert policy["outerSeed"] == 42
    assert "W4와 동일한 바깥 fold" in policy["foldReuseNote"]
    assert read_json("metadata.json")["outerGroupLeakage"] == 0


def test_representation_is_identical_across_candidates():
    policy = read_json("freeze.json")
    assert policy["representation"]["identicalAcrossCandidates"] is True
    assert policy["representation"]["changed"] is False
    assert read_json("metadata.json")["representationIdenticalAcrossCandidates"] is True


def test_no_hyperparameter_search_was_rerun():
    policy = read_json("freeze.json")
    assert policy["hyperparameterSearch"] is False
    assert policy["ridgeAlpha"] == 100.0


# --- 예측 정책과 선택 -----------------------------------------------------------


def test_bounded_is_the_primary_metric_and_the_change_is_declared():
    policy = read_json("freeze.json")["predictionPolicy"]
    assert policy["primary"].startswith("BOUNDED")
    assert "W2의 주 지표는 raw였다" in policy["differsFromW2"]
    assert read_json("metadata.json")["predictionPolicy"] == "bounded"


def test_every_candidate_is_measured_under_both_policies():
    rows = read_csv("candidate_metrics.csv")
    for name in NAMES:
        outputs = {row["output"] for row in rows if row["candidate"] == name}
        assert outputs == {"bounded", "raw"}, name


def test_the_same_policy_is_applied_to_every_candidate():
    """후보마다 범위 정책이 다르면 구조가 아니라 범위를 비교하게 된다."""
    rows = bounded(read_csv("candidate_metrics.csv"))
    assert set(rows) == set(NAMES)


def test_freeze_hash_matches_the_record():
    record = read_json("freeze.json")
    stored = record.pop("sha256OfRecordWithoutHash")
    payload = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == stored


def test_overall_guardrail_comes_before_the_tail_criterion():
    rule = read_json("freeze.json")["selectionRule"]
    assert rule[0].startswith("Gate 1")
    assert "탈락" in rule[0]
    assert "치명적 과소추정 비율(pooled)" in rule[1]


def test_selection_used_the_pooled_catastrophic_rate():
    meta = read_json("metadata.json")
    rows = bounded(read_csv("candidate_metrics.csv"))
    passed = meta["gate1Passed"]
    best = min(float(rows[name]["catastrophicRatePooled"]) for name in passed)
    assert float(rows[meta["selected"]]["catastrophicRatePooled"]) == pytest.approx(best)


def test_the_selected_candidate_passed_the_overall_guardrail():
    meta = read_json("metadata.json")
    rows = bounded(read_csv("candidate_metrics.csv"))
    assert float(rows[meta["selected"]]["mae"]) <= meta["gate1Threshold"]


def test_fold_stability_is_reported_for_both_tail_metrics():
    rows = read_csv("fold_stability.csv")
    assert {row["candidate"] for row in rows} == set(NAMES[1:])
    for row in rows:
        assert int(row["foldsCompared"]) >= 2
        assert row["highMaeImproved"] not in ("", None)
        assert row["catastrophicImproved"] not in ("", None)


def test_no_p_value_is_manufactured_from_24_groups():
    policy = read_json("freeze.json")
    assert "p-value를 억지로 만들지 않는다" in policy["statisticalClaimLimit"]
    assert "p-value" not in report_text().replace("p-value를 억지로", "")


# --- 이미 본 집합 ---------------------------------------------------------------


def test_analysis_split_is_not_called_a_test():
    meta = read_json("analysis_metadata.json")
    assert meta["isFinalTest"] is False
    assert meta["isGeneralizationTest"] is False
    assert meta["isPristineValidation"] is False
    assert meta["resultName"] == "POST-HOC FROZEN ANALYSIS"


def test_analysis_was_applied_exactly_once_after_the_selection():
    meta = read_json("analysis_metadata.json")
    assert meta["applications"] == 1
    assert meta["selectionMadeOn"] == "DEVELOPMENT group CV"
    assert read_json("metadata.json")["analysisOpened"] is False


def test_analysis_script_cannot_select_anything():
    source = (SCRIPTS / "w5_analysis.py").read_text(encoding="utf-8")
    assert "여기서는 아무것도 고르지 않는다" in source
    assert 'selected = development["selected"]' in source
    assert "argmin" not in source


def test_analysis_uses_the_frozen_policy_hash():
    policy_hash = read_json("freeze.json")["sha256OfRecordWithoutHash"]
    assert read_json("metadata.json")["policyHash"] == policy_hash
    assert read_json("analysis_metadata.json")["policyHash"] == policy_hash


def test_freeze_forbids_changing_anything_on_the_analysis_split():
    joined = " ".join(read_json("freeze.json")["forbidden"])
    for phrase in (
        "ANALYSIS에서 후보 선택",
        "ANALYSIS에서 threshold 변경",
        "ANALYSIS에서 head 구조 변경",
    ):
        assert phrase in joined, phrase


# --- 회복과 악화 ---------------------------------------------------------------


def test_recovered_and_new_regression_are_defined_in_the_freeze():
    policy = read_json("freeze.json")
    assert "RECOVERED" in policy["recoveredDefinition"]
    assert "NEW_CATASTROPHIC_REGRESSION" in policy["newRegressionDefinition"]


@requires_numpy
def test_recovery_labels_match_the_frozen_definitions():
    rows = read_csv("catastrophic_recovery.csv")
    for row in rows:
        before = row["S0_catastrophic"] == "True"
        after = row["selected_catastrophic"] == "True"
        expected = (
            "RECOVERED"
            if before and not after
            else "NEW_CATASTROPHIC_REGRESSION"
            if after and not before
            else "remaining"
            if before
            else "none"
        )
        assert row["movement"] == expected, row["sampleId"]


def test_new_catastrophic_regressions_are_reported_even_if_zero():
    movement = read_json("analysis_metadata.json")["catastrophicMovement"]
    assert "newRegression" in movement
    assert "NEW_CATASTROPHIC_REGRESSION 0건" in report_text()


def test_the_largest_failures_are_recorded_without_changing_the_model():
    rows = read_csv("w2_largest_failures.csv")
    assert len(rows) == 3
    assert "모델을 다시 바꾸지 않았다" in report_text()


def test_report_admits_the_worst_case_did_not_move():
    text = report_text()
    assert "Telur Mata Sapi는 조금도 움직이지 않았다" in text


# --- 분포 진단 -----------------------------------------------------------------


def test_prediction_distribution_is_recorded_for_ground_truth_too():
    rows = read_csv("analysis_prediction_distribution.csv")
    assert {row["candidate"] for row in rows} == set(NAMES) | {"GROUND_TRUTH"}


def test_the_structure_did_not_merely_inflate_variance():
    """분산만 키웠다면 저잔반이 나빠졌을 것이다. 그러지 않았다."""
    rows = bounded(read_csv("analysis_metrics.csv"))
    meta = read_json("metadata.json")
    assert float(rows[meta["selected"]]["lowMae"]) <= float(
        rows["S0_STANDARD_RIDGE"]["lowMae"]
    ) + 0.01


def test_directional_calibration_shows_the_remaining_shrinkage():
    """수축이 줄었다고만 적고 남은 양을 안 적으면 그건 숨김이다."""
    rows = {row["level"]: row for row in read_csv("directional_calibration.csv")}
    high = rows["high"]
    target = float(high["meanTarget"])
    selected = float(
        high[f"{read_json('metadata.json')['selected']}_meanPrediction"]
    )
    assert selected < target, "고잔반 평균 수축이 사라졌다면 확인하세요"
    assert "3분의 2는 남아 있다" in report_text()


@requires_numpy
def test_prediction_quantiles_refuse_empty_input():
    import numpy as np

    with pytest.raises(StructuredError):
        prediction_quantiles(np.array([]))


# --- 범위 밖으로 나가지 않았다 ---------------------------------------------------


def test_reliability_was_frozen_not_retuned():
    policy = read_json("freeze.json")
    assert policy["reliabilityFrozen"] is True
    assert "재튜닝하지도 않는다" in policy["reliabilityNote"]


def test_quantile_regression_was_left_out_of_scope():
    joined = " ".join(read_json("freeze.json")["forbidden"])
    assert "분위 회귀" in joined


def test_no_resampling_experiments_were_added():
    joined = " ".join(read_json("freeze.json")["forbidden"])
    assert "표본 재가중 추가 실험" in joined
    assert "backbone 변경 / fine-tuning / 데이터 증강" in joined


def test_fresh_test_absence_is_recorded_in_the_frozen_wording():
    fresh = read_json("freeze.json")["freshTest"]
    assert fresh["available"] is False
    assert "pristine external validation pending" in fresh["note"]
    assert "pristine external validation pending" in report_text()


def test_report_lists_the_threats_including_the_new_ones():
    text = report_text()
    assert "Threats to Validity" in text
    assert "S1은 정답을 내부적으로 이산화한다" in text
    assert "게이트가 놓친 표본은 구조의 도움을 전혀 못 받는다" in text


def test_report_records_all_three_hypothesis_verdicts():
    text = report_text()
    for name in ("H-W5-A", "H-W5-B", "H-W5-C"):
        assert name in text


def test_report_states_this_is_not_cafeteria_performance():
    assert "밥플래너 급식소 성능이 아니다" in report_text()


def test_report_explains_why_s0_numbers_differ_from_w4():
    text = report_text()
    assert "W2의 주 지표는 raw였다" in text
    assert "같은 정책끼리만 비교" in text


def test_charts_exist_for_the_three_required_views():
    assets = ROOT / "reports" / "assets"
    for name in (
        "waste_w5_tradeoff.svg",
        "waste_w5_catastrophic.svg",
        "waste_w5_high_waste_scatter.svg",
    ):
        assert (assets / name).exists(), name


# --- production 경계 ------------------------------------------------------------


def test_ai_service_has_no_runtime_waste_analysis():
    """최종 AI 범위는 CCTV 혼잡도와 식수 인원 예측 둘뿐이다.

    잔반 분석은 AI 서비스의 운영 경로가 아니다(잔반 인증은 Backend의 관리자 승인
    기능). 연구 결과가 운영 경로로 들어오지 않았음을 이 경계로 확인한다.
    """
    assert not (ROOT / "src" / "babplanner_ai" / "waste").exists()
    main = (ROOT / "src" / "babplanner_ai" / "main.py").read_text(encoding="utf-8")
    assert "waste" not in main


def test_no_structured_head_is_wired_into_the_service():
    service = ROOT / "src" / "babplanner_ai"
    hits = [
        path
        for path in service.rglob("*.py")
        if "waste_estimation.structured" in path.read_text(encoding="utf-8")
    ]
    assert hits == []


def test_report_keeps_the_fixture_gate_despite_good_results():
    text = report_text()
    assert "WASTE_MODEL_MODE=fixture" in text
    assert "밥플래너 현장 표본은 여전히 **0건**" in text


def test_no_raw_image_is_tracked_under_the_research_tree():
    base = ROOT / "research" / "waste_estimation"
    tracked = [
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png")
        for path in base.rglob(pattern)
        if "data" not in path.parts
    ]
    assert tracked == []


def test_no_weights_or_archives_are_tracked():
    base = ROOT / "research" / "waste_estimation" / "experiments"
    heavy = [
        path
        for pattern in ("*.pt", "*.pth", "*.npy", "*.npz", "*.xlsx", "*.zip")
        for path in base.rglob(pattern)
    ]
    assert heavy == []


def test_recovery_file_carries_no_image_data():
    rows = read_csv("catastrophic_recovery.csv")
    banned = ("image", "path", "photo", "file", "base64")
    for column in rows[0]:
        assert not any(token in column.lower() for token in banned), column
