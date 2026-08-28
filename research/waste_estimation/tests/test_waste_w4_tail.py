"""W4 — 꼬리 인지 회귀와 신뢰 이식의 규약.

여기서 지키는 것은 성능이 아니라 **경계**다.

    구간 경계와 고잔반 기준을 결과 보고 바꾸지 않는다
    가중치 빈도를 학습 fold에서만 센다
    p_high를 자기 라벨로 만들지 않는다
    전체 MAE 가드레일을 없애지 않는다
    참조 분포에 자기 자신을 넣지 않는다
    coverage 이동을 반드시 잰다
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

from waste_estimation.baselines import BaselineError, LogisticRegression, Ridge
from waste_estimation.tail import (
    CATASTROPHIC_MARGIN,
    CATASTROPHIC_NAME,
    HIGH_WASTE_THRESHOLD,
    TailError,
    bin_index,
    catastrophic_rate,
    catastrophic_underestimation,
    coverage_drift,
    inverse_sqrt_bin_weights,
    is_high_waste,
    normalized_novelty,
    realized_coverage,
    reference_loo_distances,
    signed_error,
)

requires_numpy = pytest.mark.skipif(
    importlib.util.find_spec("numpy") is None,
    reason="numpy가 필요합니다 (.venv-waste에서 돕니다)",
)

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "research" / "waste_estimation" / "experiments" / "w4_tail"
REPORT = ROOT / "reports" / "waste_w4_tail_aware_regression.md"
SCRIPTS = ROOT / "research" / "waste_estimation" / "scripts"

#: W2가 미리 고정한 등간격 5구간. W4는 이것을 그대로 쓴다.
EDGES = (0.2, 0.4, 0.6, 0.8)


def read_json(name: str) -> dict:
    path = EXP / name
    if not path.exists():
        pytest.skip(f"{path.name}이 아직 없습니다. W4 스크립트를 먼저 돌리세요.")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict]:
    path = EXP / name
    if not path.exists():
        pytest.skip(f"{path.name}이 아직 없습니다. W4 스크립트를 먼저 돌리세요.")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def report_text() -> str:
    if not REPORT.exists():
        pytest.skip("W4 보고서가 아직 없습니다.")
    return " ".join(REPORT.read_text(encoding="utf-8").split())


# --- 정의 -------------------------------------------------------------------


def test_bins_are_the_ones_w2_fixed_in_advance():
    """구간을 W4에서 새로 만들지 않았다."""
    policy = read_json("freeze.json")
    assert tuple(policy["binEdges"]) == EDGES
    assert "W2" in policy["binNote"]


def test_high_waste_threshold_is_the_last_bin_edge():
    assert HIGH_WASTE_THRESHOLD == EDGES[-1] == 0.8
    assert read_json("freeze.json")["highWasteThreshold"] == HIGH_WASTE_THRESHOLD


@requires_numpy
def test_high_waste_includes_the_boundary_value():
    """`>= 0.8`이다. 경계 규칙이 바뀌면 표본 수가 조용히 달라진다."""
    import numpy as np

    assert is_high_waste(np.array([0.79, 0.8, 0.81])).tolist() == [False, True, True]


@requires_numpy
def test_catastrophic_needs_both_high_waste_and_a_large_gap():
    """큰 오차만으로도, 고잔반만으로도 성립하지 않는다."""
    import numpy as np

    targets = np.array([0.9, 0.9, 0.5, 0.95])
    predictions = np.array([0.55, 0.85, 0.1, 0.60])
    assert catastrophic_underestimation(predictions, targets).tolist() == [
        True,  # 고잔반 + 0.35 낮게
        False,  # 고잔반이지만 0.05 차이
        False,  # 0.40 차이지만 고잔반 아님
        True,  # 고잔반 + 0.35 낮게
    ]


@requires_numpy
def test_catastrophic_ignores_overestimation():
    """높게 부른 것은 치명적 과소추정이 아니다. 방향이 있는 정의다."""
    import numpy as np

    assert not catastrophic_underestimation(np.array([1.0]), np.array([0.85]))[0]


@requires_numpy
def test_catastrophic_rate_divides_by_high_waste_not_by_everything():
    """전체로 나누면 고잔반이 드문 집합에서 비율이 저절로 작아 보인다."""
    import numpy as np

    targets = np.array([0.9, 0.9, 0.1, 0.1, 0.1])
    predictions = np.array([0.5, 0.85, 0.1, 0.1, 0.1])
    assert catastrophic_rate(predictions, targets) == pytest.approx(0.5)


@requires_numpy
def test_catastrophic_rate_refuses_a_subgroup_with_no_high_waste():
    import numpy as np

    with pytest.raises(TailError):
        catastrophic_rate(np.array([0.1]), np.array([0.1]))


@requires_numpy
def test_signed_error_keeps_the_direction():
    import numpy as np

    assert signed_error(np.array([0.3]), np.array([0.9]))[0] == pytest.approx(-0.6)


def test_catastrophic_name_is_the_same_in_code_and_freeze():
    assert read_json("freeze.json")["catastrophicName"] == CATASTROPHIC_NAME
    assert read_json("freeze.json")["catastrophicMargin"] == CATASTROPHIC_MARGIN


# --- 가중치 ------------------------------------------------------------------


@requires_numpy
def test_bin_index_puts_the_boundary_in_the_lower_bin():
    import numpy as np

    assert bin_index(np.array([0.0, 0.2, 0.21, 0.8, 0.81]), EDGES).tolist() == [
        0,
        0,
        1,
        3,
        4,
    ]


@requires_numpy
def test_rare_bins_get_more_weight_than_common_ones():
    import numpy as np

    targets = np.array([0.1] * 9 + [0.9])
    weights = inverse_sqrt_bin_weights(targets, EDGES)
    assert weights[-1] > weights[0]


@requires_numpy
def test_weights_are_inverse_sqrt_not_full_inverse():
    """완전 역빈도라면 비가 9배다. 역제곱근이면 3배다."""
    import numpy as np

    targets = np.array([0.1] * 9 + [0.9])
    weights = inverse_sqrt_bin_weights(targets, EDGES)
    assert weights[-1] / weights[0] == pytest.approx(3.0)


@requires_numpy
def test_weights_are_normalised_to_mean_one():
    """평균이 1이 아니면 같은 alpha가 사실상 다른 정규화 세기가 된다."""
    import numpy as np

    targets = np.array([0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95])
    assert float(inverse_sqrt_bin_weights(targets, EDGES).mean()) == pytest.approx(1.0)


@requires_numpy
def test_weights_depend_only_on_the_labels_they_are_given():
    """평가 fold를 섞어 넣으면 다른 값이 나온다 — 그래서 학습 fold만 넘겨야 한다."""
    import numpy as np

    train = np.array([0.1] * 9 + [0.9])
    contaminated = np.concatenate([train, np.array([0.9] * 5)])
    assert not np.allclose(
        inverse_sqrt_bin_weights(train, EDGES)[:10],
        inverse_sqrt_bin_weights(contaminated, EDGES)[:10],
    )


def test_freeze_says_weights_come_from_training_labels_only():
    policy = read_json("freeze.json")
    assert policy["weightComputedOn"] == "training fold labels only"
    assert policy["weightCandidates"] == ["inverse_sqrt_bin_frequency"]


@requires_numpy
def test_weighted_ridge_reduces_to_the_unweighted_one():
    """가중치를 안 주면 W2·W3이 쓰던 계산과 완전히 같아야 한다."""
    import numpy as np

    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 6))
    targets = rng.normal(size=40)
    plain = Ridge(alpha=1.0).fit(features, targets).predict(features)
    ones = (
        Ridge(alpha=1.0)
        .fit(features, targets, sample_weight=np.ones(40))
        .predict(features)
    )
    assert np.allclose(plain, ones)


@requires_numpy
def test_weighted_ridge_moves_toward_the_upweighted_samples():
    import numpy as np

    features = np.array([[0.0], [1.0], [2.0]])
    targets = np.array([0.0, 0.0, 1.0])
    weights = np.array([0.1, 0.1, 2.8])
    plain = Ridge(alpha=0.01).fit(features, targets).predict(np.array([[2.0]]))
    tilted = (
        Ridge(alpha=0.01)
        .fit(features, targets, sample_weight=weights)
        .predict(np.array([[2.0]]))
    )
    assert tilted[0] > plain[0]


@requires_numpy
def test_weighted_ridge_refuses_negative_weights():
    import numpy as np

    with pytest.raises(BaselineError):
        Ridge(alpha=1.0).fit(
            np.array([[0.0], [1.0]]), np.array([0.0, 1.0]), sample_weight=[-1.0, 1.0]
        )


# --- 보조 분류기 --------------------------------------------------------------


@requires_numpy
def test_auxiliary_label_is_the_high_waste_indicator():
    import numpy as np

    targets = np.array([0.1, 0.85, 0.8, 0.79])
    assert is_high_waste(targets).astype(int).tolist() == [0, 1, 1, 0]


@requires_numpy
def test_classifier_separates_a_trivially_separable_problem():
    import numpy as np

    rng = np.random.default_rng(1)
    features = np.vstack([rng.normal(-2, 0.4, (40, 3)), rng.normal(2, 0.4, (40, 3))])
    labels = np.array([0.0] * 40 + [1.0] * 40)
    model = LogisticRegression(alpha=1.0, class_weight="balanced").fit(features, labels)
    assert (model.predict(features) == labels).mean() > 0.95


@requires_numpy
def test_classifier_regularisation_keeps_probabilities_off_the_rails():
    """정규화가 없으면 완전 분리에서 확률이 0과 1로 붙고 stacking 입력이 죽는다."""
    import numpy as np

    rng = np.random.default_rng(2)
    features = np.vstack([rng.normal(-3, 0.2, (20, 40)), rng.normal(3, 0.2, (20, 40))])
    labels = np.array([0.0] * 20 + [1.0] * 20)
    probability = (
        LogisticRegression(alpha=100.0).fit(features, labels).predict_proba(features)
    )
    assert probability.min() > 1e-6
    assert probability.max() < 1.0 - 1e-6


@requires_numpy
def test_classifier_refuses_a_single_class():
    import numpy as np

    with pytest.raises(BaselineError):
        LogisticRegression().fit(np.array([[0.0], [1.0]]), np.array([1.0, 1.0]))


def test_classifier_primary_metric_is_recall_not_auroc():
    """제품 위험은 고잔반을 놓치는 쪽이다."""
    policy = read_json("freeze.json")
    assert "recall" in policy["classifierPrimaryMetric"]
    assert "AUROC만 강조하지 않는다" in policy["classifierMetricNote"]


def test_classifier_metrics_are_recorded_per_fold():
    rows = read_csv("high_waste_metrics.csv")
    assert len(rows) >= 2
    for row in rows:
        for key in ("recall", "precision", "f1", "auroc"):
            assert row[key] not in ("", None), key


# --- 중첩 분할과 누수 ---------------------------------------------------------


def test_outer_folds_match_the_ones_w2_and_w3_used():
    policy = read_json("freeze.json")
    assert policy["outerFolds"] == 5
    assert policy["outerSeed"] == 42
    assert policy["groupKey"] == "food category"


def test_outer_split_had_no_group_leakage():
    assert read_json("metadata.json")["outerGroupLeakage"] == 0


def test_inner_folds_are_only_for_p_high_not_for_selection():
    policy = read_json("freeze.json")
    assert policy["innerFolds"] >= 2
    assert "모델 선택에 쓰지 않는다" in policy["nestedNote"]


def test_training_p_high_comes_from_an_inner_out_of_fold_model():
    """자기 라벨로 학습한 분류기의 확률을 자기 특징에 붙이면 누수다."""
    source = (SCRIPTS / "w4_experiment.py").read_text(encoding="utf-8")
    assert "inner_oof_probability" in source
    assert "누수" in source
    assert "안쪽 group OOF" in read_json("metadata.json")["note"]


@requires_numpy
def test_stacked_p_high_would_leak_if_taken_from_the_same_labels():
    """누수 시나리오가 실제로 정답을 흘리는지 확인한다.

    특징을 **순수한 잡음**으로 둔다. 배울 것이 아무것도 없으므로 정직한
    분류기라면 다수 클래스 비율밖에 못 낸다. 그런데 자기 라벨로 맞춘 확률은
    그보다 훨씬 잘 맞힌다 — 그 초과분이 곧 정답이 새어 나온 양이다.

    그 확률을 특징에 붙이면 회귀기는 임베딩이 아니라 그 정답을 읽는다.
    """
    import numpy as np

    rng = np.random.default_rng(3)
    features = rng.normal(size=(60, 30))
    labels = (rng.random(60) < 0.3).astype(float)
    majority = max(labels.mean(), 1.0 - labels.mean())
    leaked = (
        LogisticRegression(alpha=0.1).fit(features, labels).predict_proba(features)
    )
    accuracy = ((leaked >= 0.5).astype(float) == labels).mean()
    assert accuracy > majority + 0.1, (accuracy, majority)


def test_no_hyperparameter_search_was_rerun():
    policy = read_json("freeze.json")
    assert policy["hyperparameterSearch"] is False
    assert policy["ridgeAlpha"] == 100.0


def test_category_id_is_not_a_model_input():
    joined = " ".join(read_json("freeze.json")["forbidden"])
    assert "category ID를 모델 입력으로 사용" in joined


# --- 선택 규칙 ---------------------------------------------------------------


def test_freeze_hash_matches_the_record():
    record = read_json("freeze.json")
    stored = record.pop("sha256OfRecordWithoutHash")
    payload = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == stored


def test_freeze_records_everything_the_spec_required():
    policy = read_json("freeze.json")
    for key in (
        "candidates",
        "binEdges",
        "weightFormula",
        "highWasteThreshold",
        "catastrophicRule",
        "outerFolds",
        "innerFolds",
        "selectionRule",
        "reliability",
    ):
        assert key in policy, key
    assert len(policy["candidates"]) == 3


def test_overall_mae_guardrail_exists_so_the_tail_cannot_be_bought_at_any_price():
    """가드레일이 없으면 '전부 0.9라고 답하는 모델'이 꼬리 지표로 이긴다."""
    policy = read_json("freeze.json")
    assert policy["overallGuardMargin"] == 0.02
    assert "Gate 1" in policy["selectionRule"][0]


def test_selection_rule_puts_the_tail_second_not_first():
    rule = read_json("freeze.json")["selectionRule"]
    assert "탈락" in rule[0]
    assert "고잔반 MAE가 가장 낮은" in rule[1]


def test_the_guardrail_actually_rejected_a_candidate():
    """규칙이 장식이 아니었는지 결과로 확인한다."""
    meta = read_json("metadata.json")
    assert "R1_BALANCED" not in meta["gate1Passed"]
    assert meta["selected"] in meta["gate1Passed"]


def test_fold_stability_is_reported_so_one_lucky_fold_cannot_carry_a_claim():
    rows = read_csv("fold_stability.csv")
    assert rows
    for row in rows:
        assert int(row["foldsCompared"]) >= 2
        assert row["worstDelta"] not in ("", None)


def test_high_waste_bias_is_reported_not_just_absolute_error():
    rows = read_csv("candidate_summary.csv")
    for row in rows:
        assert row["highBias"] not in ("", None)
        assert float(row["highBias"]) < 0, "고잔반에서 과소추정이 사라졌다면 확인하세요"


def test_catastrophic_rate_is_reported_for_every_candidate():
    for row in read_csv("candidate_summary.csv"):
        assert row["catastrophicRate"] not in ("", None)


def test_low_and_medium_are_reported_so_the_tail_fix_cannot_hide_damage():
    for row in read_csv("candidate_summary.csv"):
        assert row["lowMae"] not in ("", None)
        assert row["mediumMae"] not in ("", None)


# --- 신뢰 이식 ---------------------------------------------------------------


@requires_numpy
def test_reference_distribution_excludes_self_distance():
    """자기 자신을 세면 거리 0이 섞여 분포 전체가 왼쪽으로 밀린다."""
    import numpy as np

    reference = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.2]])
    distances = reference_loo_distances(reference, k=1)
    assert (distances > 1e-9).all()


@requires_numpy
def test_reference_distribution_refuses_too_few_neighbours():
    import numpy as np

    with pytest.raises(TailError):
        reference_loo_distances(np.array([[1.0, 0.0], [0.0, 1.0]]), k=5)


@requires_numpy
def test_normalized_novelty_is_a_percentile_between_zero_and_one():
    import numpy as np

    rng = np.random.default_rng(4)
    reference = rng.normal(size=(60, 8))
    scores = normalized_novelty(rng.normal(size=(10, 8)), reference, k=3)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


@requires_numpy
def test_normalized_novelty_survives_a_reference_size_change_better_than_raw():
    """W3이 걸린 자리다. 참조가 커지면 raw 거리가 통째로 짧아진다."""
    import numpy as np

    from waste_estimation.reliability import knn_novelty

    rng = np.random.default_rng(5)
    pool = rng.normal(size=(400, 12))
    query = rng.normal(size=(80, 12))
    small, large = pool[:100], pool

    raw_shift = abs(
        float(np.median(knn_novelty(query, large, 3)))
        - float(np.median(knn_novelty(query, small, 3)))
    )
    norm_shift = abs(
        float(np.median(normalized_novelty(query, large, 3)))
        - float(np.median(normalized_novelty(query, small, 3)))
    )
    assert norm_shift < raw_shift


@requires_numpy
def test_coverage_drift_is_the_absolute_gap():
    assert coverage_drift(0.8, 0.629) == pytest.approx(0.171)
    assert coverage_drift(0.8, 0.8) == 0.0


@requires_numpy
def test_coverage_drift_refuses_values_outside_zero_one():
    with pytest.raises(TailError):
        coverage_drift(1.2, 0.5)


@requires_numpy
def test_realized_coverage_counts_what_the_threshold_actually_accepts():
    import numpy as np

    assert realized_coverage(np.array([0.1, 0.2, 0.3, 0.4]), 0.25) == pytest.approx(0.5)


def test_raw_method_was_kept_not_deleted():
    """W3의 방식을 지우지 않는다. 비교 대상이 없으면 개선을 말할 수 없다."""
    rows = read_csv("reliability_transport_summary.csv")
    assert {row["method"] for row in rows} == {"C0_RAW", "C0_NORMALIZED"}


def test_coverage_drift_is_measured_at_every_target():
    policy = read_json("freeze.json")
    rows = read_csv("reliability_transport_summary.csv")
    targets = {float(row["targetCoverage"]) for row in rows}
    assert targets == {float(value) for value in policy["reliability"]["coverageTargets"]}
    for row in rows:
        assert row["meanCoverageDrift"] not in ("", None)


def test_normalized_was_adopted_only_by_the_frozen_rule():
    decision = read_json("reliability_decision.json")
    if decision["adoptNormalized"]:
        assert (
            decision["normalizedMeanAbsoluteCoverageDrift"]
            < decision["rawMeanAbsoluteCoverageDrift"]
        )
        assert (
            decision["normalizedAcceptedMaeAt80"]
            <= decision["rawAcceptedMaeAt80"] + 0.01
        )


def test_reliability_is_not_claimed_as_a_high_waste_safety_mechanism():
    """네 번 재서 전부 0 근처였다. 그 결론을 흐리지 않는다."""
    decision = read_json("reliability_decision.json")
    assert decision["tailSafetyMechanism"] is False
    for row in read_csv("tail_reliability.csv"):
        if row["scope"] == "high_waste_only":
            assert abs(float(row["spearman"])) < 0.2
    assert "고잔반 안전장치가 아니다" in report_text()


def test_score_is_not_called_a_probability():
    decision = read_json("reliability_decision.json")
    assert decision["calibrated"] is False
    assert "확률이 아니다" in decision["calibrationNote"]
    assert "보정된 정확도 확률이 아니다" in report_text()


# --- 이미 본 집합 -------------------------------------------------------------


def test_analysis_split_is_not_called_a_test():
    meta = read_json("analysis_metadata.json")
    assert meta["isFinalTest"] is False
    assert meta["isGeneralizationTest"] is False
    assert meta["isPristineValidation"] is False
    assert meta["resultName"] == "POST-HOC ANALYSIS"


def test_analysis_was_applied_exactly_once():
    assert read_json("analysis_metadata.json")["applications"] == 1


def test_selection_was_made_before_the_analysis_split_was_opened():
    assert read_json("metadata.json")["analysisOpened"] is False
    assert (
        read_json("analysis_metadata.json")["selectionMadeOn"]
        == "DEVELOPMENT nested group CV"
    )


def test_analysis_uses_the_same_frozen_policy_hash():
    policy_hash = read_json("freeze.json")["sha256OfRecordWithoutHash"]
    assert read_json("metadata.json")["policyHash"] == policy_hash
    assert read_json("analysis_metadata.json")["policyHash"] == policy_hash


def test_analysis_script_cannot_select_anything():
    source = (SCRIPTS / "w4_analysis.py").read_text(encoding="utf-8")
    assert "여기서는 아무것도 고르지 않는다" in source
    assert "argmin" not in source
    assert 'selected = development["selected"]' in source


def test_analysis_records_which_candidate_was_the_selected_one():
    rows = read_csv("analysis_metrics.csv")
    selected = [row for row in rows if row["isSelected"] == "True"]
    assert len(selected) == 1
    assert selected[0]["candidate"] == read_json("metadata.json")["selected"]


def test_freeze_forbids_choosing_anything_on_the_analysis_split():
    joined = " ".join(read_json("freeze.json")["forbidden"])
    for phrase in (
        "W2_ANALYSIS에서 모델 선택",
        "W2_ANALYSIS에서 sample weight 선택",
        "W2_ANALYSIS에서 classifier threshold 선택",
        "W2_ANALYSIS에서 reliability normalization 방식 선택",
    ):
        assert phrase in joined, phrase


# --- 보고 ---------------------------------------------------------------------


def test_report_carries_the_w3_limitation_forward():
    text = report_text()
    assert "0.250 -> 0.250" in text or "0.250에서 0.250" in text
    assert "19건 중 10건" in text


def test_report_states_this_is_not_a_final_test():
    text = report_text()
    assert "POST-HOC ANALYSIS" in text
    assert "FINAL TEST도" in text


def test_report_records_all_three_hypothesis_verdicts():
    text = report_text()
    for name in ("H-W4-A", "H-W4-B", "H-W4-C"):
        assert name in text


def test_report_admits_the_candidate_ideas_came_from_consumed_data():
    assert "소비된 데이터에서 나왔다" in report_text()


def test_report_states_this_is_not_cafeteria_performance():
    assert "밥플래너 급식소 성능이 아니다" in report_text()


def test_report_discloses_the_boundary_correction():
    text = report_text()
    assert "89건" in text and "90건" in text


def test_charts_exist_for_the_three_required_views():
    assets = ROOT / "reports" / "assets"
    for name in (
        "waste_w4_tradeoff.svg",
        "waste_w4_high_waste_scatter.svg",
        "waste_w4_coverage_drift.svg",
    ):
        assert (assets / name).exists(), name


# --- production 경계 -----------------------------------------------------------


def test_fixture_mode_was_not_replaced():
    from babplanner_ai.waste.fixture_estimator import FixtureWasteEstimator

    assert FixtureWasteEstimator is not None


def test_no_w4_research_code_is_wired_into_the_service():
    service = ROOT / "src" / "babplanner_ai"
    hits = [
        path
        for path in service.rglob("*.py")
        if "waste_estimation.tail" in path.read_text(encoding="utf-8")
    ]
    assert hits == []


def test_no_raw_image_is_tracked_under_the_research_tree():
    base = ROOT / "research" / "waste_estimation"
    tracked = [
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png")
        for path in base.rglob(pattern)
        if "data" not in path.parts
    ]
    assert tracked == []


def test_no_model_weights_or_embeddings_are_tracked():
    base = ROOT / "research" / "waste_estimation" / "experiments"
    heavy = [
        path
        for pattern in ("*.pt", "*.pth", "*.npy", "*.npz", "*.xlsx", "*.zip")
        for path in base.rglob(pattern)
    ]
    assert heavy == []


def test_failure_cases_carry_no_image_data():
    rows = read_csv("failure_cases.csv")
    banned = ("image", "path", "photo", "file", "base64")
    for column in rows[0]:
        assert not any(token in column.lower() for token in banned), column
