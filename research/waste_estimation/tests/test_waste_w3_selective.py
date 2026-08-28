"""W3 — 선택적 예측의 규약.

여기서 지키는 것은 성능이 아니라 **경계**다.

    정답을 신뢰 신호에 넣지 않는다
    자기 자신을 이웃으로 세지 않는다
    coverage 없이 위험을 말하지 않는다
    oracle을 실행 가능한 방법처럼 적지 않는다
    confidence를 확률이라고 부르지 않는다
    이미 본 집합을 다시 시험지라고 부르지 않는다
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

from waste_estimation.reliability import (
    ReliabilityError,
    Signal,
    aurc,
    combine,
    knn_novelty,
    oracle_rejection_risk,
    prediction_extremeness,
    random_rejection_risk,
    rank_normalize,
    risk_at_coverage_points,
    spearman,
)

requires_numpy = pytest.mark.skipif(
    importlib.util.find_spec("numpy") is None,
    reason="numpy가 필요합니다 (.venv-waste에서 돕니다)",
)

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "research" / "waste_estimation" / "experiments" / "w3_selective"
W2 = ROOT / "research" / "waste_estimation" / "experiments" / "w2_external"
REPORT = ROOT / "reports" / "waste_w3_selective_prediction.md"
SCRIPTS = ROOT / "research" / "waste_estimation" / "scripts"


def read_json(name: str, base: Path = EXP) -> dict:
    path = base / name
    if not path.exists():
        pytest.skip(f"{path.name}이 아직 없습니다. W3 스크립트를 먼저 돌리세요.")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(name: str, base: Path = EXP) -> list[dict]:
    path = base / name
    if not path.exists():
        pytest.skip(f"{path.name}이 아직 없습니다. W3 스크립트를 먼저 돌리세요.")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def report_text() -> str:
    if not REPORT.exists():
        pytest.skip("W3 보고서가 아직 없습니다.")
    return " ".join(REPORT.read_text(encoding="utf-8").split())


# --- 신호의 정의 ------------------------------------------------------------


def test_signal_names_are_stable_identifiers():
    """기록에 남는 이름이다. 바뀌면 과거 실험과 대조가 끊긴다."""
    assert Signal.EMBEDDING_NOVELTY.value == "C0_embedding_novelty"
    assert Signal.FOLD_DISAGREEMENT.value == "C1_fold_disagreement"
    assert Signal.PREDICTION_EXTREMENESS.value == "C2_prediction_extremeness"
    assert Signal.COMBINED.value == "combined"


@requires_numpy
def test_novelty_is_zero_for_a_point_already_in_the_reference():
    """참조에 같은 방향의 벡터가 있으면 코사인 거리는 0이다.

    이 성질 때문에 **자기 자신을 참조에 넣으면 신호가 무너진다.** 아래 두
    테스트가 그 규칙을 지킨다.
    """
    import numpy as np

    reference = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    scores = knn_novelty(np.array([[1.0, 0.0]]), reference, k=1)
    assert scores[0] == pytest.approx(0.0, abs=1e-9)


@requires_numpy
def test_novelty_grows_with_distance_from_the_reference():
    import numpy as np

    reference = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
    near = knn_novelty(np.array([[1.0, 0.05]]), reference, k=2)[0]
    far = knn_novelty(np.array([[-1.0, 0.0]]), reference, k=2)[0]
    assert far > near


@requires_numpy
def test_oof_scores_never_used_the_samples_own_fold_as_reference():
    """OOF 참조에서 자기 fold를 뺐는지 결과로 확인한다.

    자기 임베딩이 참조에 섞였다면 최근접 거리가 0에 가까워지고, 신호가
    거의 전부 0 근처로 몰린다. 그런 분포가 아닌지 본다.
    """
    rows = read_csv("oof_predictions.csv")
    scores = [float(row["C0_knn3"]) for row in rows]
    assert min(scores) > 1e-6, "novelty가 0인 표본이 있습니다 — 자기 참조 의심"


def test_oof_used_group_folds_with_zero_leakage():
    meta = read_json("metadata.json")
    assert meta["groupLeakage"] == 0
    assert meta["folds"] >= 2


def test_oof_did_not_retune_the_estimator():
    """W3은 추정기를 다시 고르지 않는다. W2가 고른 것을 그대로 쓴다."""
    meta = read_json("metadata.json")
    assert meta["baseEstimator"]["retuned"] is False
    w2 = read_json("development_selection.json", base=W2)["after"]
    assert meta["baseEstimator"]["backbone"] == w2["backbone"]
    assert float(meta["baseEstimator"]["alpha"]) == float(w2["alpha"])


def test_reliability_features_do_not_include_ground_truth():
    """정답이나 잔차가 신호 열에 섞이면 그건 예측이 아니라 채점이다."""
    rows = read_csv("oof_predictions.csv")
    signal_columns = [
        name
        for name in rows[0]
        if name.startswith("C0_") or name.startswith("C1_") or name.startswith("C2_")
    ]
    assert signal_columns
    for row in rows[:50]:
        for column in signal_columns:
            assert float(row[column]) != float(row["absError"])
            assert float(row[column]) != float(row["target"])


@requires_numpy
def test_extremeness_uses_predictions_not_targets():
    """C2는 예측과 학습 평균만 본다. 정답을 넣을 자리가 없다."""
    import numpy as np

    scores = prediction_extremeness(np.array([0.1, 0.5, 0.9]), 0.5)
    assert scores.tolist() == pytest.approx([0.4, 0.0, 0.4])


@requires_numpy
def test_extremeness_is_kept_as_a_control_and_lost():
    """대조군이 이겼다면 C0/C1의 근거가 없다는 뜻이다. 실제로는 졌다."""
    rows = {row["signal"]: row for row in read_csv("reliability_scores.csv")}
    assert float(rows["C2_extremeness"]["aurc"]) > float(rows["C0_knn3"]["aurc"])


# --- 결합 ------------------------------------------------------------------


@requires_numpy
def test_rank_normalize_maps_to_zero_one_preserving_order():
    import numpy as np

    values = rank_normalize(np.array([5.0, 1.0, 3.0]))
    assert values.tolist() == pytest.approx([1.0, 0.0, 0.5])


@requires_numpy
def test_combine_needs_at_least_two_signals():
    import numpy as np

    with pytest.raises(ReliabilityError):
        combine(np.array([1.0, 2.0, 3.0]))


@requires_numpy
def test_combine_does_not_let_scale_dominate():
    """축척이 1000배 다른 두 신호를 합쳐도 한쪽이 결과를 지배하지 않는다."""
    import numpy as np

    small = np.array([0.001, 0.002, 0.003])
    huge = np.array([3000.0, 2000.0, 1000.0])
    result = combine(small, huge)
    assert result.tolist() == pytest.approx([0.5, 0.5, 0.5])


def test_combination_rule_is_not_a_learned_meta_model():
    policy = read_json("freeze.json")
    assert "메타 모델을 학습하지 않는다" in policy["combinationRule"]


# --- coverage와 위험 --------------------------------------------------------


@requires_numpy
def test_zero_coverage_has_no_risk_instead_of_zero_risk():
    """coverage 0에서 0을 돌려주면 아무것도 안 한 정책이 완벽해 보인다."""
    import numpy as np

    result = risk_at_coverage_points(
        np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0, 3.0]), (0.0, 1.0)
    )
    assert result[0.0] is None
    assert result[1.0] == pytest.approx(0.2)


@requires_numpy
def test_risk_at_full_coverage_equals_the_overall_mae():
    import numpy as np

    errors = np.array([0.05, 0.4, 0.15, 0.2])
    result = risk_at_coverage_points(errors, np.array([3.0, 1.0, 4.0, 2.0]), (1.0,))
    assert result[1.0] == pytest.approx(errors.mean())


@requires_numpy
def test_risk_accepts_the_most_reliable_samples_first():
    import numpy as np

    errors = np.array([0.9, 0.1, 0.8, 0.2])
    scores = np.array([4.0, 1.0, 3.0, 2.0])  # 낮을수록 믿을 만함
    result = risk_at_coverage_points(errors, scores, (0.5,))
    assert result[0.5] == pytest.approx(0.15)


@requires_numpy
def test_aurc_is_lower_when_the_signal_ranks_errors_correctly():
    import numpy as np

    errors = np.array([0.0, 0.1, 0.2, 0.9])
    good = np.array([0.0, 1.0, 2.0, 3.0])
    bad = good[::-1].copy()
    assert aurc(errors, good) < aurc(errors, bad)


@requires_numpy
def test_aurc_rejects_mismatched_lengths():
    import numpy as np

    with pytest.raises(ReliabilityError):
        aurc(np.array([0.1, 0.2]), np.array([1.0]))


@requires_numpy
def test_random_rejection_stays_near_the_overall_mae():
    """무작위로 버려도 MAE는 안 내려간다. 이 대조가 없으면 아무 말도 못 한다."""
    import numpy as np

    errors = np.linspace(0.0, 1.0, 200)
    value = random_rejection_risk(errors, 0.8, trials=100, seed=7)
    assert value == pytest.approx(errors.mean(), abs=0.02)


@requires_numpy
def test_oracle_is_an_upper_bound_not_a_method():
    """oracle이 실제 신호보다 나쁘면 계산이 잘못된 것이다."""
    import numpy as np

    errors = np.array([0.05, 0.4, 0.15, 0.9])
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    assert oracle_rejection_risk(errors, 0.5) <= risk_at_coverage_points(
        errors, scores, (0.5,)
    )[0.5]


def test_oracle_is_named_as_an_oracle_in_code_and_report():
    source = (
        ROOT
        / "research"
        / "waste_estimation"
        / "src"
        / "waste_estimation"
        / "reliability.py"
    ).read_text(encoding="utf-8")
    assert "ORACLE UPPER BOUND" in source
    assert "실행 가능한 방법이 **아니다.**" in source
    assert "실행할 수 없다" in report_text()


def test_coverage_is_reported_next_to_every_risk_number():
    rows = read_csv("coverage_risk.csv")
    assert rows
    for row in rows:
        assert row["coverage"], "coverage 없는 위험 값이 있습니다"


def test_reported_coverage_points_match_the_frozen_ones():
    policy = read_json("freeze.json")
    recorded = {float(row["coverage"]) for row in read_csv("coverage_risk.csv")}
    assert recorded == {float(value) for value in policy["coveragePoints"]}


# --- 상관 ------------------------------------------------------------------


@requires_numpy
def test_spearman_is_one_for_a_monotone_but_nonlinear_relation():
    """Pearson이라면 1이 안 나온다. 순위 상관을 쓰는 이유다."""
    import numpy as np

    first = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(first, first**3) == pytest.approx(1.0)


@requires_numpy
def test_spearman_averages_tied_ranks():
    import numpy as np

    # 동점을 무시하면 상관이 부풀려진다.
    value = spearman(np.array([1.0, 1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0, 4.0]))
    assert 0.0 < value < 1.0


@requires_numpy
def test_spearman_refuses_a_constant_input():
    import numpy as np

    with pytest.raises(ReliabilityError):
        spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))


def test_selected_signal_correlates_positively_with_error():
    rows = {row["signal"]: row for row in read_csv("reliability_scores.csv")}
    policy = read_json("freeze.json")
    chosen = rows[policy["selectedSignal"]]
    assert float(chosen["spearman"]) > 0


# --- 동결 ------------------------------------------------------------------


def test_freeze_hash_matches_the_record():
    record = read_json("freeze.json")
    stored = record.pop("sha256OfRecordWithoutHash")
    payload = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == stored


def test_freeze_records_everything_needed_to_rerun_the_policy():
    policy = read_json("freeze.json")
    for key in (
        "baseEstimator",
        "selectedSignal",
        "knnCandidates",
        "ensembleMembers",
        "combinationRule",
        "coveragePoints",
        "selectionPrimary",
        "thresholds",
    ):
        assert key in policy, key


def test_freeze_says_the_selection_criterion_was_fixed_in_advance():
    policy = read_json("freeze.json")
    assert "AURC" in policy["selectionPrimary"]
    assert policy["developmentOnly"] is True


def test_freeze_forbids_tuning_on_the_analysis_split():
    policy = read_json("freeze.json")
    joined = " ".join(policy["forbidden"])
    assert "ANALYSIS" in joined
    assert "coverage 없이" in joined


def test_thresholds_come_from_development_not_from_the_analysis_split():
    policy = read_json("freeze.json")
    assert "DEVELOPMENT" in policy["thresholdSource"]
    assert "다시 잡지 않는다" in policy["thresholdSource"]


def test_thresholds_increase_with_coverage():
    policy = read_json("freeze.json")
    values = [policy["thresholds"][f"{c:.2f}"] for c in sorted(policy["coveragePoints"])]
    assert values == sorted(values)


# --- confidence는 확률이 아니다 ----------------------------------------------


def test_policy_states_the_score_is_not_calibrated():
    policy = read_json("freeze.json")
    assert policy["calibrated"] is False
    assert "확률이 아니다" in policy["calibrationNote"]


def test_report_refuses_to_call_the_score_a_probability():
    text = report_text()
    assert "확률이 아니다" in text
    assert "보정하지 않았고" in text


def test_report_forbids_percentage_confidence_copy_in_the_ui():
    text = report_text()
    assert "신뢰 점수를 퍼센트로 바꿔서 보여 주는 것" in text


# --- 이미 본 집합을 시험지라고 부르지 않는다 ----------------------------------


def test_analysis_split_is_not_labelled_as_a_test():
    meta = read_json("analysis_metadata.json")
    assert meta["isFinalTest"] is False
    assert meta["isGeneralizationTest"] is False
    assert meta["split"] == "W2_ANALYSIS"


def test_analysis_script_cannot_select_a_signal():
    """정책을 고를 수 있는 코드 경로 자체를 두지 않는다."""
    source = (SCRIPTS / "w3_analysis.py").read_text(encoding="utf-8")
    assert "freeze.json" in source
    assert "argmin" not in source
    assert "quantile" not in source, "분석 집합에서 분위수를 다시 잡고 있습니다"


def test_analysis_reused_the_w2_predictions_without_refitting():
    meta = read_json("analysis_metadata.json")
    assert meta["refit"] is False
    assert "B2_after" in meta["predictionSource"]


def test_analysis_used_the_frozen_policy_hash():
    assert (
        read_json("analysis_metadata.json")["policyHash"]
        == read_json("freeze.json")["sha256OfRecordWithoutHash"]
    )


def test_w2_report_records_the_reclassification():
    text = " ".join(
        (ROOT / "reports" / "waste_w2_external_benchmark.md").read_text(
            encoding="utf-8"
        ).split()
    )
    assert "재분류" in text
    assert "W2_ANALYSIS" in text


# --- 숨김 방지 --------------------------------------------------------------


def test_high_waste_rejection_rate_is_reported_alongside_the_gain():
    """MAE 하락이 구성 변화인지 볼 수 있어야 한다."""
    rows = read_csv("high_waste_selective.csv")
    at_80 = [
        row
        for row in rows
        if row["signal"] == "C0_knn3" and abs(float(row["coverage"]) - 0.8) < 1e-9
    ]
    assert at_80
    assert float(at_80[0]["highRejectionRate"]) > 0.2


def test_level_controlled_curve_exists_so_the_gain_can_be_discounted():
    rows = read_csv("stratified_coverage_risk.csv")
    assert {row["signal"] for row in rows}
    assert any(abs(float(row["coverage"]) - 0.8) < 1e-9 for row in rows)


def test_report_admits_part_of_the_gain_is_composition():
    text = report_text()
    assert "구성 변화" in text
    assert "수준을 통제" in text


def test_report_records_the_large_errors_that_were_approved():
    text = report_text()
    assert "승인해서 놓친 것" in text


def test_failure_cases_carry_no_image_data():
    rows = read_csv("analysis_failure_cases.csv")
    banned = ("image", "path", "photo", "file", "base64")
    for column in rows[0]:
        assert not any(token in column.lower() for token in banned), column


def test_report_states_this_is_not_cafeteria_performance():
    text = report_text()
    assert "밥플래너 급식소 성능이 아니다" in text


def test_report_records_the_hypothesis_verdict():
    assert "H-W3" in report_text()


def test_report_lists_the_threats_to_validity():
    text = report_text()
    assert "Threats to Validity" in text
    assert "새 시험 집합이 없다" in text


# --- production 경계 ---------------------------------------------------------


def test_fixture_mode_was_not_replaced():
    from babplanner_ai.waste.fixture_estimator import FixtureWasteEstimator

    assert FixtureWasteEstimator is not None


def test_no_reliability_code_is_wired_into_the_service():
    service = ROOT / "src" / "babplanner_ai"
    hits = [
        path
        for path in service.rglob("*.py")
        if "waste_estimation.reliability" in path.read_text(encoding="utf-8")
    ]
    assert hits == []


def test_no_raw_lefood_image_is_tracked():
    tracked = list((ROOT / "research" / "waste_estimation").rglob("*.jpg"))
    tracked += list((ROOT / "research" / "waste_estimation").rglob("*.jpeg"))
    tracked += list((ROOT / "research" / "waste_estimation").rglob("*.png"))
    visible = [path for path in tracked if "data" not in path.parts]
    assert visible == []


def test_charts_exist_for_the_three_required_views():
    assets = ROOT / "reports" / "assets"
    for name in (
        "waste_w3_risk_coverage.svg",
        "waste_w3_reliability_error.svg",
        "waste_w3_high_waste.svg",
    ):
        assert (assets / name).exists(), name
