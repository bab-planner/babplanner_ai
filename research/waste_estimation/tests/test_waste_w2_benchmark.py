"""W2 — 외부 벤치마크의 규약.

여기서 지키는 것은 성능이 아니라 **경계**다.

    외부 데이터의 라벨을 밥플래너 정답이라고 부르지 않는다
    관문을 통과하지 못한 행을 조용히 쓰지 않는다
    분할이 새지 않는다
    지표 전에 몰래 자르지 않는다
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

from waste_estimation.baselines import BaselineError, GlobalMean, Ridge, clip
from waste_estimation.lefood import (
    CITATION,
    DOI,
    LICENSE,
    SOURCE_URL,
    LeFoodRecord,
    RejectionReason,
)
from waste_estimation.metrics import (
    MetricError,
    class_index_mae,
    macro_f1,
    quadratic_weighted_kappa,
    to_class_index,
    unbounded_bias,
    unbounded_mae,
)
from waste_estimation.schemas import waste_rate_from_mass

#: 회귀 baseline은 numpy를 쓴다. 서비스 환경(`.venv`)에는 numpy가 없고 그것이
#: 맞다 — 잔반 연구의 무거운 의존성이 서비스로 새면 안 된다. 그래서 몇 개만
#: numpy가 있는 환경(`.venv-waste`)에서 돌고, 나머지 규약 검사는 어디서든 돈다.
requires_numpy = pytest.mark.skipif(
    importlib.util.find_spec("numpy") is None,
    reason="numpy가 필요합니다 (.venv-waste에서 돕니다)",
)

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "research" / "waste_estimation" / "experiments" / "w2_external"
REPORT = ROOT / "reports" / "waste_w2_external_benchmark.md"


def read_json(name: str) -> dict:
    path = EXP / name
    if not path.exists():
        pytest.skip(f"{name}이 아직 없습니다.")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict]:
    path = EXP / name
    if not path.exists():
        pytest.skip(f"{name}이 아직 없습니다.")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def record(before: float = 200.0, after: float = 50.0) -> LeFoodRecord:
    return LeFoodRecord(
        row_id="1",
        food_category_id="001",
        food_name="Nasi",
        before_image="001_001_DSC_0001_bef.JPG",
        before_mass_g=before,
        after_image="001_001_DSC_0002_aft.JPG",
        after_mass_g=after,
        observer_level=4,
        external_waste_ratio=after / before,
    )


# --- 외부 라벨 유도 ---------------------------------------------------------


def test_external_ratio_uses_the_same_formula_as_the_protocol():
    """계산식은 같다. 그래서 이름을 분리해야 한다."""
    assert record(200.0, 50.0).external_waste_ratio == pytest.approx(0.25)
    assert waste_rate_from_mass(200.0, 50.0) == pytest.approx(0.25)


def test_external_label_is_named_apart_from_the_service_label():
    """`waste_rate`라는 이름을 외부 데이터에 붙이지 않는다."""
    fields = set(vars(record()))
    assert "external_waste_ratio" in fields
    assert "waste_rate" not in fields


def test_group_id_comes_from_the_food_category():
    """참가자/회차 식별자가 없다. 없는 것을 발명하지 않는다."""
    assert record().group_id == "food:001"


def test_invalid_before_mass_is_refused():
    with pytest.raises(Exception):
        waste_rate_from_mass(0.0, 10.0)


def test_leftover_exceeding_served_is_refused():
    with pytest.raises(Exception):
        waste_rate_from_mass(100.0, 150.0)


def test_every_rejection_reason_is_named():
    """왜 버렸는지 셀 수 없으면 그 표본은 되짚을 수 없다."""
    values = {reason.value for reason in RejectionReason}
    assert {"MISSING_IMAGE", "AFTER_EXCEEDS_BEFORE", "PAIR_MISMATCH"} <= values


def test_dataset_citation_names_the_official_source_and_license():
    assert "mendeley" in SOURCE_URL
    assert LICENSE == "CC BY 4.0"
    assert DOI in CITATION


# --- 실제 실행 기록 ---------------------------------------------------------


def test_load_report_rejected_rows_without_images():
    """표에는 678행이 있는데 공개된 이미지는 524쌍뿐이다."""
    metadata = read_json("dataset_metadata.json")
    assert metadata["totalRows"] == 678
    assert metadata["rejections"]["MISSING_IMAGE"] > 0
    assert metadata["usableRecords"] < metadata["totalRows"]


def test_pair_integrity_left_no_mismatched_pairs():
    metadata = read_json("dataset_metadata.json")
    assert metadata["rejections"]["PAIR_MISMATCH"] == 0


def test_split_has_no_group_leakage():
    manifest = read_json("split_manifest.json")
    development = set(manifest["development"]["groups"])
    test = set(manifest["test"]["groups"])
    assert development and test
    assert development & test == set()


def test_no_sample_appears_in_both_splits():
    manifest = read_json("split_manifest.json")
    development = set(manifest["development"]["sampleIds"])
    test = set(manifest["test"]["sampleIds"])
    assert development & test == set()
    assert len(development) == manifest["development"]["count"]
    assert len(test) == manifest["test"]["count"]


def test_freeze_hash_matches_the_record():
    record_ = read_json("freeze.json")
    expected = record_.pop("sha256OfRecordWithoutHash")
    payload = json.dumps(record_, ensure_ascii=False, indent=2, sort_keys=True)
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == expected


def test_freeze_forbids_changing_things_after_seeing_test():
    freeze = read_json("freeze.json")
    joined = " ".join(freeze["forbidden"])
    assert "TEST를 보고" in joined
    assert "지표 전 clip" in joined
    assert freeze["backbone"]["fineTuning"] is False
    assert freeze["preprocessing"]["randomAugmentation"] is False


def test_freeze_records_the_ordinal_bins_before_results():
    freeze = read_json("freeze.json")
    assert freeze["ordinalEdges"] == [0.2, 0.4, 0.6, 0.8]


# --- 지표 규약 --------------------------------------------------------------


@requires_numpy
def test_global_mean_is_the_floor_every_model_must_beat():
    model = GlobalMean().fit(None, [0.0, 0.5, 1.0])
    assert model.predict([1, 2])[0] == pytest.approx(0.5)


@requires_numpy
def test_global_mean_refuses_empty_training_data():
    with pytest.raises(BaselineError):
        GlobalMean().fit(None, [])


@requires_numpy
def test_ridge_standardisation_uses_training_statistics_only():
    """평가 집합의 통계를 쓰면 그 순간 누수다."""
    import numpy as np

    train = np.array([[0.0], [1.0], [2.0]])
    model = Ridge(alpha=0.0).fit(train, np.array([0.0, 1.0, 2.0]))
    # 학습에 없던 범위를 예측해도 학습 통계로 표준화된다.
    far = model.predict(np.array([[10.0]]))
    assert far[0] > 2.0


@requires_numpy
def test_raw_and_clipped_are_reported_separately():
    """지표 전에 몰래 자르면 성능이 좋아 보인다."""
    raw = [-0.2, 0.5, 1.4]
    actual = [0.0, 0.5, 1.0]
    raw_mae = unbounded_mae(actual, raw)
    clipped_mae = unbounded_mae(actual, list(clip(raw)))
    assert clipped_mae < raw_mae


def test_service_metric_would_refuse_raw_regression_output():
    """서비스 지표는 0~1 밖을 거부한다. 그래서 연구용 판이 따로 있다."""
    from babplanner_ai.dataset.metrics import mean_absolute_error

    with pytest.raises(Exception):
        mean_absolute_error([0.0], [-0.2])
    # 제한 없는 판은 계산한다.
    assert unbounded_mae([0.0], [-0.2]) == pytest.approx(0.2)


def test_bias_separates_over_and_under_prediction():
    assert unbounded_bias([0.5, 0.5], [0.3, 0.3]) == pytest.approx(-0.2)
    assert unbounded_bias([0.5, 0.5], [0.7, 0.7]) == pytest.approx(0.2)


def test_ordinal_bin_boundary_is_inclusive_downward():
    edges = [0.2, 0.4, 0.6, 0.8]
    assert to_class_index(0.2, edges) == 0
    assert to_class_index(0.21, edges) == 1
    assert to_class_index(1.0, edges) == 4


def test_class_index_mae_and_macro_f1_disagree_on_purpose():
    """정확도만 보면 안 되는 이유를 지표가 드러내야 한다."""
    truth = [0, 0, 0, 4]
    guess = [0, 0, 0, 0]
    assert class_index_mae(truth, guess) == pytest.approx(1.0)
    # 다수 구간만 맞혀도 정확도는 0.75지만 macro F1은 낮다.
    assert macro_f1(truth, guess, 5) < 0.5


def test_kappa_returns_none_when_undefined():
    assert quadratic_weighted_kappa([0, 0], [0, 0], 2) is None


def test_metrics_refuse_empty_input():
    with pytest.raises(MetricError):
        unbounded_mae([], [])


# --- 결과 파일이 필요한 것을 담고 있는가 ------------------------------------


def test_information_gap_is_measured_on_a_matched_backbone():
    """backbone이 다르면 정보량 차이가 아니라 모델 차이를 잰 것이 된다."""
    rows = read_csv("information_gap_matched.csv")
    assert rows
    backbones = {row["backbone"] for row in rows}
    assert len(backbones) >= 2
    for row in rows:
        assert row["pairArm"] in {"concat", "concat_diff"}
        # 같은 backbone의 after-only와 비교한 값이어야 한다.
        assert float(row["afterTestMae"]) > 0


def test_regression_metrics_report_both_raw_and_clipped():
    rows = read_csv("regression_metrics.csv")
    outputs = {row["output"] for row in rows}
    assert {"raw", "clipped"} <= outputs


def test_baseline_relative_improvement_is_computable():
    rows = read_csv("regression_metrics.csv")
    baseline = next(r for r in rows if r["model"] == "B0_global_mean")
    best = min(
        (r for r in rows if r["model"] == "B2_frozen" and r["output"] == "raw"),
        key=lambda r: float(r["mae"]),
    )
    assert float(best["mae"]) < float(baseline["mae"])


def test_failure_cases_carry_no_image_data():
    rows = read_csv("failure_cases.csv")
    assert rows
    for row in rows:
        assert set(row) == {
            "sampleId",
            "foodName",
            "group",
            "target",
            "prediction",
            "absError",
        }


# --- 제품 경계 --------------------------------------------------------------


def test_fixture_was_not_replaced():
    """외부 벤치마크 결과만으로 운영 분석기를 바꾸지 않는다."""
    from babplanner_ai.waste.fixture_estimator import (
        FIXTURE_MODEL_VERSION,
        FixtureWasteEstimator,
    )

    estimate = FixtureWasteEstimator().estimate(b"x", menu_id=1)
    assert estimate.model_version == FIXTURE_MODEL_VERSION
    assert "fixture" in FIXTURE_MODEL_VERSION


def test_no_trained_estimator_is_wired_into_the_service():
    """W2에서 FastAPI 경로에 학습된 모델을 붙이지 않았다."""
    api = (ROOT / "src" / "babplanner_ai" / "waste" / "api.py").read_text(
        encoding="utf-8"
    )
    assert "waste_estimation" not in api
    assert "torch" not in api


def test_report_states_this_is_not_cafeteria_performance():
    text = " ".join(REPORT.read_text(encoding="utf-8").split())
    assert "밥플래너 급식소" in text
    assert "급식소 성능이 아니다" in text


def test_report_records_the_hypothesis_verdict():
    text = " ".join(REPORT.read_text(encoding="utf-8").split())
    assert "H-W2" in text
    assert "지지되지 않는다" in text


def test_report_marks_human_baseline_as_not_measured():
    text = " ".join(REPORT.read_text(encoding="utf-8").split())
    assert "HUMAN BASELINE: NOT MEASURED" in text


def test_dataset_metadata_marks_the_source_as_external():
    metadata = read_json("dataset_metadata.json")
    assert "급식소 성능이라고 말하지 않는다" in metadata["note"]
    assert metadata["license"] == "CC BY 4.0"


def test_no_raw_dataset_or_weights_are_tracked():
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    for path in tracked:
        lowered = path.lower()
        assert "waste_estimation/data/" not in lowered, path
        assert not lowered.endswith((".jpg", ".jpeg", ".xlsx", ".npz", ".pth", ".pt"))
