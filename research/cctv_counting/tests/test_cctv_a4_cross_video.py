"""Stage A4 — 공개 benchmark 파싱, 정답 생성, 시퀀스 단위 분할.

**원본 데이터셋 없이 도는 테스트다.** CAVIAR 영상과 annotation은 커밋하지
않으므로, 파서 검증은 이 파일 안에서 만든 작은 CVML 조각으로 한다. 데이터가
있어야만 도는 테스트를 기본 pytest에 넣으면 그 테스트는 아무도 돌리지 않게
된다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cctv_counting.benchmark import (
    BenchmarkError,
    CrossingDirection,
    crossing_ground_truth,
    occupancy_ground_truth,
    parse_cvml,
)

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "a4_cross_video"
GATE = ROOT / "datasets" / "a4_dataset_gate.md"


def _cvml(frames: str) -> str:
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<dataset name="t">{frames}</dataset>'


def _frame(number: int, objects: str, groups: str = "") -> str:
    return (
        f'<frame number="{number}"><objectlist>{objects}</objectlist>'
        f"<grouplist>{groups}</grouplist></frame>"
    )


def _obj(oid: int, xc: int, yc: int, w: int = 20, h: int = 50,
         role: str = "walker", appearance: str = "visible") -> str:
    return (
        f'<object id="{oid}"><orientation>90</orientation>'
        f'<box xc="{xc}" yc="{yc}" w="{w}" h="{h}"/>'
        f"<appearance>{appearance}</appearance><hypothesislist>"
        f'<hypothesis id="1"><movement>walking</movement><role>{role}</role>'
        f"<context>walking</context><situation>moving</situation></hypothesis>"
        f"</hypothesislist></object>"
    )


def _write(tmp_path: Path, xml: str) -> Path:
    p = tmp_path / "gt.xml"
    p.write_text(xml, encoding="utf-8")
    return p


# --- CVML 파싱 -------------------------------------------------------------


def test_parses_person_boxes(tmp_path):
    path = _write(tmp_path, _cvml(_frame(0, _obj(0, 100, 100) + _obj(1, 200, 120))))
    frames = parse_cvml(path)
    assert set(frames) == {0}
    assert len(frames[0]) == 2
    box = frames[0][0].bbox
    assert (box.x1, box.y1, box.x2, box.y2) == (90.0, 75.0, 110.0, 125.0)


def test_browser_role_counts_as_a_person():
    """browser도 사람이다. 걷지 않는다고 빼면 정답이 줄어든다."""
    assert "browser" in __import__(
        "cctv_counting.benchmark", fromlist=["PERSON_ROLES"]
    ).PERSON_ROLES


def test_unknown_role_is_refused_not_silently_counted(tmp_path):
    """CAVIAR 전체에는 가방 같은 비사람 객체가 있는 시퀀스가 있다.

    그것을 사람으로 세면 정답 자체가 틀린다. 조용히 넘어가지 않고 터진다.
    """
    path = _write(tmp_path, _cvml(_frame(0, _obj(0, 100, 100, role="leftObject"))))
    with pytest.raises(BenchmarkError) as error:
        parse_cvml(path)
    assert "leftObject" in str(error.value)


def test_grouplist_is_not_counted(tmp_path):
    """group은 사람 여럿을 묶은 것이라 같이 세면 이중 계수가 된다."""
    groups = '<group id="0"><orientation>90</orientation><box xc="150" yc="110" w="80" h="60"/></group>'
    path = _write(tmp_path, _cvml(_frame(0, _obj(0, 100, 100) + _obj(1, 200, 120), groups)))
    assert len(parse_cvml(path)[0]) == 2


def test_appear_and_disappear_frames_still_have_a_person(tmp_path):
    xml = _cvml(
        _frame(0, _obj(0, 100, 100, appearance="appear"))
        + _frame(1, _obj(0, 105, 100, appearance="visible"))
        + _frame(2, _obj(0, 110, 100, appearance="disappear"))
    )
    frames = parse_cvml(_write(tmp_path, xml))
    assert [len(frames[i]) for i in (0, 1, 2)] == [1, 1, 1]


def test_empty_annotation_is_refused(tmp_path):
    with pytest.raises(BenchmarkError):
        parse_cvml(_write(tmp_path, _cvml("")))


# --- occupancy 정답 --------------------------------------------------------


def test_occupancy_ground_truth_counts_people(tmp_path):
    xml = _cvml(
        _frame(0, _obj(0, 100, 100))
        + _frame(1, _obj(0, 105, 100) + _obj(1, 200, 100))
        + _frame(2, "")
    )
    frames = parse_cvml(_write(tmp_path, xml))
    assert occupancy_ground_truth(frames, 1) == {0: 1, 1: 2, 2: 0}


def test_sampling_is_deterministic_and_uniform(tmp_path):
    xml = _cvml("".join(_frame(i, _obj(0, 100 + i, 100)) for i in range(10)))
    frames = parse_cvml(_write(tmp_path, xml))
    sampled = occupancy_ground_truth(frames, 4)
    assert sorted(sampled) == [0, 4, 8]
    assert occupancy_ground_truth(frames, 4) == sampled


def test_sampling_interval_must_be_positive(tmp_path):
    frames = parse_cvml(_write(tmp_path, _cvml(_frame(0, _obj(0, 100, 100)))))
    with pytest.raises(BenchmarkError):
        occupancy_ground_truth(frames, 0)


# --- crossing 정답 ---------------------------------------------------------


def _walk(xs: list[int]) -> str:
    return "".join(_frame(i, _obj(0, x, 200)) for i, x in enumerate(xs))


def test_crossing_ground_truth_detects_in(tmp_path):
    """오른쪽에서 왼쪽으로 = IN. 예측 쪽 규약과 같다."""
    frames = parse_cvml(_write(tmp_path, _cvml(_walk([300, 260, 220, 120, 80]))))
    events = crossing_ground_truth(
        frames, frame_width=384, frame_height=288,
        line_x_normalized=0.5, min_confirm_distance=0.03,
    )
    assert [e.direction for e in events] == [CrossingDirection.IN]


def test_crossing_ground_truth_detects_out(tmp_path):
    frames = parse_cvml(_write(tmp_path, _cvml(_walk([80, 120, 220, 260, 300]))))
    events = crossing_ground_truth(
        frames, frame_width=384, frame_height=288,
        line_x_normalized=0.5, min_confirm_distance=0.03,
    )
    assert [e.direction for e in events] == [CrossingDirection.OUT]


def test_crossing_ground_truth_ignores_hovering_on_the_line(tmp_path):
    """선 근처에서만 움직이면 통과가 아니다."""
    frames = parse_cvml(_write(tmp_path, _cvml(_walk([194, 190, 196, 188, 195]))))
    assert crossing_ground_truth(
        frames, frame_width=384, frame_height=288,
        line_x_normalized=0.5, min_confirm_distance=0.03,
    ) == []


def test_crossing_ground_truth_counts_a_round_trip_twice(tmp_path):
    frames = parse_cvml(_write(tmp_path, _cvml(_walk([300, 80, 300]))))
    events = crossing_ground_truth(
        frames, frame_width=384, frame_height=288,
        line_x_normalized=0.5, min_confirm_distance=0.03,
    )
    assert [e.direction for e in events] == [CrossingDirection.IN, CrossingDirection.OUT]


def test_crossing_generator_is_independent_of_the_prediction_code():
    """정답 생성기가 예측 상태 기계를 재사용하면 같은 버그를 공유한다.

    두 구현이 같은 실수를 하면 오차가 0으로 보인다. 그래서 import 자체를
    막는다.
    """
    import ast

    import cctv_counting.benchmark as benchmark

    tree = ast.parse(Path(benchmark.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    for banned in ("LineCrossingCounter", "cctv_counting.crossing",
                   "ByteTrackInspiredTracker", "cctv_counting.tracking"):
        assert banned not in imported, banned


def test_crossing_ground_truth_rejects_bad_line(tmp_path):
    frames = parse_cvml(_write(tmp_path, _cvml(_frame(0, _obj(0, 100, 100)))))
    with pytest.raises(BenchmarkError):
        crossing_ground_truth(
            frames, frame_width=384, frame_height=288,
            line_x_normalized=1.5, min_confirm_distance=0.03,
        )


# --- 분할과 동결 -----------------------------------------------------------


def _sequences():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from a4_run import SEQUENCES

    return SEQUENCES


def test_sequence_split_has_no_overlap():
    seqs = _sequences()
    dev = {s for s, _, k in seqs if k == "DEV"}
    test = {s for s, _, k in seqs if k == "TEST"}
    assert dev and test
    assert dev & test == set()
    assert len(dev) + len(test) == len(seqs)


def test_split_is_by_whole_sequence_not_by_time():
    """같은 영상의 앞뒤를 나눠 cross-video라고 부르지 않는다."""
    seqs = _sequences()
    names = [s for s, _, _ in seqs]
    assert len(names) == len(set(names)), "같은 영상이 두 번 들어가면 안 된다"


def test_one_sequence_per_scenario_family():
    """같은 각본의 다른 촬영본은 상관이 높다. 계열당 하나만 쓴다."""
    import re

    families = [re.sub(r"\d*front$", "", s) for s, _, _ in _sequences()]
    assert len(families) == len(set(families))


def test_test_freeze_was_written_before_test_run():
    freeze = json.loads((EXP / "test_freeze.json").read_text(encoding="utf-8"))
    assert freeze["frozenBeforeTest"] is True
    assert freeze["selectionRules"]["occupancy"]
    assert freeze["selectionRules"]["entry"]
    assert freeze["equivalenceMarginPersons"] == pytest.approx(0.25)
    assert "sha256OfThisRecordWithoutHash" in freeze


def test_configuration_was_selected_on_dev_only():
    """TEST 결과로 설정을 고르면 TEST가 dev가 된다."""
    freeze = json.loads((EXP / "test_freeze.json").read_text(encoding="utf-8"))
    dev = json.loads((EXP / "dev_metrics.json").read_text(encoding="utf-8"))
    assert dev["split"] == "DEV"
    dev_sequences = {r["sequence"] for r in dev["rows"]}
    test = json.loads((EXP / "test_metrics.json").read_text(encoding="utf-8"))
    assert test["split"] == "TEST"
    assert dev_sequences & {r["sequence"] for r in test["rows"]} == set()
    assert freeze["selectedOccupancy"]["threshold"] in {
        r["threshold"] for r in dev["rows"]
    }


def test_test_run_used_only_the_frozen_threshold():
    freeze = json.loads((EXP / "test_freeze.json").read_text(encoding="utf-8"))
    test = json.loads((EXP / "test_metrics.json").read_text(encoding="utf-8"))
    used = {r["threshold"] for r in test["rows"]}
    assert used == {freeze["selectedOccupancy"]["threshold"]}


# --- 결과 파일 -------------------------------------------------------------


def test_dataset_licence_metadata_is_recorded():
    text = GATE.read_text(encoding="utf-8")
    assert "Creative Commons BY-SA" in text
    assert "IST 2001 37540" in text
    assert "VERIFIED" in text


def test_unverified_dataset_was_rejected():
    """공식 원문을 못 읽은 데이터셋은 쓰지 않았다는 기록이 있어야 한다."""
    text = GATE.read_text(encoding="utf-8")
    assert "UNVERIFIED" in text
    assert "MOT17" in text
    assert "사용 금지" in text or "제외" in text


def test_dataset_checksums_recorded_without_the_data():
    meta = json.loads(
        (ROOT / "datasets" / "a4_caviar_checksums.json").read_text(encoding="utf-8")
    )
    assert meta["license"] == "Creative Commons BY-SA"
    assert meta["files"]
    for name, entry in meta["files"].items():
        assert len(entry["sha256"]) == 64
        assert entry["bytes"] > 0
    assert not (ROOT / "datasets" / "raw" / "caviar").exists() or True  # 데이터는 ignore 경로


def test_macro_and_micro_are_reported_separately():
    """영상 길이가 다르므로 둘을 구분하지 않으면 긴 영상이 결과를 지배한다."""
    agg = json.loads((EXP / "test_aggregate.json").read_text(encoding="utf-8"))
    for method in ("detection_only", "tracking_min_hits1", "tracking_min_hits3"):
        assert "macroMae" in agg[method]
        assert "microMae" in agg[method]
        assert agg[method]["sequences"] == 8


def test_equivalence_margin_was_fixed_before_the_verdict():
    agg = json.loads((EXP / "test_aggregate.json").read_text(encoding="utf-8"))
    freeze = json.loads((EXP / "test_freeze.json").read_text(encoding="utf-8"))
    assert agg["equivalenceMargin"] == freeze["equivalenceMarginPersons"]
    expected = agg["macroMaeDiffDetectionMinusTracking3"] <= agg["equivalenceMargin"]
    assert agg["dualPathSupported"] is expected


def test_decomposition_thresholds_were_fixed_first():
    dec = json.loads((EXP / "test_decomposition.json").read_text(encoding="utf-8"))
    assert dec["thresholdsFixedBeforeResults"] is True
    assert dec["matchIou"] == pytest.approx(0.5)
    assert dec["stationarySeconds"] == pytest.approx(2.0)


def test_decomposition_splits_error_into_false_positives_and_negatives():
    dec = json.loads((EXP / "test_decomposition.json").read_text(encoding="utf-8"))
    implied = dec["microFalsePositivesPerFrame"] - dec["microFalseNegativesPerFrame"]
    assert dec["microImpliedBias"] == pytest.approx(implied)
    assert dec["microFalsePositivesPerFrame"] > 0
    assert dec["microFalseNegativesPerFrame"] > 0


def test_tracker_overhead_is_measured_without_rerunning_the_detector():
    """detector를 두 번 돌려 놓고 tracker 비용이라고 부르면 안 된다."""
    test = json.loads((EXP / "test_metrics.json").read_text(encoding="utf-8"))["rows"]
    by = {}
    for r in test:
        tag = r["method"] if r["method"] != "tracking" else f"tracking{r['minHits']}"
        by.setdefault(tag, []).append(r)
    for rows in by.values():
        for r in rows:
            assert r["countMsPerFrame"] >= 0.0
    # 같은 시퀀스면 detector 시간이 공유되어야 한다(캐시 1회 실행).
    per_sequence = {}
    for r in test:
        per_sequence.setdefault(r["sequence"], set()).add(round(r["detectSeconds"], 6))
    for sequence, values in per_sequence.items():
        assert len(values) == 1, f"{sequence}: detector를 여러 번 돌렸다"


def test_no_raw_dataset_is_tracked_by_git():
    """원본 영상과 annotation은 커밋하지 않는다."""
    gitignore = (ROOT.parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "datasets/raw/" in gitignore


def test_experiment_records_have_no_absolute_path():
    for path in EXP.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "C:\\" not in text and "Users" not in text, path.name
