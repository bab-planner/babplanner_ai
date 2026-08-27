"""정답 데이터 스키마."""

from __future__ import annotations

from pathlib import Path

import pytest

from cctv_counting.groundtruth import (
    CROSSING_GT_FIELDS,
    OCCUPANCY_GT_FIELDS,
    Direction,
    GroundTruthError,
    entries_of,
    read_crossing_ground_truth,
    read_occupancy_ground_truth,
    write_template,
)

ANNOTATIONS = Path(__file__).resolve().parents[1] / "datasets" / "annotations"


def test_occupancy_and_crossing_are_separate_files():
    """occupancy와 entry는 다른 값이므로 정답도 하나로 합치지 않는다."""
    assert OCCUPANCY_GT_FIELDS != CROSSING_GT_FIELDS
    assert "actual_count" in OCCUPANCY_GT_FIELDS
    assert "direction" in CROSSING_GT_FIELDS


def test_ground_truth_schemas_carry_no_identity():
    for fields in (OCCUPANCY_GT_FIELDS, CROSSING_GT_FIELDS):
        for field in fields:
            assert "person" not in field
            assert "student" not in field
            assert "track" not in field


def test_shipped_templates_are_header_only():
    """정답이 없는데 행이 있으면 그것은 지어낸 정답이다."""
    for name, fields in (
        ("occupancy_ground_truth.csv", OCCUPANCY_GT_FIELDS),
        ("crossing_ground_truth.csv", CROSSING_GT_FIELDS),
    ):
        lines = (ANNOTATIONS / name).read_text(encoding="utf-8").strip().splitlines()
        assert lines == [",".join(fields)]


def test_shipped_templates_parse_as_empty():
    assert read_occupancy_ground_truth(ANNOTATIONS / "occupancy_ground_truth.csv") == []
    assert read_crossing_ground_truth(ANNOTATIONS / "crossing_ground_truth.csv") == []


def test_write_template_then_read(tmp_path):
    path = tmp_path / "occupancy.csv"
    write_template(path, OCCUPANCY_GT_FIELDS)
    assert read_occupancy_ground_truth(path) == []


def test_reads_occupancy_rows(tmp_path):
    path = tmp_path / "occupancy.csv"
    path.write_text(
        ",".join(OCCUPANCY_GT_FIELDS)
        + "\ndemo-01,120,4.8,17,daylight,me,\n",
        encoding="utf-8",
    )
    rows = read_occupancy_ground_truth(path)
    assert rows[0].actual_count == 17
    assert rows[0].condition == "daylight"
    assert rows[0].notes is None


def test_blank_condition_stays_none(tmp_path):
    """비운 것은 비운 채로 남는다. 조건을 지어내지 않는다."""
    path = tmp_path / "occupancy.csv"
    path.write_text(
        ",".join(OCCUPANCY_GT_FIELDS) + "\ndemo-01,0,0.0,3,,,\n", encoding="utf-8"
    )
    assert read_occupancy_ground_truth(path)[0].condition is None


def test_negative_count_is_rejected(tmp_path):
    path = tmp_path / "occupancy.csv"
    path.write_text(
        ",".join(OCCUPANCY_GT_FIELDS) + "\ndemo-01,0,0.0,-2,,,\n", encoding="utf-8"
    )
    with pytest.raises(GroundTruthError):
        read_occupancy_ground_truth(path)


def test_wrong_header_is_rejected(tmp_path):
    path = tmp_path / "occupancy.csv"
    path.write_text("video_id,count\ndemo-01,3\n", encoding="utf-8")
    with pytest.raises(GroundTruthError):
        read_occupancy_ground_truth(path)


def test_reads_crossing_rows_and_counts_entries(tmp_path):
    path = tmp_path / "crossing.csv"
    header = ",".join(CROSSING_GT_FIELDS)
    path.write_text(
        header
        + "\ndemo-01,0,10,0.4,IN,,,"
        + "\ndemo-01,1,40,1.6,IN,,,"
        + "\ndemo-01,2,90,3.6,OUT,,,"
        + "\ndemo-02,0,10,0.4,IN,,,\n",
        encoding="utf-8",
    )
    rows = read_crossing_ground_truth(path)
    assert len(rows) == 4
    assert rows[2].direction is Direction.OUT
    assert entries_of(rows, "demo-01") == 2
    assert entries_of(rows, "demo-02") == 1
    assert entries_of(rows, "demo-99") == 0


def test_unknown_direction_is_rejected(tmp_path):
    path = tmp_path / "crossing.csv"
    path.write_text(
        ",".join(CROSSING_GT_FIELDS) + "\ndemo-01,0,10,0.4,SIDEWAYS,,,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        read_crossing_ground_truth(path)
