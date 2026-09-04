"""매니페스트와 파일 점검."""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
from pathlib import Path

import pytest
from PIL import Image

from babplanner_ai.dataset.manifest import ManifestError, read_manifest
from babplanner_ai.dataset.validation import validate_dataset

FIXTURE = Path(__file__).parent / "fixtures" / "dataset"


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """고쳐 가며 볼 수 있게 fixture를 복사한다."""
    target = tmp_path / "dataset"
    shutil.copytree(FIXTURE, target)
    return target


def rows_of(root: Path) -> list[dict[str, str]]:
    with (root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(root: Path, rows: list[dict[str, str]], name: str = "manifest.csv") -> Path:
    path = root / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_fixture_dataset_is_clean(dataset: Path) -> None:
    report = validate_dataset(read_manifest(dataset / "manifest.csv"), dataset)

    assert report.ok
    assert not report.warnings


def test_missing_image_is_an_error(dataset: Path) -> None:
    rows = rows_of(dataset)
    rows[0]["image_path"] = "images/nope.png"
    path = write_rows(dataset, rows, "broken.csv")

    report = validate_dataset(read_manifest(path), dataset)

    assert not report.ok
    assert any("이미지가 없습니다" in finding.message for finding in report.errors)


def test_undecodable_image_is_an_error(dataset: Path) -> None:
    rows = rows_of(dataset)
    target = dataset / rows[0]["image_path"]
    target.write_bytes(b"not really an image")
    rows[0]["image_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    path = write_rows(dataset, rows, "broken.csv")

    report = validate_dataset(read_manifest(path), dataset)

    assert any("열 수 없습니다" in finding.message for finding in report.errors)


def test_png_and_jpeg_both_decode(dataset: Path) -> None:
    rows = rows_of(dataset)
    jpeg = dataset / "images" / f"{rows[0]['sample_id']}.jpg"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(jpeg, format="JPEG")
    rows[0]["image_path"] = f"images/{jpeg.name}"
    rows[0]["image_sha256"] = hashlib.sha256(jpeg.read_bytes()).hexdigest()
    path = write_rows(dataset, rows, "jpeg.csv")

    report = validate_dataset(read_manifest(path), dataset)

    assert report.ok


def test_recorded_hash_must_match_the_file(dataset: Path) -> None:
    rows = rows_of(dataset)
    rows[0]["image_sha256"] = "0" * 64
    path = write_rows(dataset, rows, "hash.csv")

    report = validate_dataset(read_manifest(path), dataset)

    assert any("해시가 실제 파일과 다릅니다" in finding.message for finding in report.errors)


def test_duplicate_image_is_an_error(dataset: Path) -> None:
    rows = rows_of(dataset)
    # 두 번째 표본이 첫 번째와 같은 파일을 가리키게 한다.
    rows[1]["image_path"] = rows[0]["image_path"]
    rows[1]["image_sha256"] = rows[0]["image_sha256"]
    path = write_rows(dataset, rows, "dup.csv")

    report = validate_dataset(read_manifest(path), dataset)

    assert any("같은 이미지가 여러 표본" in finding.message for finding in report.errors)


def test_duplicate_sample_id_is_an_error(dataset: Path) -> None:
    rows = rows_of(dataset)
    rows[1]["sample_id"] = rows[0]["sample_id"]
    path = write_rows(dataset, rows, "dupid.csv")

    report = validate_dataset(read_manifest(path), dataset)

    assert any("같은 sample_id" in finding.message for finding in report.errors)


def test_mixed_label_versions_are_an_error(dataset: Path) -> None:
    rows = rows_of(dataset)
    rows[0]["label_version"] = "waste-mass-ratio-v2"
    path = write_rows(dataset, rows, "mixed.csv")

    report = validate_dataset(read_manifest(path), dataset)

    assert any("정답 정의 버전이 섞여" in finding.message for finding in report.errors)


def test_filename_other_than_sample_id_warns(dataset: Path) -> None:
    rows = rows_of(dataset)
    renamed = dataset / "images" / "IMG_20260115_121314.png"
    (dataset / rows[0]["image_path"]).rename(renamed)
    rows[0]["image_path"] = f"images/{renamed.name}"
    path = write_rows(dataset, rows, "named.csv")

    report = validate_dataset(read_manifest(path), dataset)

    assert report.ok
    assert any("파일 이름이 sample_id와 다릅니다" in f.message for f in report.warnings)


@pytest.mark.parametrize(
    "column",
    ["name", "phone", "student_id", "email", "access_token", "refresh_token",
     "verification_token"],
)
def test_personal_columns_are_refused(dataset: Path, column: str) -> None:
    rows = rows_of(dataset)
    for row in rows:
        row[column] = "x"
    path = write_rows(dataset, rows, "pii.csv")

    with pytest.raises(ManifestError, match="개인정보"):
        read_manifest(path)


def test_personal_column_check_ignores_case(dataset: Path) -> None:
    rows = rows_of(dataset)
    for row in rows:
        row["Student_ID"] = "x"
    path = write_rows(dataset, rows, "pii-case.csv")

    with pytest.raises(ManifestError, match="개인정보"):
        read_manifest(path)


def test_confidence_column_is_refused(dataset: Path) -> None:
    # confidence를 사람이 붙여 학습 target으로 삼지 않는다.
    rows = rows_of(dataset)
    for row in rows:
        row["confidence"] = "0.9"
    path = write_rows(dataset, rows, "conf.csv")

    with pytest.raises(ManifestError, match="정답으로 쓸 수 없는"):
        read_manifest(path)


def test_fixture_manifest_carries_no_personal_data() -> None:
    # 합성 데이터라도 개인정보처럼 보이는 값이 섞이지 않았는지 본다.
    # 값 단위로 본다. 통째로 훑으면 16진수 식별자 안의 숫자에 걸린다.
    phone = re.compile(r"^01[016789]\d{7,8}$")
    with (FIXTURE / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        for value in row.values():
            assert "@" not in value
            assert not phone.match(value)
            assert not re.search(r"[가-힣]", value)


def test_empty_manifest_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ManifestError):
        read_manifest(path)


def test_row_error_points_at_the_line(dataset: Path) -> None:
    rows = rows_of(dataset)
    rows[0]["leftover_edible_mass_g"] = "9000"
    path = write_rows(dataset, rows, "bad.csv")

    with pytest.raises(ManifestError, match="2행"):
        read_manifest(path)


def test_string_mass_is_refused(dataset: Path) -> None:
    # "100g" 같은 문자열이 아니라 숫자로 적는다.
    rows = rows_of(dataset)
    rows[0]["served_edible_mass_g"] = "500g"
    path = write_rows(dataset, rows, "unit.csv")

    with pytest.raises(ManifestError):
        read_manifest(path)


def test_split_leakage_is_an_error(dataset: Path) -> None:
    rows = rows_of(dataset)
    # 같은 참가자를 학습과 평가에 나눠 넣는다.
    rows[0]["split"] = "train"
    rows[1]["split"] = "test"
    path = write_rows(dataset, rows, "leak.csv")

    report = validate_dataset(read_manifest(path), dataset)

    assert any("여러 분할에 있습니다" in finding.message for finding in report.errors)


def test_visual_estimate_warns_about_the_benchmark(dataset: Path) -> None:
    rows = rows_of(dataset)
    rows[0]["served_mass_source"] = "VISUAL_ESTIMATE"
    path = write_rows(dataset, rows, "visual.csv")

    report = validate_dataset(read_manifest(path), dataset)

    assert report.ok
    assert any("주 평가 집합에서 제외" in finding.message for finding in report.warnings)


# --- 정답 제외 --------------------------------------------------------------


def mark_excluded(rows: list[dict[str, str]], count: int, reason: str = "BONE_IN") -> None:
    """앞의 몇 행에 정답 제외 표시를 붙인다. 옛 매니페스트에는 열이 없다."""
    for row in rows:
        row.setdefault("gt_exclusion_reason", "")
        row.setdefault("gt_exclusion_note", "")
    for row in rows[:count]:
        row["gt_exclusion_reason"] = reason


def test_manifest_without_the_columns_still_reads(dataset: Path) -> None:
    # 옛 매니페스트에는 이 열이 없다. 그대로 읽혀야 한다.
    samples = read_manifest(dataset / "manifest.csv")

    assert all(sample.gt_exclusion_reason is None for sample in samples)
    assert validate_dataset(samples, dataset).ok


def test_exclusions_are_reported_with_counts(dataset: Path) -> None:
    rows = rows_of(dataset)
    mark_excluded(rows, 2)
    path = write_rows(dataset, rows, "excluded.csv")

    report = validate_dataset(read_manifest(path), dataset)

    # 총계 뒤에 숨지 않게 사유별·메뉴별로 센다.
    assert report.ok
    assert any(
        "정답 제외 표본 2건" in finding.message and "BONE_IN 2건" in finding.message
        for finding in report.warnings
    )
    assert any("메뉴별 정답 제외" in finding.message for finding in report.warnings)


def test_excluded_sample_with_a_split_is_an_error(dataset: Path) -> None:
    rows = rows_of(dataset)
    mark_excluded(rows, 1)
    rows[0]["split"] = "train"
    path = write_rows(dataset, rows, "excluded_split.csv")

    report = validate_dataset(read_manifest(path), dataset)

    # 정답으로 쓰지 않기로 한 표본이 학습에 섞여 들어간 상태다.
    assert not report.ok
    assert any("분할 train" in finding.message for finding in report.errors)


def test_excluded_samples_are_not_called_unsplit(dataset: Path) -> None:
    rows = rows_of(dataset)
    mark_excluded(rows, 1)
    path = write_rows(dataset, rows, "excluded_only.csv")

    report = validate_dataset(read_manifest(path), dataset)

    # 일부러 뺀 것을 "아직 안 나눴다"고 말하지 않는다.
    assert not any("분할이 없는 표본" in finding.message for finding in report.warnings)
