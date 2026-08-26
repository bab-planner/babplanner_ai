"""이미지 받아 두기와 비식별 사본."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from babplanner_ai.collection.image_processing import (
    ImageIngestError,
    has_exif,
    ingest,
    sha256_of,
)

SAMPLE_ID = "0123456789abcdef0123456789abcdef"


def png(path: Path, size: tuple[int, int] = (24, 16)) -> Path:
    Image.new("RGB", size, (200, 170, 120)).save(path, format="PNG")
    return path


def jpeg_with_metadata(path: Path, *, orientation: int | None = None) -> Path:
    """기기 정보와 방향 태그가 든 사진.

    실제 휴대폰 사진이 이런 모양이다.
    """
    image = Image.new("RGB", (40, 20), (30, 60, 90))
    exif = image.getexif()
    exif[271] = "TestPhone"
    exif[272] = "Model-X"
    exif[306] = "2026:08:27 13:00:00"
    if orientation is not None:
        exif[274] = orientation
    image.save(path, format="JPEG", exif=exif.tobytes())
    return path


def run(source: Path, root: Path, sample_id: str = SAMPLE_ID):
    return ingest(
        source,
        root=root,
        raw_dir=root / "raw_images",
        sanitized_dir=root / "sanitized_images",
        sample_id=sample_id,
    )


def test_png_is_ingested(tmp_path: Path) -> None:
    result = run(png(tmp_path / "src.png"), tmp_path / "store")

    assert result.sanitized_path.is_file()
    assert result.image_format == "PNG"


def test_jpeg_is_ingested(tmp_path: Path) -> None:
    result = run(jpeg_with_metadata(tmp_path / "src.jpg"), tmp_path / "store")

    assert result.sanitized_path.is_file()
    assert result.image_format == "JPEG"


def test_raw_original_is_kept_untouched(tmp_path: Path) -> None:
    source = jpeg_with_metadata(tmp_path / "src.jpg")
    before = sha256_of(source)

    result = run(source, tmp_path / "store")

    # 원본은 그대로 두고 사본을 만든다.
    assert sha256_of(source) == before
    assert result.raw_sha256 == before
    assert has_exif(result.raw_path)


def test_exif_is_removed_from_the_copy(tmp_path: Path) -> None:
    result = run(jpeg_with_metadata(tmp_path / "src.jpg"), tmp_path / "store")

    assert not has_exif(result.sanitized_path)


def test_device_metadata_is_gone(tmp_path: Path) -> None:
    result = run(jpeg_with_metadata(tmp_path / "src.jpg"), tmp_path / "store")

    assert b"TestPhone" not in result.sanitized_path.read_bytes()
    assert b"Model-X" not in result.sanitized_path.read_bytes()


def test_location_metadata_is_gone(tmp_path: Path) -> None:
    image = Image.new("RGB", (20, 20), (10, 10, 10))
    exif = image.getexif()
    exif[271] = "TestPhone"
    gps = exif.get_ifd(0x8825)
    gps[1] = "N"
    gps[3] = "E"
    source = tmp_path / "gps.jpg"
    image.save(source, format="JPEG", exif=exif.tobytes())

    result = run(source, tmp_path / "store")

    with Image.open(result.sanitized_path) as clean:
        assert not clean.getexif().get_ifd(0x8825)


def test_orientation_is_baked_into_the_pixels(tmp_path: Path) -> None:
    # 방향 태그만 지우면 휴대폰 사진이 90도 누운 채로 남는다.
    source = jpeg_with_metadata(tmp_path / "rot.jpg", orientation=6)

    result = run(source, tmp_path / "store")

    with Image.open(source) as original, Image.open(result.sanitized_path) as clean:
        assert original.size == (40, 20)
        assert clean.size == (20, 40)


def test_upright_photo_keeps_its_size(tmp_path: Path) -> None:
    source = jpeg_with_metadata(tmp_path / "up.jpg", orientation=1)

    result = run(source, tmp_path / "store")

    with Image.open(result.sanitized_path) as clean:
        assert clean.size == (40, 20)


def test_sanitized_hash_differs_from_raw(tmp_path: Path) -> None:
    # 매니페스트가 가리키는 것은 학습에 쓰는 사본이다. 두 해시를 섞으면
    # 검증기가 엉뚱한 파일을 대조하게 된다.
    result = run(jpeg_with_metadata(tmp_path / "src.jpg"), tmp_path / "store")

    assert result.sanitized_sha256 == sha256_of(result.sanitized_path)
    assert result.sanitized_sha256 != result.raw_sha256


def test_destination_is_named_after_the_sample(tmp_path: Path) -> None:
    result = run(png(tmp_path / "IMG_20260827_홍길동.png"), tmp_path / "store")

    # 촬영 기기가 붙인 이름을 그대로 두면 그 안에 무엇이 들어 있는지
    # 통제할 수 없다.
    assert result.sanitized_path.name == f"{SAMPLE_ID}.png"
    assert result.raw_path.name == f"{SAMPLE_ID}.png"
    assert "홍길동" not in str(result.sanitized_path)


def test_extension_comes_from_the_content_not_the_name(tmp_path: Path) -> None:
    # 이름이 .png여도 내용이 JPEG면 .jpg로 저장한다.
    source = jpeg_with_metadata(tmp_path / "lying.png")

    result = run(source, tmp_path / "store")

    assert result.sanitized_path.suffix == ".jpg"


def test_traversal_in_the_sample_id_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ImageIngestError, match="루트 밖"):
        run(png(tmp_path / "src.png"), tmp_path / "store", sample_id="../../escaped")


def test_undecodable_file_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "fake.png"
    source.write_bytes(b"not really an image")

    with pytest.raises(ImageIngestError, match="해석할 수 없"):
        run(source, tmp_path / "store")


def test_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ImageIngestError, match="파일이 없습니다"):
        run(tmp_path / "nope.png", tmp_path / "store")


def test_identical_images_get_identical_hashes(tmp_path: Path) -> None:
    # 같은 사진이 두 표본으로 들어오면 이 해시로 잡는다.
    first = run(png(tmp_path / "a.png"), tmp_path / "one", sample_id=SAMPLE_ID)
    second = run(png(tmp_path / "b.png"), tmp_path / "two", sample_id=SAMPLE_ID)

    assert first.sanitized_sha256 == second.sanitized_sha256
