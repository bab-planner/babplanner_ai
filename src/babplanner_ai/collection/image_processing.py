"""이미지 받아 두기와 비식별 사본 만들기.

원본은 손대지 않고 그대로 둔다. 학습에 쓰는 것은 EXIF를 벗겨 다시 저장한
사본이다.

**EXIF를 지운다고 개인정보가 사라지지는 않는다.** 화면 안에 얼굴이나
학생증이 찍혀 있으면 그것은 픽셀이라 여기서 할 수 있는 일이 없다. 1차
방어는 촬영 구도이고 이것은 2차 방어다.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

#: 사본을 저장할 때 쓰는 확장자. 파일명에서 받지 않고 실제로 열어 본
#: 형식에서 정한다. 사용자가 준 이름을 목적지 경로에 쓰면 그 이름으로
#: 디렉터리를 빠져나갈 수 있다.
EXTENSION_OF = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
    "TIFF": ".tif",
    "GIF": ".gif",
}

#: JPEG를 다시 저장할 때의 화질. EXIF를 벗기려면 다시 인코딩해야 하는데,
#: 그때 눈에 띄는 손실이 생기지 않을 만큼 높게 잡는다.
JPEG_QUALITY = 95

#: 색차 성분을 줄이지 않는다(4:4:4). 잔반은 색으로 구분되는 부분이 있어
#: 기본 4:2:0으로 다시 인코딩하면 그 정보가 깎인다.
JPEG_SUBSAMPLING = 0


class ImageIngestError(ValueError):
    """이미지를 받아들일 수 없다."""


@dataclass(frozen=True)
class IngestedImage:
    raw_path: Path
    sanitized_path: Path
    raw_sha256: str
    sanitized_sha256: str
    image_format: str
    width: int
    height: int


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detect_format(source: Path) -> str:
    try:
        with Image.open(source) as image:
            image.verify()
        with Image.open(source) as image:
            return (image.format or "").upper()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImageIngestError(f"이미지를 해석할 수 없습니다: {source.name}") from error


def _inside(root: Path, candidate: Path) -> Path:
    """목적지가 수집 루트 안인지 확인한다."""
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ImageIngestError("수집 루트 밖으로는 쓰지 않습니다.")
    return resolved


def sanitize(source: Path, destination: Path) -> None:
    """EXIF 없는 사본을 만든다.

    방향 정보를 지우기 전에 픽셀을 실제로 돌려 놓는다. 그러지 않으면 휴대폰
    사진이 90도 누운 채 저장된다. 사람 눈에는 사소해 보여도 모델에게는 다른
    사진이다.
    """
    with Image.open(source) as image:
        image_format = (image.format or "PNG").upper()
        # exif_transpose가 방향 태그를 픽셀에 반영하고 태그를 떼어 준다.
        upright = ImageOps.exif_transpose(image)

        options: dict[str, object] = {}
        if image_format == "JPEG":
            if upright.mode not in {"RGB", "L"}:
                upright = upright.convert("RGB")
            options["quality"] = JPEG_QUALITY
            options["subsampling"] = JPEG_SUBSAMPLING

        # 픽셀만 새 이미지로 옮긴다. 원본의 info를 물려받지 않으므로
        # EXIF·GPS·기기 정보가 사본에 남지 않는다.
        clean = Image.new(upright.mode, upright.size)
        clean.paste(upright)
        clean.save(destination, format=image_format, **options)


def ingest(
    source: Path,
    *,
    root: Path,
    raw_dir: Path,
    sanitized_dir: Path,
    sample_id: str,
) -> IngestedImage:
    """원본을 보관하고 비식별 사본을 만든다.

    파일 이름은 sample_id로 정한다. 촬영 기기가 붙인 이름을 그대로 두면 그
    안에 무엇이 들어 있는지 통제할 수 없다.
    """
    if not source.is_file():
        raise ImageIngestError(f"파일이 없습니다: {source}")

    image_format = _detect_format(source)
    extension = EXTENSION_OF.get(image_format)
    if extension is None:
        raise ImageIngestError(f"다룰 수 없는 이미지 형식입니다: {image_format}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    sanitized_dir.mkdir(parents=True, exist_ok=True)

    raw_path = _inside(root, raw_dir / f"{sample_id}{extension}")
    sanitized_path = _inside(root, sanitized_dir / f"{sample_id}{extension}")

    # 원본은 바꾸지 않고 그대로 복사한다.
    shutil.copy2(source, raw_path)
    sanitize(raw_path, sanitized_path)

    with Image.open(sanitized_path) as saved:
        width, height = saved.size

    return IngestedImage(
        raw_path=raw_path,
        sanitized_path=sanitized_path,
        raw_sha256=sha256_of(raw_path),
        sanitized_sha256=sha256_of(sanitized_path),
        image_format=image_format,
        width=width,
        height=height,
    )


def has_exif(path: Path) -> bool:
    """저장된 파일에 EXIF가 남아 있는지."""
    with Image.open(path) as image:
        exif = image.getexif()
        return bool(exif)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """반쪽 파일이 남지 않게 쓴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_write_text(path: Path, payload: str) -> None:
    atomic_write_bytes(path, payload.encode("utf-8"))
