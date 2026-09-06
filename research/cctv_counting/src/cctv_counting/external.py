"""공개 데이터셋을 공통 평가 스키마로 옮긴다.

**데이터셋마다 다른 사정을 detector 안에 넣지 않는다.** YOLO 정규화 좌표,
클래스가 둘로 나뉜 것, 파일 이름에 출처가 박혀 있는 것 — 전부 여기서
흡수하고, 아래쪽은 `FrameAnnotation` 하나만 안다.

그래야 데이터셋이 늘어날 때 detector와 지표 코드가 그대로 남는다. 반대로
하면 "이 조건은 어느 데이터셋 때문인가"를 나중에 아무도 답하지 못한다.

여기서 하지 않는 것:

- 상자를 지어내지 않는다. 라벨이 비면 사람 0명이다.
- 클래스를 늘리지 않는다. 우리 과제는 사람 수 하나다.
- 이미지를 열어 보정하지 않는다. 크기만 읽는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from cctv_counting.geometry import BoundingBox, GeometryError

#: 라이선스 등급. 코드 경로에 들어갈 수 있는 것과 아닌 것을 값으로 구분한다.
#: 문자열을 자유롭게 쓰면 오타 하나로 비상업 데이터가 섞인다.
PERMISSIVE = "permissive"
RESEARCH_ONLY = "research_only"

#: 이 데이터셋의 클래스는 `person`과 `pedestrian` 둘이다. 두 이름의 의미
#: 차이가 문서화돼 있지 않고, 우리가 세는 것은 사람 한 종류다. 그래서
#: 합친다 — 나눠 둔 채로 쓰면 무엇을 셌는지 설명할 수 없다.
PERSON_CLASS_IDS = frozenset({0, 1})

#: `mot_mot17_MOT17-02-SDP_f000163` 처럼 출처·시퀀스·프레임이 이름에 있다.
_MOT = re.compile(r"^mot_(?P<sub>mot\d+)_(?P<sequence>.+)_f(?P<frame>\d+)$")


class AdapterError(ValueError):
    """주석을 공통 스키마로 옮길 수 없다."""


@dataclass(frozen=True)
class FrameAnnotation:
    """프레임 하나의 정답.

    상자는 픽셀 좌표다. 정규화 좌표를 그대로 들고 다니면 지표를 계산할 때마다
    이미지 크기를 다시 찾아야 하고, 한 번이라도 빠뜨리면 조용히 틀린다.
    """

    frame_id: str
    source: str
    image_path: str
    width: int
    height: int
    boxes: tuple[BoundingBox, ...]
    license_class: str
    sequence: str | None = None
    frame_index: int | None = None

    @property
    def person_count(self) -> int:
        return len(self.boxes)

    @property
    def group_key(self) -> str:
        """분할을 묶는 단위.

        같은 영상의 프레임이 학습과 평가에 갈리면 인접 프레임이 사실상 같은
        이미지라 성능이 실제보다 좋아 보인다. 시퀀스가 없는 정지 이미지는
        자기 자신이 그룹이다 — 서로 독립이므로 묶을 이유가 없다.
        """
        if self.sequence:
            return f"{self.source}:{self.sequence}"
        return f"{self.source}:{self.frame_id}"

    @property
    def is_usable_for_product(self) -> bool:
        return self.license_class == PERMISSIVE


def parse_stem(stem: str) -> tuple[str, str | None, int | None]:
    """파일 이름에서 (출처, 시퀀스, 프레임 번호)를 읽는다."""
    match = _MOT.match(stem)
    if match:
        return "mot", f"{match['sub']}:{match['sequence']}", int(match["frame"])
    return stem.split("_")[0], None, None


def _to_box(
    parts: list[str], width: int, height: int
) -> BoundingBox | None:
    """YOLO 한 줄을 픽셀 상자로.

    이미지 밖으로 나간 상자는 잘라 넣는다. 잘라서 넓이가 0이 되면 상자가
    아니므로 버린다 — 버린 수는 호출자가 센다.
    """
    center_x, center_y, box_width, box_height = (float(value) for value in parts[1:5])
    x1 = (center_x - box_width / 2) * width
    y1 = (center_y - box_height / 2) * height
    x2 = (center_x + box_width / 2) * width
    y2 = (center_y + box_height / 2) * height

    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(width), x2), min(float(height), y2)
    try:
        return BoundingBox(x1, y1, x2, y2)
    except GeometryError:
        return None


def read_yolo_frame(
    image_path: Path,
    label_path: Path,
    *,
    license_class: str,
    root: Path | None = None,
    size: tuple[int, int] | None = None,
) -> tuple[FrameAnnotation, list[str]]:
    """이미지 하나와 그 라벨을 공통 스키마로.

    두 번째 값은 **버린 줄의 사유**다. 조용히 버리지 않는다 — 몇 줄이 왜
    빠졌는지 모르면 사람 수 정답이 달라진 것을 알아챌 수 없다.
    """
    if license_class not in (PERMISSIVE, RESEARCH_ONLY):
        raise AdapterError(f"알 수 없는 라이선스 등급입니다: {license_class}")

    if size is None:
        from PIL import Image

        with Image.open(image_path) as image:
            size = image.size
    width, height = size

    issues: list[str] = []
    boxes: list[BoundingBox] = []
    # 라벨 파일이 없거나 비어 있으면 사람이 없는 프레임이다. 정상이다.
    text = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
    for number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            issues.append(f"{number}행: 칸이 {len(parts)}개입니다")
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            issues.append(f"{number}행: 클래스를 읽을 수 없습니다")
            continue
        if class_id not in PERSON_CLASS_IDS:
            issues.append(f"{number}행: 사람이 아닌 클래스 {class_id}")
            continue
        try:
            box = _to_box(parts, width, height)
        except ValueError:
            issues.append(f"{number}행: 좌표를 읽을 수 없습니다")
            continue
        if box is None:
            issues.append(f"{number}행: 잘라내니 넓이가 0입니다")
            continue
        boxes.append(box)

    stem = image_path.stem
    source, sequence, frame_index = parse_stem(stem)
    relative = (
        image_path.relative_to(root).as_posix() if root else image_path.as_posix()
    )
    return (
        FrameAnnotation(
            frame_id=stem,
            source=source,
            image_path=relative,
            width=width,
            height=height,
            boxes=tuple(boxes),
            license_class=license_class,
            sequence=sequence,
            frame_index=frame_index,
        ),
        issues,
    )


#: 기본 분할 비율. 정책이 아니라 출발점이고 호출할 때 바꿀 수 있다.
DEFAULT_RATIOS = (0.7, 0.15, 0.15)
DEFAULT_SEED = 42


def split_by_group(
    frames: list[FrameAnnotation],
    *,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> dict[str, str]:
    """그룹째로 나눈다. 프레임 하나씩 무작위로 나누지 않는다.

    제공된 split을 쓰지 않는 이유가 여기 있다 — 그쪽은 같은 영상의 연속
    프레임이 train과 test에 갈려 있다. 인접 프레임은 사실상 같은 이미지다.

    같은 프레임 목록과 같은 seed면 언제나 같은 결과가 나온다. 그러지 않으면
    어제의 숫자와 오늘의 숫자를 비교할 수 없다.
    """
    import random

    if not frames:
        raise AdapterError("프레임이 없습니다.")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise AdapterError(f"비율의 합이 1이 아닙니다: {sum(ratios)}")

    members: dict[str, list[str]] = {}
    for frame in frames:
        members.setdefault(frame.group_key, []).append(frame.frame_id)

    keys = sorted(members)
    if len(keys) < 3:
        raise AdapterError(
            f"그룹이 {len(keys)}개뿐이라 train/val/test로 나눌 수 없습니다."
        )

    # 정렬한 뒤 섞는다. 입력 순서가 결과를 바꾸지 않게 하려는 것이다.
    random.Random(seed).shuffle(keys)
    train_count = max(1, round(len(keys) * ratios[0]))
    val_count = max(1, round(len(keys) * ratios[1]))
    if train_count + val_count >= len(keys):
        train_count, val_count = len(keys) - 2, 1

    assignment: dict[str, str] = {}
    for name, group_keys in (
        ("train", keys[:train_count]),
        ("val", keys[train_count : train_count + val_count]),
        ("test", keys[train_count + val_count :]),
    ):
        for key in group_keys:
            for frame_id in members[key]:
                assignment[frame_id] = name
    return assignment


def read_yolo_directory(
    images_dir: Path,
    labels_dir: Path,
    *,
    license_class: str,
    root: Path | None = None,
) -> tuple[list[FrameAnnotation], dict[str, list[str]]]:
    """디렉터리 한 쌍을 통째로 읽는다. 이름 순서가 결정적이다."""
    frames: list[FrameAnnotation] = []
    problems: dict[str, list[str]] = {}
    for image_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{image_path.stem}.txt"
        frame, issues = read_yolo_frame(
            image_path, label_path, license_class=license_class, root=root
        )
        frames.append(frame)
        if issues:
            problems[frame.frame_id] = issues
    return frames, problems
