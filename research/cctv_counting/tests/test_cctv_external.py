"""공개 데이터셋 어댑터.

여기서 지키는 것은 두 가지다. 데이터셋 사정이 아래쪽으로 새지 않는 것,
그리고 같은 영상의 프레임이 학습과 평가에 갈리지 않는 것.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cctv_counting.external import (
    PERMISSIVE,
    RESEARCH_ONLY,
    AdapterError,
    FrameAnnotation,
    parse_stem,
    read_yolo_frame,
    split_by_group,
)


def write_pair(root: Path, stem: str, lines: list[str], size=(200, 100)) -> tuple[Path, Path]:
    from PIL import Image

    image = root / "images" / f"{stem}.jpg"
    label = root / "labels" / f"{stem}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (40, 60, 80)).save(image, format="JPEG")
    label.write_text("\n".join(lines), encoding="utf-8")
    return image, label


# --- 파일 이름에서 출처 읽기 -------------------------------------------------


def test_still_image_has_no_sequence() -> None:
    assert parse_stem("coco_train_000000288442") == ("coco", None, None)


def test_mot_frame_carries_sequence_and_index() -> None:
    source, sequence, index = parse_stem("mot_mot17_MOT17-02-SDP_f000163")

    assert source == "mot"
    assert sequence == "mot17:MOT17-02-SDP"
    assert index == 163


# --- YOLO -> 픽셀 상자 -------------------------------------------------------


def test_normalized_box_becomes_pixels(tmp_path: Path) -> None:
    image, label = write_pair(tmp_path, "coco_a", ["0 0.5 0.5 0.5 0.5"])

    frame, issues = read_yolo_frame(image, label, license_class=PERMISSIVE)

    assert not issues
    assert frame.person_count == 1
    box = frame.boxes[0]
    # 200x100 이미지의 한가운데, 폭 50% 높이 50%.
    assert (box.x1, box.y1, box.x2, box.y2) == (50.0, 25.0, 150.0, 75.0)


def test_both_classes_count_as_person(tmp_path: Path) -> None:
    # person(0)과 pedestrian(1)의 의미 차이가 문서화돼 있지 않다. 우리는
    # 사람 수를 세므로 합친다.
    image, label = write_pair(
        tmp_path, "coco_b", ["0 0.3 0.3 0.2 0.2", "1 0.7 0.7 0.2 0.2"]
    )

    frame, _ = read_yolo_frame(image, label, license_class=PERMISSIVE)

    assert frame.person_count == 2


def test_box_outside_the_image_is_clipped(tmp_path: Path) -> None:
    image, label = write_pair(tmp_path, "coco_c", ["0 0.05 0.05 0.4 0.4"])

    frame, issues = read_yolo_frame(image, label, license_class=PERMISSIVE)

    assert not issues
    assert frame.boxes[0].x1 == 0.0
    assert frame.boxes[0].y1 == 0.0


def test_empty_label_means_nobody(tmp_path: Path) -> None:
    # 사람이 없는 프레임은 정상이다. 상자를 지어내지 않는다.
    image, label = write_pair(tmp_path, "coco_d", [])

    frame, issues = read_yolo_frame(image, label, license_class=PERMISSIVE)

    assert frame.person_count == 0
    assert not issues


def test_missing_label_file_means_nobody(tmp_path: Path) -> None:
    image, label = write_pair(tmp_path, "coco_e", [])
    label.unlink()

    frame, _ = read_yolo_frame(image, label, license_class=PERMISSIVE)

    assert frame.person_count == 0


def test_dropped_lines_are_reported_not_hidden(tmp_path: Path) -> None:
    # 조용히 버리면 사람 수 정답이 달라진 것을 알아챌 수 없다.
    image, label = write_pair(tmp_path, "coco_f", ["0 0.5 0.5", "7 0.5 0.5 0.2 0.2"])

    frame, issues = read_yolo_frame(image, label, license_class=PERMISSIVE)

    assert frame.person_count == 0
    assert len(issues) == 2


def test_zero_area_box_is_dropped_with_a_reason(tmp_path: Path) -> None:
    image, label = write_pair(tmp_path, "coco_g", ["0 0.5 0.5 0.0 0.3"])

    frame, issues = read_yolo_frame(image, label, license_class=PERMISSIVE)

    assert frame.person_count == 0
    assert any("넓이가 0" in issue for issue in issues)


def test_unknown_license_class_is_refused(tmp_path: Path) -> None:
    image, label = write_pair(tmp_path, "coco_h", [])

    with pytest.raises(AdapterError, match="라이선스 등급"):
        read_yolo_frame(image, label, license_class="아마도 괜찮음")


def test_only_permissive_is_usable_for_product(tmp_path: Path) -> None:
    image, label = write_pair(tmp_path, "coco_i", [])

    permissive, _ = read_yolo_frame(image, label, license_class=PERMISSIVE)
    research, _ = read_yolo_frame(image, label, license_class=RESEARCH_ONLY)

    assert permissive.is_usable_for_product
    assert not research.is_usable_for_product


# --- 분할 --------------------------------------------------------------------


def frame(stem: str) -> FrameAnnotation:
    source, sequence, index = parse_stem(stem)
    return FrameAnnotation(
        frame_id=stem,
        source=source,
        image_path=f"images/{stem}.jpg",
        width=200,
        height=100,
        boxes=(),
        license_class=PERMISSIVE,
        sequence=sequence,
        frame_index=index,
    )


def test_video_frames_group_by_sequence() -> None:
    a = frame("mot_mot17_MOT17-02-SDP_f000001")
    b = frame("mot_mot17_MOT17-02-SDP_f000002")

    # 인접 프레임은 사실상 같은 이미지다. 갈리면 안 된다.
    assert a.group_key == b.group_key


def test_still_images_are_their_own_group() -> None:
    a = frame("coco_train_000000000001")
    b = frame("coco_train_000000000002")

    assert a.group_key != b.group_key


def test_a_sequence_never_spans_two_splits() -> None:
    frames = [
        frame(f"mot_mot17_MOT17-{seq:02d}-SDP_f{index:06d}")
        for seq in range(1, 9)
        for index in range(4)
    ]

    assignment = split_by_group(frames, seed=7)

    by_group: dict[str, set[str]] = {}
    for item in frames:
        by_group.setdefault(item.group_key, set()).add(assignment[item.frame_id])
    assert all(len(splits) == 1 for splits in by_group.values())


def test_same_seed_gives_the_same_split() -> None:
    frames = [frame(f"coco_train_{index:012d}") for index in range(30)]

    assert split_by_group(frames, seed=1) == split_by_group(frames, seed=1)


def test_input_order_does_not_change_the_split() -> None:
    frames = [frame(f"coco_train_{index:012d}") for index in range(30)]

    assert split_by_group(frames, seed=1) == split_by_group(frames[::-1], seed=1)


def test_every_split_gets_something() -> None:
    frames = [frame(f"coco_train_{index:012d}") for index in range(30)]

    assignment = split_by_group(frames, seed=1)

    assert set(assignment.values()) == {"train", "val", "test"}


def test_too_few_groups_is_an_error() -> None:
    # 그룹이 둘뿐인데 셋으로 나누면 그게 누수다.
    frames = [
        frame("mot_mot17_MOT17-02-SDP_f000001"),
        frame("mot_mot17_MOT17-04-SDP_f000001"),
    ]

    with pytest.raises(AdapterError, match="나눌 수 없습니다"):
        split_by_group(frames)
