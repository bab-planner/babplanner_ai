"""현장 도구 전체 흐름과 QC."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from babplanner_ai.collection.cli import CURRENT_PROTOCOL_VERSION, main
from babplanner_ai.collection.exporter import export
from babplanner_ai.collection.ids import is_safe_id
from babplanner_ai.collection.journal import CollectionStore
from babplanner_ai.collection.qc import build_report
from babplanner_ai.dataset.schema import CURRENT_LABEL_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def photo(path: Path, colour: tuple[int, int, int] = (200, 170, 120)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 16), colour).save(path, format="JPEG", quality=95)
    return path


def run(*argv: str) -> int:
    return main(list(argv))


@pytest.fixture
def field(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """회차 하나가 준비된 상태."""
    root = tmp_path / "collection"
    assert run(
        "init-session",
        "--root", str(root),
        "--dataset-version", "test-2026-08",
        "--scale-resolution-g", "1.0",
    ) == 0
    session_id = re.search(r"(cs_[0-9a-f]+)", capsys.readouterr().out).group(1)

    assert run(
        "tare-add",
        "--root", str(root),
        "--session", session_id,
        "--description", "4칸 식판",
        "--weight", "400.0",
        "--weight", "400.5",
        "--weight", "399.5",
    ) == 0
    tare_id = re.search(r"(tp_[0-9a-f]+)", capsys.readouterr().out).group(1)

    return root, session_id, tare_id


def add(
    root: Path,
    session: str,
    tare: str,
    image: Path,
    *extra: str,
    capture: str = "cap_000000000001",
    participant: str = "p_000000000001",
    menu_id: str = "3",
    post: str = "500.0",
) -> int:
    return run(
        "add-sample",
        "--root", str(root),
        "--session", session,
        "--image", str(image),
        "--capture-session", capture,
        "--participant", participant,
        "--menu-id", menu_id,
        "--tare-profile", tare,
        "--pre-weight", "900.0",
        "--post-weight", post,
        "--mass-source", "MEASURED",
        *extra,
    )


def test_init_session_creates_the_store(field, capsys) -> None:
    root, session_id, _ = field

    assert is_safe_id(session_id)
    assert (root / "sessions" / session_id / "session.json").is_file()


def test_new_participant_prints_an_opaque_id(capsys) -> None:
    assert run("new-participant") == 0

    value = capsys.readouterr().out.strip()
    assert is_safe_id(value)
    assert value.startswith("p_")


def test_new_capture_session_prints_an_opaque_id(capsys) -> None:
    assert run("new-capture-session") == 0

    assert capsys.readouterr().out.strip().startswith("cap_")


def test_tare_add_reports_the_spread(field, capsys) -> None:
    root, session_id, _ = field
    assert run(
        "tare-add",
        "--root", str(root),
        "--session", session_id,
        "--description", "국그릇만",
        "--weight", "120.0",
        "--weight", "121.0",
    ) == 0

    output = capsys.readouterr().out
    assert "측정 2회" in output
    assert "편차" in output


def test_tare_weight_must_be_positive(field) -> None:
    root, session_id, _ = field

    assert run(
        "tare-add",
        "--root", str(root),
        "--session", session_id,
        "--description", "이상한 식판",
        "--weight", "0",
    ) == 1


def test_add_sample_reports_the_rate(field, tmp_path, capsys) -> None:
    root, session_id, tare_id = field

    assert add(root, session_id, tare_id, photo(tmp_path / "a.jpg")) == 0

    output = capsys.readouterr().out
    assert "잔반률   0.2000" in output


def test_add_sample_requires_the_essentials(field, tmp_path) -> None:
    root, session_id, _ = field

    with pytest.raises(SystemExit):
        run("add-sample", "--root", str(root), "--session", session_id,
            "--image", str(photo(tmp_path / "a.jpg")))


def test_invalid_measurement_fails_and_leaves_nothing(field, tmp_path) -> None:
    root, session_id, tare_id = field
    store = CollectionStore(root, session_id)

    # 식사 후 중량이 식기보다 가볍다.
    assert add(
        root, session_id, tare_id, photo(tmp_path / "a.jpg"), post="100.0"
    ) == 1

    # 기록으로 남지 못한 이미지가 남지 않는다.
    assert not list(store.sanitized_images_dir.glob("*"))
    assert not store.journal_path.exists()


def test_duplicate_image_is_refused(field, tmp_path, capsys) -> None:
    root, session_id, tare_id = field
    source = photo(tmp_path / "a.jpg")
    assert add(root, session_id, tare_id, source) == 0
    capsys.readouterr()

    assert add(root, session_id, tare_id, source, participant="p_000000000002") == 1


def test_correction_appends_and_keeps_history(field, tmp_path, capsys) -> None:
    root, session_id, tare_id = field
    assert add(root, session_id, tare_id, photo(tmp_path / "a.jpg")) == 0
    sample_id = re.search(r"기록했습니다: ([0-9a-f]{32})", capsys.readouterr().out).group(1)

    assert run(
        "correct-sample",
        "--root", str(root),
        "--session", session_id,
        "--sample-id", sample_id,
        "--post-weight", "450.0",
        "--note", "저울을 잘못 읽음",
    ) == 0

    store = CollectionStore(root, session_id)
    assert len(store.read_records()) == 2
    assert len(store.latest_records()) == 1
    assert store.latest_records()[0].post_meal_total_weight_g == 450.0


def test_correction_needs_something_to_change(field, tmp_path, capsys) -> None:
    root, session_id, tare_id = field
    assert add(root, session_id, tare_id, photo(tmp_path / "a.jpg")) == 0
    sample_id = re.search(r"기록했습니다: ([0-9a-f]{32})", capsys.readouterr().out).group(1)

    assert run(
        "correct-sample", "--root", str(root), "--session", session_id,
        "--sample-id", sample_id,
    ) == 1


def test_export_then_validate(field, tmp_path, capsys) -> None:
    root, session_id, tare_id = field
    add(root, session_id, tare_id, photo(tmp_path / "a.jpg", (10, 20, 30)))
    add(
        root, session_id, tare_id, photo(tmp_path / "b.jpg", (40, 50, 60)),
        participant="p_000000000002", menu_id="4",
    )
    capsys.readouterr()

    assert run("export", "--root", str(root), "--session", session_id) == 0

    store = CollectionStore(root, session_id)
    manifest = store.exports_dir / "manifest.csv"
    assert manifest.is_file()

    from babplanner_ai.dataset.cli import main as dataset_main

    assert dataset_main(
        ["validate", "--manifest", str(manifest), "--root", str(store.session_dir)]
    ) == 0


def test_qc_reports_the_scale_resolution(field, tmp_path, capsys) -> None:
    root, session_id, tare_id = field
    add(root, session_id, tare_id, photo(tmp_path / "a.jpg"))
    capsys.readouterr()

    assert run("qc", "--root", str(root), "--session", session_id) == 0

    output = capsys.readouterr().out
    assert "저울 눈금" in output
    assert "1.0 g" in output


def test_qc_never_lists_participants(field, tmp_path, capsys) -> None:
    root, session_id, tare_id = field
    add(root, session_id, tare_id, photo(tmp_path / "a.jpg"), participant="p_abcdef123456")
    capsys.readouterr()

    run("qc", "--root", str(root), "--session", session_id)

    output = capsys.readouterr().out
    # 몇 명인지만 센다. 누구인지는 찍지 않는다.
    assert "p_abcdef123456" not in output
    assert "참가자 그룹" in output


def test_qc_counts_corrections(field, tmp_path, capsys) -> None:
    root, session_id, tare_id = field
    add(root, session_id, tare_id, photo(tmp_path / "a.jpg"))
    sample_id = re.search(r"기록했습니다: ([0-9a-f]{32})", capsys.readouterr().out).group(1)
    run("correct-sample", "--root", str(root), "--session", session_id,
        "--sample-id", sample_id, "--post-weight", "450.0")
    capsys.readouterr()

    report = build_report(CollectionStore(root, session_id))

    assert report.total_raw_records == 2
    assert report.effective_samples == 1
    assert report.corrected_samples == 1


def test_qc_shows_tare_spread(field, tmp_path, capsys) -> None:
    root, session_id, tare_id = field
    add(root, session_id, tare_id, photo(tmp_path / "a.jpg"))

    report = build_report(CollectionStore(root, session_id))

    assert report.tare_profiles
    _, count, representative, spread = report.tare_profiles[0]
    assert count == 3
    assert representative == pytest.approx(400.0)
    assert spread == pytest.approx(1.0)


def test_qc_shows_the_edges_without_judging(field, tmp_path, capsys) -> None:
    root, session_id, tare_id = field
    add(root, session_id, tare_id, photo(tmp_path / "a.jpg", (10, 20, 30)), post="400.0")
    add(
        root, session_id, tare_id, photo(tmp_path / "b.jpg", (40, 50, 60)),
        participant="p_000000000002", post="900.0",
    )

    report = build_report(CollectionStore(root, session_id))

    assert report.near_zero == 1
    assert report.near_one == 1
    # 통계만 보여주고 합격 여부를 말하지 않는다.
    assert "실패" not in report.render()


def test_close_session_blocks_new_samples(field, tmp_path, capsys) -> None:
    root, session_id, tare_id = field

    assert run("close-session", "--root", str(root), "--session", session_id) == 0
    capsys.readouterr()

    assert add(root, session_id, tare_id, photo(tmp_path / "a.jpg")) == 1


def test_closing_twice_is_harmless(field, capsys) -> None:
    root, session_id, _ = field
    run("close-session", "--root", str(root), "--session", session_id)
    capsys.readouterr()

    assert run("close-session", "--root", str(root), "--session", session_id) == 0
    assert "이미 닫혀" in capsys.readouterr().out


def test_unknown_session_fails(tmp_path: Path) -> None:
    assert run("qc", "--root", str(tmp_path), "--session", "cs_000000000009") == 1


# --- 정답 제외 --------------------------------------------------------------
#
# 메뉴에 한 번 걸면 그 메뉴의 표본에 자동으로 붙는다. 수집자는 표본마다 다시
# 판단하지 않는다.


def exclude(root: Path, session: str, *extra: str, menu_id: str = "7") -> int:
    return run(
        "menu-exclusion",
        "--root", str(root),
        "--session", session,
        "--menu-id", menu_id,
        *extra,
    )


def test_menu_exclusion_propagates_to_new_samples(field, tmp_path, capsys) -> None:
    root, session_id, tare_id = field
    assert exclude(root, session_id, "--reason", "BONE_IN", "--note", "닭볶음탕") == 0
    capsys.readouterr()

    assert add(root, session_id, tare_id, photo(tmp_path / "a.jpg"), menu_id="7") == 0

    output = capsys.readouterr().out
    assert "정답 제외 BONE_IN" in output
    assert "관문 계산에서 빠집니다" in output

    record = CollectionStore(root, session_id).latest_records()[0]
    assert record.gt_exclusion_reason is not None
    assert record.gt_exclusion_reason.value == "BONE_IN"


def test_other_menus_are_untouched(field, tmp_path, capsys) -> None:
    root, session_id, tare_id = field
    assert exclude(root, session_id, "--reason", "BONE_IN") == 0

    assert add(root, session_id, tare_id, photo(tmp_path / "b.jpg"), menu_id="3") == 0

    record = CollectionStore(root, session_id).latest_records()[0]
    assert record.gt_exclusion_reason is None


def test_other_needs_a_note_and_says_so_plainly(field, capsys) -> None:
    root, session_id, _ = field

    assert exclude(root, session_id, "--reason", "OTHER") == 1

    # pydantic이 내는 말은 현장에서 읽기 어렵다.
    assert "이유를 함께 적습니다" in capsys.readouterr().err


def test_clearing_does_not_reach_back(field, tmp_path, capsys) -> None:
    root, session_id, tare_id = field
    assert exclude(root, session_id, "--reason", "BONE_IN") == 0
    assert add(root, session_id, tare_id, photo(tmp_path / "c.jpg"), menu_id="7") == 0
    capsys.readouterr()

    assert exclude(root, session_id, "--reason", "NONE") == 0

    output = capsys.readouterr().out
    assert "이미 기록한 표본이 1건" in output
    assert "소급되지 않습니다" in output
    # 이미 적힌 표본은 그대로다. 조용히 바꾸면 어제 내보낸 매니페스트와
    # 오늘 것이 달라지고 이유를 알 수 없다.
    record = CollectionStore(root, session_id).latest_records()[0]
    assert record.gt_exclusion_reason is not None


def test_correct_sample_can_clear_one_sample(field, tmp_path, capsys) -> None:
    root, session_id, tare_id = field
    assert exclude(root, session_id, "--reason", "BONE_IN") == 0
    assert add(root, session_id, tare_id, photo(tmp_path / "d.jpg"), menu_id="7") == 0
    sample_id = CollectionStore(root, session_id).latest_records()[0].sample_id
    capsys.readouterr()

    assert run(
        "correct-sample",
        "--root", str(root),
        "--session", session_id,
        "--sample-id", sample_id,
        "--gt-exclusion-reason", "NONE",
        "--note", "뼈 없는 부위만 배식된 것을 확인",
    ) == 0

    assert "정답 제외를 해제했습니다" in capsys.readouterr().out
    assert CollectionStore(root, session_id).latest_records()[0].gt_exclusion_reason is None


def test_qc_separates_usable_from_excluded(field, tmp_path, capsys) -> None:
    root, session_id, tare_id = field
    assert exclude(root, session_id, "--reason", "BONE_IN") == 0
    assert add(root, session_id, tare_id, photo(tmp_path / "e.jpg", (10, 20, 30)), menu_id="7") == 0
    assert add(root, session_id, tare_id, photo(tmp_path / "f.jpg", (40, 50, 60)), menu_id="3") == 0

    report = build_report(CollectionStore(root, session_id))

    assert report.effective_samples == 2
    assert report.supervised_usable == 1
    assert report.gt_excluded == 1
    assert report.gt_exclusion_reasons["BONE_IN"] == 1
    assert report.gt_exclusion_by_menu[7] == 1
    # 관문에 쓰는 메뉴 수는 제외 표본을 세지 않는다.
    assert report.menu_count == 2
    assert report.usable_menu_count == 1


def test_export_carries_the_exclusion(field, tmp_path, capsys) -> None:
    root, session_id, tare_id = field
    assert exclude(root, session_id, "--reason", "BONE_IN") == 0
    assert add(root, session_id, tare_id, photo(tmp_path / "g.jpg", (11, 22, 33)), menu_id="7") == 0
    assert add(root, session_id, tare_id, photo(tmp_path / "h.jpg", (44, 55, 66)), menu_id="3") == 0

    result = export(CollectionStore(root, session_id), write=False)

    by_menu = {sample.menu_id: sample for sample in result.samples}
    assert by_menu[7].is_gt_excluded
    assert not by_menu[3].is_gt_excluded
    # 제외 표본도 매니페스트에 남는다. 버리는 것이 아니라 빼는 것이다.
    assert len(result.samples) == 2


def test_init_session_records_the_new_protocol_version(field) -> None:
    root, session_id, _ = field

    session = CollectionStore(root, session_id).read_session()

    # 산술식은 그대로이므로 label_version은 바뀌지 않는다.
    assert session.label_version == CURRENT_LABEL_VERSION
    # 수집 규칙이 바뀌었으므로 protocol_version은 올라간다.
    assert session.protocol_version == CURRENT_PROTOCOL_VERSION


def test_module_entry_point_runs() -> None:
    # python -m 으로 실제 실행되는지 본다. 현장에서는 이 형태로 쓴다.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    result = subprocess.run(
        [sys.executable, "-m", "babplanner_ai.collection", "new-participant"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().startswith("p_")
