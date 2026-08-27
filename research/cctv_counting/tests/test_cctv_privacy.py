"""개인정보 경계.

이 파일은 성능이 아니라 약속을 지킨다. "얼굴 인식은 안 합니다"는 README에만
있으면 언제든 조용히 깨진다. 코드에 그런 것이 들어오면 여기가 빨개지도록
둔다.
"""

from __future__ import annotations

import re
from pathlib import Path

from cctv_counting.groundtruth import CROSSING_GT_FIELDS, OCCUPANCY_GT_FIELDS
from cctv_counting.observations import FORBIDDEN_FIELD_TOKENS, OBSERVATION_FIELDS
from cctv_counting.summary import VideoSummary

SRC = Path(__file__).resolve().parents[1] / "src" / "cctv_counting"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

#: 코드에 나타나면 안 되는 것들. 주석과 문서 문자열은 이 규칙에서 뺀다.
#: "얼굴을 다루지 않는다"라고 적은 문장 자체가 걸리면 안 되기 때문이다.
FACE_PATTERNS = (
    r"\bface_recognition\b",
    r"\bfacenet\b",
    r"\bdeepface\b",
    r"\barcface\b",
    r"\binsightface\b",
    r"\bdetect_faces?\b",
    r"\bface_embedding\b",
    r"\bface_encoding\b",
    r"\bhaarcascade\w*\b",
    r"\breid_",
    r"\breidentif",
)

#: 신원을 가리키는 필드 이름. 이쪽은 observations.py 한 곳에 예외를 둔다.
#: 그 파일이 바로 "이 이름들은 스키마에 들어오면 안 된다"는 목록을 선언하는
#: 곳이기 때문이다. 금지어 목록이 스스로를 위반한다고 보면 목록을 코드로
#: 남길 수 없다. 대신 그 파일에서는 목록 안에만 있는지를 따로 확인한다.
IDENTITY_PATTERNS = (
    r"\bperson_id\b",
    r"\bstudent_id\b",
    r"\bphone_number\b",
    r"\buser_id\b",
)

DENYLIST_MODULE = "observations.py"


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """주석과 docstring을 뺀 실제 코드 줄.

    아주 단순하게 처리한다. 삼중 따옴표 블록을 통째로 건너뛰고 # 이후를
    자른다. 문자열 안에 우연히 들어간 것까지 완벽히 가리지는 못하지만,
    금지된 라이브러리를 실제로 import하는 것은 확실히 잡힌다.
    """
    lines: list[tuple[int, str]] = []
    in_block = False
    delimiter = ""
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw
        if in_block:
            if delimiter in line:
                line = line.split(delimiter, 1)[1]
                in_block = False
            else:
                continue
        while True:
            match = re.search(r'"""|\'\'\'', line)
            if match is None:
                break
            opening = match.group(0)
            before = line[: match.start()]
            rest = line[match.end() :]
            if opening in rest:
                line = before + rest.split(opening, 1)[1]
                continue
            line = before
            in_block = True
            delimiter = opening
            break
        line = line.split("#", 1)[0]
        if line.strip():
            lines.append((number, line))
    return lines


def _python_files() -> list[Path]:
    return sorted([*SRC.rglob("*.py"), *SCRIPTS.rglob("*.py")])


def test_there_are_files_to_check():
    assert len(_python_files()) >= 10


def test_no_face_recognition_code_anywhere():
    """얼굴 검출/대조와 re-identification 라이브러리가 코드에 없어야 한다."""
    offences: list[str] = []
    for path in _python_files():
        for number, line in _code_lines(path):
            for pattern in FACE_PATTERNS:
                if re.search(pattern, line, flags=re.IGNORECASE):
                    offences.append(f"{path.name}:{number} {line.strip()}")
    assert not offences, "얼굴/재식별 코드가 있다:\n" + "\n".join(offences)


def test_no_identity_fields_outside_the_denylist():
    offences: list[str] = []
    for path in _python_files():
        if path.name == DENYLIST_MODULE:
            continue
        for number, line in _code_lines(path):
            for pattern in IDENTITY_PATTERNS:
                if re.search(pattern, line, flags=re.IGNORECASE):
                    offences.append(f"{path.name}:{number} {line.strip()}")
    assert not offences, "신원 필드가 있다:\n" + "\n".join(offences)


def test_denylist_module_only_mentions_them_inside_the_denylist():
    """예외를 둔 파일이 실제로 목록 선언에만 쓰고 있는지 확인한다."""
    source = (SRC / DENYLIST_MODULE).read_text(encoding="utf-8")
    start = source.index("FORBIDDEN_FIELD_TOKENS")
    declaration = source[start : source.index(")", start)]
    for _, line in _code_lines(SRC / DENYLIST_MODULE):
        for pattern in IDENTITY_PATTERNS:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                assert match.group(0) in declaration, line.strip()


def test_observation_schema_has_no_identifying_field():
    for field in OBSERVATION_FIELDS:
        for token in FORBIDDEN_FIELD_TOKENS:
            assert token not in field


def test_ground_truth_schemas_have_no_identifying_field():
    for fields in (OCCUPANCY_GT_FIELDS, CROSSING_GT_FIELDS):
        for field in fields:
            for token in FORBIDDEN_FIELD_TOKENS:
                assert token not in field


def test_summary_payload_has_no_identifying_field():
    summary = VideoSummary(
        video="demo.mp4",
        duration_seconds=1.0,
        frames_processed=1,
        peak_occupancy=1,
        average_occupancy=1.0,
        entries_total=1,
        exits_total=0,
        processing_fps=10.0,
        low_reliability_frame_ratio=0.0,
    )
    for key in summary.to_dict():
        for token in FORBIDDEN_FIELD_TOKENS:
            assert token.replace("_", "") not in key.lower()


def test_summary_video_field_is_a_name_not_a_full_path():
    """전체 경로에는 사람 이름이나 현장 이름이 섞이곤 한다.

    run_video.py가 파일 이름만 넣는 것을 여기서 고정한다. 경로 전체는
    metadata.json 쪽에만 남는다.
    """
    source = (SCRIPTS / "run_video.py").read_text(encoding="utf-8")
    assert "video=args.input.name" in source
