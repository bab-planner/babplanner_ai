"""아직 모델을 만들지 않았다는 것을 소스로 고정한다.

지금까지 한 것은 정답을 정의하고 모으는 도구까지다. 학습 코드가 슬쩍
들어오면 "모델이 있다"는 인상만 생기고 실제로 맞히는 것은 없다.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

#: 학습 프레임워크. 하나라도 들어오면 이 단계의 약속이 깨진 것이다.
ML_FRAMEWORKS = (
    "torch",
    "tensorflow",
    "keras",
    "transformers",
    "ultralytics",
    "onnxruntime",
    "sklearn",
    "scikit-learn",
    "cv2",
    "opencv",
    "xgboost",
    "lightgbm",
)

#: 학습 루프에서만 나오는 말.
TRAINING_TERMS = (
    "optimizer",
    "backward()",
    "train_epoch",
    "checkpoint",
    "nn.Module",
    "fit(",
)


def sources() -> list[tuple[str, str]]:
    return [
        (str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"), path.read_text("utf-8"))
        for path in SOURCE_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    ]


def test_no_ml_framework_is_imported() -> None:
    for path, text in sources():
        for framework in ML_FRAMEWORKS:
            assert f"import {framework}" not in text, f"{path}: {framework}"
            assert f"from {framework}" not in text, f"{path}: {framework}"


def test_no_ml_framework_is_declared() -> None:
    manifest = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))
    declared = " ".join(
        manifest["project"]["dependencies"]
        + manifest["project"].get("optional-dependencies", {}).get("dev", [])
    ).lower()

    for framework in ML_FRAMEWORKS:
        assert framework not in declared, framework


def test_no_training_loop() -> None:
    for path, text in sources():
        for term in TRAINING_TERMS:
            assert term not in text, f"{path}: {term}"


def test_dependencies_stayed_small() -> None:
    # 그룹 분할과 통계는 stdlib로 충분하다. pandas나 Typer를 끌어들이지 않는다.
    manifest = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))
    names = {
        entry.split(">=")[0].split("==")[0].strip().lower()
        for entry in manifest["project"]["dependencies"]
    }

    assert names == {
        "fastapi",
        "uvicorn",
        "python-multipart",
        "pydantic",
        "pydantic-settings",
        "pillow",
    }


def test_collected_data_is_ignored_by_git() -> None:
    # 급식 사진은 한 번 올라가면 지웠다고 사라지지 않는다.
    ignore = (PROJECT_ROOT / ".gitignore").read_text("utf-8")

    assert "data/*" in ignore
    assert "!data/README.md" in ignore


def test_only_the_data_readme_is_tracked() -> None:
    tracked = [
        path
        for path in (PROJECT_ROOT / "data").rglob("*")
        if path.is_file() and path.name != "README.md"
    ]

    assert not tracked, [str(path) for path in tracked]
