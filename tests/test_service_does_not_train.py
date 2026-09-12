"""서비스는 학습하지 않는다는 것을 소스로 고정한다.

처음에는 "아직 모델이 없다"를 고정하던 테스트다. 이제 서비스는 식수 예측
모델 artifact 하나를 **읽어서 추론**한다. 그 모델은 합성데이터 PoC이고,
학습은 research/demand_forecasting 에서 한다.

그래서 지키는 선을 이렇게 옮긴다.

- 서비스 소스에 학습 코드가 없다. fit, optimizer, checkpoint 같은 말이 없다.
- 서비스 소스가 학습 프레임워크를 직접 import하지 않는다. artifact는 joblib으로
  읽고, sklearn과 lightgbm은 그 안에서 딸려 올 뿐이다.
- ML 런타임은 핵심 의존성이 아니다. 선택 extra(demand-model)이고 버전을 못 박는다.
  extra 없이도 서비스 전체가 import되고 기동된다(수요 예측만 503).
"""

from __future__ import annotations

import ast
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


#: artifact를 읽을 때만 필요한 런타임. 모듈 맨 위에서 import하면 extra가 없는
#: 환경에서 서비스 전체가 import되지 않는다.
MODEL_RUNTIME = ("joblib", "pandas", "numpy", "sklearn", "lightgbm", "xgboost", "scipy")


def test_model_runtime_is_imported_lazily() -> None:
    for path, text in sources():
        tree = ast.parse(text)
        for node in tree.body:
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0] not in MODEL_RUNTIME, f"{path}: {name}"


def test_model_runtime_is_an_optional_pinned_extra() -> None:
    manifest = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))
    extra = manifest["project"]["optional-dependencies"]["demand-model"]
    names = {entry.split("==")[0].strip().lower() for entry in extra}

    assert {"joblib", "scikit-learn", "lightgbm", "pandas", "numpy"} <= names
    # pickle은 라이브러리 버전이 다르면 조용히 다르게 동작할 수 있다.
    assert all("==" in entry for entry in extra), extra


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
