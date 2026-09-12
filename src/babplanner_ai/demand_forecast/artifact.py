"""식수 예측 모델 artifact 로더.

학습은 research/demand_forecasting 에서 한다. 이 서비스는 거기서 저장한
artifact(joblib 파일 + metadata.json)를 **읽어서 추론만** 한다. 학습 코드를
import하지 않는다.

- 한 번만 로드한다. 성공이든 실패든 결과를 기억한다. 요청마다 파일을 다시 읽지
  않는다. artifact를 바꾸면 서버를 다시 띄운다.
- 파일이 없거나 쓸 수 없어도 서버는 뜬다. 수요 예측만 unavailable/503이 된다.
- joblib, pandas는 로드할 때 import한다. demand-model extra가 없는 환경에서도
  서비스 전체는 import되고 기동된다.

**joblib 파일은 pickle이다.** 로드하는 순간 파일 안의 코드가 실행될 수 있다.
경로는 운영자가 정하는 설정값이지 요청으로 들어오는 값이 아니며, metadata의
model_sha256과 파일 해시가 맞을 때만 로드한다.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

EXPECTED_TARGET = "meal_count"

#: metadata.versions 의 키 → 설치된 배포판 이름.
_RUNTIME_PACKAGES = {
    "sklearn": "scikit-learn",
    "lightgbm": "lightgbm",
    "numpy": "numpy",
    "pandas": "pandas",
}


class ArtifactError(RuntimeError):
    """artifact를 쓸 수 없다. 메시지는 status의 reason으로 그대로 나간다.

    파일 경로를 담지 않는다.
    """


class Predictor(Protocol):
    def predict_one(self, row: Mapping[str, object], feature_order: Sequence[str]) -> float:
        """한 행을 예측한다. row는 feature_order 순서의 값을 가진다."""
        ...


@dataclass(frozen=True)
class ModelMetadata:
    model_name: str
    target: str
    features: tuple[str, ...]
    synthetic_data: bool
    model_sha256: str | None = None
    trained_at: str | None = None
    training_period: tuple[dt.date, dt.date] | None = None
    categorical_levels: Mapping[str, frozenset[str]] = field(default_factory=dict)
    versions: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Any, buildable: frozenset[str]) -> "ModelMetadata":
        if not isinstance(raw, dict):
            raise ArtifactError("model metadata is invalid")
        # synthetic_data는 반드시 명시돼야 한다. 빠진 것을 실데이터 모델로 읽으면
        # 합성데이터 예측이 경고 없이 나간다.
        synthetic = raw.get("synthetic_data")
        if not isinstance(synthetic, bool):
            raise ArtifactError("model metadata must declare synthetic_data")
        name = raw.get("model_name")
        if not isinstance(name, str) or not name:
            raise ArtifactError("model metadata has no model_name")
        if raw.get("target") != EXPECTED_TARGET:
            raise ArtifactError(f"model target is not {EXPECTED_TARGET}")
        features = raw.get("features")
        if (
            not isinstance(features, list)
            or not features
            or not all(isinstance(f, str) for f in features)
            or len(set(features)) != len(features)
        ):
            raise ArtifactError("model metadata has no valid feature list")
        unknown = sorted(set(features) - buildable)
        if unknown:
            raise ArtifactError(f"model requires features this server cannot build: {unknown}")

        period = raw.get("training_period")
        training_period = None
        if isinstance(period, list) and len(period) == 2:
            try:
                training_period = (dt.date.fromisoformat(period[0]), dt.date.fromisoformat(period[1]))
            except (TypeError, ValueError):
                training_period = None
        levels = raw.get("categorical_levels") or {}
        return cls(
            model_name=name,
            target=raw["target"],
            features=tuple(features),
            synthetic_data=synthetic,
            model_sha256=raw.get("model_sha256"),
            trained_at=raw.get("training_date"),
            training_period=training_period,
            categorical_levels={k: frozenset(map(str, v)) for k, v in levels.items()},
            versions={k: str(v) for k, v in (raw.get("versions") or {}).items()},
        )


@dataclass(frozen=True)
class LoadedModel:
    metadata: ModelMetadata
    predictor: Predictor
    #: 모델 파일 SHA-256 앞 12자리. 저장된 예측이 어느 모델에서 나왔는지 가린다.
    version: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArtifactState:
    model: LoadedModel | None
    reason: str | None = None


class SklearnPipelinePredictor:
    """joblib으로 저장한 sklearn Pipeline을 감싼다.

    Pipeline은 학습 때처럼 컬럼 이름이 있는 DataFrame을 받아야 한다. pandas는
    여기서만, 실제로 예측할 때 import한다.
    """

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline
        # 동시 요청이 같은 모델 객체를 부른다. 추론은 수 ms라 직렬화해도 된다.
        self._lock = threading.Lock()

    def predict_one(self, row: Mapping[str, object], feature_order: Sequence[str]) -> float:
        import pandas as pd

        frame = pd.DataFrame([[row[name] for name in feature_order]], columns=list(feature_order))
        if list(frame.columns) != list(feature_order):
            raise ArtifactError("prediction frame columns differ from metadata features")
        with self._lock:
            return float(self._pipeline.predict(frame)[0])


def load_joblib_pipeline(path: Path, features: Sequence[str]) -> Predictor:
    try:
        import joblib
    except ImportError as error:
        raise ArtifactError(
            "model runtime not installed (install the demand-model extra)"
        ) from error
    try:
        pipeline = joblib.load(path)
    except ImportError as error:
        raise ArtifactError(
            "model runtime not installed (install the demand-model extra)"
        ) from error
    except Exception as error:  # noqa: BLE001 - 어떤 이유든 쓸 수 없는 artifact다
        raise ArtifactError("model artifact could not be loaded") from error
    if not callable(getattr(pipeline, "predict", None)):
        raise ArtifactError("model artifact has no predict()")
    # 학습 때 받은 컬럼 이름을 모델이 기억한다면, metadata와 한 번 더 맞춘다.
    names = getattr(pipeline, "feature_names_in_", None)
    if names is not None and [str(n) for n in names] != list(features):
        raise ArtifactError("feature schema mismatch between model artifact and metadata")
    return SklearnPipelinePredictor(pipeline)


def installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


class ModelArtifactLoader:
    """artifact를 한 번만 로드하고 결과를 기억한다. 스레드 안전하다."""

    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        *,
        buildable_features: frozenset[str],
        allow_synthetic: bool,
        load_predictor: Callable[[Path, Sequence[str]], Predictor] = load_joblib_pipeline,
        version_lookup: Callable[[str], str | None] = installed_version,
    ) -> None:
        self._model_path = model_path
        self._metadata_path = metadata_path
        self._buildable = buildable_features
        self._allow_synthetic = allow_synthetic
        self._load_predictor = load_predictor
        self._version_lookup = version_lookup
        self._lock = threading.Lock()
        self._state: ArtifactState | None = None
        #: 실제로 로드를 시도한 횟수. 한 번만 로드하는지 테스트가 본다.
        self.load_attempts = 0

    def get(self) -> ArtifactState:
        with self._lock:
            if self._state is None:
                self.load_attempts += 1
                try:
                    model = self._load()
                    self._state = ArtifactState(model=model)
                    log.info(
                        "식수 예측 모델을 로드했습니다. name=%s version=%s synthetic=%s",
                        model.metadata.model_name,
                        model.version,
                        model.metadata.synthetic_data,
                    )
                except ArtifactError as error:
                    self._state = ArtifactState(model=None, reason=str(error))
                    log.warning("식수 예측 모델을 쓸 수 없습니다. reason=%s", error)
            return self._state

    def _load(self) -> LoadedModel:
        if not self._model_path.is_file():
            raise ArtifactError("model artifact not found")
        if not self._metadata_path.is_file():
            raise ArtifactError("model metadata not found")
        try:
            raw = json.loads(self._metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ArtifactError("model metadata is invalid") from error
        metadata = ModelMetadata.parse(raw, self._buildable)

        # 합성데이터로 학습한 모델은 운영에서 쓰지 않는다. fixture를 운영에서
        # 막는 것과 같은 이유다. 지어낸 관계를 배운 숫자로 식재료를 주문하게 된다.
        if metadata.synthetic_data and not self._allow_synthetic:
            raise ArtifactError("synthetic-data model is not allowed in production")

        digest = hashlib.sha256(self._model_path.read_bytes()).hexdigest()
        warnings: list[str] = []
        if metadata.model_sha256 is None:
            warnings.append("metadata에 model_sha256이 없어 모델 파일과 짝이 맞는지 확인하지 못했습니다.")
        elif metadata.model_sha256 != digest:
            raise ArtifactError("model artifact does not match metadata (sha256)")

        for key, package in _RUNTIME_PACKAGES.items():
            trained_with = metadata.versions.get(key)
            running = self._version_lookup(package)
            if trained_with is not None and running is not None and trained_with != running:
                warnings.append(f"{package} 버전이 다릅니다. 학습 {trained_with}, 실행 {running}.")

        predictor = self._load_predictor(self._model_path, metadata.features)
        return LoadedModel(
            metadata=metadata, predictor=predictor, version=digest[:12], warnings=tuple(warnings)
        )
