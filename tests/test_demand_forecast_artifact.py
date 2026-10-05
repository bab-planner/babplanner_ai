"""식수 예측 모델 artifact 로더.

한 번만 로드하는지, 쓸 수 없는 artifact를 쓰지 않는지, 그리고 그 이유를
경로 없이 알려주는지를 본다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from babplanner_ai.config import (
    DEFAULT_DEMAND_FORECAST_METADATA_PATH,
    DEFAULT_DEMAND_FORECAST_MODEL_PATH,
)
from babplanner_ai.demand_forecast.artifact import (
    ArtifactError,
    ModelArtifactLoader,
    load_joblib_pipeline,
)
from babplanner_ai.demand_forecast.features import BUILDABLE_FEATURES
from tests.demand_forecast_support import (
    FAKE_MODEL_BYTES,
    FEATURES,
    CountingPredictorFactory,
    FakePredictor,
    fake_loader,
    metadata,
    write_artifact,
)


def reason_of(loader: ModelArtifactLoader) -> str | None:
    return loader.get().reason


class TestLoading:
    def test_valid_artifact_is_loaded(self, tmp_path: Path) -> None:
        loader, _ = fake_loader(tmp_path)

        state = loader.get()

        assert state.model is not None
        assert state.reason is None
        assert state.model.metadata.model_name == "fake_model"
        assert state.model.metadata.synthetic_data is True
        assert state.model.metadata.features == tuple(FEATURES)
        assert state.model.version == hashlib.sha256(FAKE_MODEL_BYTES).hexdigest()[:12]

    def test_artifact_is_loaded_only_once(self, tmp_path: Path) -> None:
        loader, factory = fake_loader(tmp_path)

        for _ in range(5):
            loader.get()

        # 요청마다 joblib 파일을 다시 읽으면 안 된다.
        assert factory.loads == 1
        assert loader.load_attempts == 1

    def test_a_failed_load_is_not_retried_per_request(self, tmp_path: Path) -> None:
        loader = ModelArtifactLoader(
            tmp_path / "missing.joblib",
            tmp_path / "missing.json",
            buildable_features=BUILDABLE_FEATURES,
            allow_synthetic=True,
        )

        for _ in range(5):
            loader.get()

        assert loader.load_attempts == 1


class TestRefusals:
    def test_missing_artifact(self, tmp_path: Path) -> None:
        loader = ModelArtifactLoader(
            tmp_path / "missing.joblib",
            tmp_path / "missing.json",
            buildable_features=BUILDABLE_FEATURES,
            allow_synthetic=True,
        )

        assert reason_of(loader) == "model artifact not found"
        assert loader.get().model is None

    def test_missing_metadata(self, tmp_path: Path) -> None:
        model_path, metadata_path = write_artifact(tmp_path)
        metadata_path.unlink()
        loader = ModelArtifactLoader(
            model_path, metadata_path, buildable_features=BUILDABLE_FEATURES, allow_synthetic=True
        )

        assert reason_of(loader) == "model metadata not found"

    def test_broken_metadata(self, tmp_path: Path) -> None:
        loader, factory = fake_loader(tmp_path)
        (tmp_path / "metadata.json").write_text("{not json", encoding="utf-8")

        assert reason_of(loader) == "model metadata is invalid"
        assert factory.loads == 0

    @pytest.mark.parametrize("value", [None, "true", 1])
    def test_synthetic_flag_must_be_an_explicit_boolean(self, tmp_path: Path, value: object) -> None:
        meta = metadata(synthetic_data=value)
        if value is None:
            del meta["synthetic_data"]
        loader, factory = fake_loader(tmp_path, meta=meta)

        # 빠진 것을 실데이터 모델로 읽으면 합성데이터 예측이 경고 없이 나간다.
        assert reason_of(loader) == "model metadata must declare synthetic_data"
        assert factory.loads == 0

    def test_wrong_target(self, tmp_path: Path) -> None:
        loader, _ = fake_loader(tmp_path, meta=metadata(target="waste_rate"))

        assert reason_of(loader) == "model target is not meal_count"

    def test_feature_the_server_cannot_build(self, tmp_path: Path) -> None:
        loader, factory = fake_loader(tmp_path, meta=metadata(features=FEATURES + ["tomorrow_cctv"]))

        assert "cannot build" in (reason_of(loader) or "")
        assert "tomorrow_cctv" in (reason_of(loader) or "")
        assert factory.loads == 0

    def test_artifact_that_does_not_match_metadata_is_never_unpickled(self, tmp_path: Path) -> None:
        loader, factory = fake_loader(tmp_path, meta=metadata(model_sha256="0" * 64))

        assert reason_of(loader) == "model artifact does not match metadata (sha256)"
        # 짝이 안 맞는 파일은 열어 보지도 않는다. pickle은 여는 순간 코드가 돈다.
        assert factory.loads == 0

    def test_production_refuses_a_synthetic_model(self, tmp_path: Path) -> None:
        loader, factory = fake_loader(tmp_path, allow_synthetic=False)

        reason = reason_of(loader)
        assert reason.startswith("synthetic-data model is not allowed in production")
        # 막힌 이유와 함께, 명시적으로 허용하는 방법도 알려 준다.
        assert "DEMAND_FORECAST_MODE=poc" in reason
        assert factory.loads == 0

    def test_production_accepts_a_real_data_model(self, tmp_path: Path) -> None:
        loader, _ = fake_loader(tmp_path, meta=metadata(synthetic_data=False), allow_synthetic=False)

        assert loader.get().model is not None

    def test_missing_runtime_is_reported(self, tmp_path: Path) -> None:
        model_path, metadata_path = write_artifact(tmp_path)

        def no_runtime(path: Path, features: object) -> FakePredictor:
            raise ArtifactError("model runtime not installed (install the demand-model extra)")

        loader = ModelArtifactLoader(
            model_path, metadata_path, buildable_features=BUILDABLE_FEATURES,
            allow_synthetic=True, load_predictor=no_runtime,
        )

        assert "model runtime not installed" in (reason_of(loader) or "")

    def test_without_the_model_runtime_the_default_loader_reports_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # demand-model extra가 없는 환경을 흉내 낸다. import joblib 이 실패한다.
        monkeypatch.setitem(__import__("sys").modules, "joblib", None)
        model_path, metadata_path = write_artifact(tmp_path)
        loader = ModelArtifactLoader(
            model_path, metadata_path, buildable_features=BUILDABLE_FEATURES, allow_synthetic=True
        )

        assert reason_of(loader) == "model runtime not installed (install the demand-model extra)"

    def test_reasons_never_carry_a_path(self, tmp_path: Path) -> None:
        cases = [
            ModelArtifactLoader(tmp_path / "a.joblib", tmp_path / "a.json",
                                buildable_features=BUILDABLE_FEATURES, allow_synthetic=True),
            fake_loader(tmp_path / "b", meta=metadata(model_sha256="0" * 64))[0],
            fake_loader(tmp_path / "c", allow_synthetic=False)[0],
        ]
        for loader in cases:
            assert str(tmp_path) not in (reason_of(loader) or "")


class TestVersionWarnings:
    def test_runtime_version_mismatch_is_warned(self, tmp_path: Path) -> None:
        model_path, metadata_path = write_artifact(
            tmp_path, metadata(versions={"sklearn": "1.8.0", "lightgbm": "4.7.0"})
        )
        loader = ModelArtifactLoader(
            model_path, metadata_path, buildable_features=BUILDABLE_FEATURES,
            allow_synthetic=True, load_predictor=CountingPredictorFactory(FakePredictor()),
            version_lookup=lambda package: {"scikit-learn": "1.9.0", "lightgbm": "4.7.0"}.get(package),
        )

        warnings = loader.get().model.warnings

        assert any("scikit-learn" in w and "1.8.0" in w and "1.9.0" in w for w in warnings)
        assert not any("lightgbm" in w for w in warnings)

    def test_missing_model_hash_is_warned_not_refused(self, tmp_path: Path) -> None:
        meta = metadata()
        del meta["model_sha256"]
        loader, _ = fake_loader(tmp_path, meta=meta)

        model = loader.get().model

        assert model is not None
        assert any("model_sha256" in w for w in model.warnings)


needs_real_artifact = pytest.mark.skipif(
    not (DEFAULT_DEMAND_FORECAST_MODEL_PATH.is_file() and DEFAULT_DEMAND_FORECAST_METADATA_PATH.is_file()),
    reason="저장된 artifact가 없다. research/demand_forecasting 에서 train.py 를 먼저 실행한다.",
)


@needs_real_artifact
class TestRealArtifact:
    def test_real_artifact_loads_with_the_production_loader(self) -> None:
        pytest.importorskip("sklearn")
        loader = ModelArtifactLoader(
            DEFAULT_DEMAND_FORECAST_MODEL_PATH,
            DEFAULT_DEMAND_FORECAST_METADATA_PATH,
            buildable_features=BUILDABLE_FEATURES,
            allow_synthetic=True,
        )

        state = loader.get()

        assert state.reason is None, state.reason
        assert state.model.metadata.synthetic_data is True
        assert state.model.metadata.target == "meal_count"
        # 학습 환경과 같은 버전으로 설치했다. 다르면 여기서 드러난다.
        assert state.model.warnings == ()

    def test_pipeline_column_names_are_checked_against_metadata(self) -> None:
        pytest.importorskip("sklearn")

        with pytest.raises(ArtifactError, match="feature schema mismatch"):
            load_joblib_pipeline(DEFAULT_DEMAND_FORECAST_MODEL_PATH, list(reversed(FEATURES)))
