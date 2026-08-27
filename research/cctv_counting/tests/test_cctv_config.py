"""실험 설정."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cctv_counting.config import (
    CctvConfig,
    ConfigError,
    Device,
    LineConfig,
    RoiConfig,
    TrackerConfig,
    config_to_dict,
    load_config,
)
from cctv_counting.crossing import InDirection
from cctv_counting.detection import Anchor

MINIMAL = {"name": "test"}


def test_defaults_are_a_tracking_baseline():
    config = CctvConfig.model_validate(MINIMAL)
    assert config.tracker.enabled
    assert config.line.enabled
    assert config.counts_entries
    assert config.detector.device is Device.AUTO


def test_unknown_key_is_rejected():
    """설정 오타가 조용히 무시되면 돌린 실험이 무엇이었는지 알 수 없다."""
    with pytest.raises(ValidationError):
        CctvConfig.model_validate({"trakcer": {"enabled": True}})


def test_unknown_nested_key_is_rejected():
    with pytest.raises(ValidationError):
        CctvConfig.model_validate({"tracker": {"enabled": True, "kalman": True}})


def test_confidence_threshold_out_of_range_is_rejected():
    with pytest.raises(ValidationError):
        CctvConfig.model_validate({"detector": {"confidence_threshold": 1.5}})


def test_tracker_thresholds_must_be_ordered():
    with pytest.raises(ValidationError):
        TrackerConfig(high_threshold=0.3, low_threshold=0.8)


def test_roi_must_not_be_inverted():
    with pytest.raises(ValidationError):
        RoiConfig(enabled=True, x1=0.8, y1=0.1, x2=0.2, y2=0.9)


def test_disabled_roi_is_not_validated_for_shape():
    """끈 ROI의 좌표까지 따지면 기본값을 남겨 두기 어렵다."""
    assert RoiConfig(enabled=False, x1=0.8, x2=0.2).to_rect() is None


def test_roi_converts_to_a_normalized_rect():
    rect = RoiConfig(enabled=True, x1=0.1, y1=0.2, x2=0.9, y2=0.95).to_rect()
    assert rect is not None
    assert rect.x1 == pytest.approx(0.1)


def test_line_coordinates_must_be_normalized():
    with pytest.raises(ValidationError):
        LineConfig(start=(10.0, 600.0), end=(900.0, 600.0))


def test_line_endpoints_must_differ():
    with pytest.raises(ValidationError):
        LineConfig(start=(0.5, 0.5), end=(0.5, 0.5))


def test_confirm_distance_must_not_be_below_band():
    with pytest.raises(ValidationError):
        LineConfig(band=0.05, min_confirm_distance=0.01)


def test_detection_only_config_cannot_count_entries():
    """추적을 끄면 입장 인원은 잴 수 없다. 금지가 아니라 baseline A다."""
    config = CctvConfig.model_validate({"tracker": {"enabled": False}})
    assert config.line.enabled
    assert not config.counts_entries


def test_line_defaults_to_bottom_anchor():
    assert LineConfig().anchor is Anchor.BOTTOM_CENTER
    assert LineConfig().in_direction is InDirection.NEGATIVE


def test_config_is_frozen():
    config = CctvConfig.model_validate(MINIMAL)
    with pytest.raises(ValidationError):
        config.seed = 7


def test_config_dict_is_json_serializable():
    """실험 metadata에 그대로 실려야 한다."""
    payload = config_to_dict(CctvConfig.model_validate(MINIMAL))
    assert json.loads(json.dumps(payload))["name"] == "test"


def test_load_json_config(tmp_path):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps({"name": "from-json", "seed": 3}), encoding="utf-8")
    assert load_config(path).seed == 3


def test_missing_config_file_is_reported(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_unsupported_config_format_is_reported(tmp_path):
    path = tmp_path / "experiment.ini"
    path.write_text("name = x", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_non_mapping_config_is_reported(tmp_path):
    path = tmp_path / "experiment.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_shipped_yaml_configs_are_valid():
    """저장소에 든 설정이 깨져 있으면 시연 당일에 안다."""
    yaml = pytest.importorskip("yaml", reason="pyyaml이 없으면 YAML 설정을 못 읽는다")
    from pathlib import Path

    configs = Path(__file__).resolve().parents[1] / "configs"
    files = sorted(configs.glob("*.yaml"))
    assert files, "설정 파일이 하나도 없다"
    for path in files:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        CctvConfig.model_validate(payload)


def test_shipped_detection_baseline_cannot_count_entries():
    yaml = pytest.importorskip("yaml")
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs" / "baseline_detection.yaml"
    config = CctvConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert not config.counts_entries


def test_shipped_baselines_share_detector_settings():
    """detector가 다르면 두 baseline의 차이를 추적 탓으로 돌릴 수 없다."""
    yaml = pytest.importorskip("yaml")
    from pathlib import Path

    configs = Path(__file__).resolve().parents[1] / "configs"
    detection = CctvConfig.model_validate(
        yaml.safe_load((configs / "baseline_detection.yaml").read_text(encoding="utf-8"))
    )
    tracking = CctvConfig.model_validate(
        yaml.safe_load((configs / "baseline_tracking.yaml").read_text(encoding="utf-8"))
    )
    assert detection.detector == tracking.detector
    assert detection.line == tracking.line
    assert detection.roi == tracking.roi
