"""torchvision 어댑터에서 무거운 의존성 없이 검증할 수 있는 부분.

모델을 실제로 내려받지 않고도 확인해야 하는 것이 둘 있다. 사람 클래스 번호를
추측하지 않는 것, 그리고 허용 목록에 없는 모델 이름을 그냥 부르지 않는 것이다.
"""

from __future__ import annotations

import pytest

from cctv_counting.detectors.torchvision_detector import (
    SUPPORTED_MODELS,
    DetectorLoadError,
    TorchvisionPersonDetector,
    person_class_id_from,
)


def test_person_index_comes_from_the_weight_metadata():
    """COCO 계열에서 person은 보통 1이지만, 그것을 코드에 박지 않는다."""
    categories = ["__background__", "person", "bicycle", "car"]
    assert person_class_id_from(categories) == 1


def test_person_index_follows_a_different_layout():
    """모델을 바꿨을 때 조용히 다른 클래스를 세는 것이 가장 찾기 어려운 버그다."""
    assert person_class_id_from(["car", "dog", "person"]) == 2


def test_model_without_person_is_refused():
    with pytest.raises(DetectorLoadError):
        person_class_id_from(["car", "dog", "bicycle"])


def test_supported_models_are_an_allowlist():
    """임의의 이름을 getattr로 부르면 detection이 아닌 것까지 불려 온다."""
    assert "fasterrcnn_mobilenet_v3_large_fpn" in SUPPORTED_MODELS
    assert "ssdlite320_mobilenet_v3_large" in SUPPORTED_MODELS
    assert all(name.islower() for name in SUPPORTED_MODELS)


def test_unknown_model_is_refused_before_anything_is_downloaded():
    with pytest.raises(DetectorLoadError) as error:
        TorchvisionPersonDetector(model_name="yolov8n")
    assert "yolov8n" in str(error.value)


def test_every_supported_model_names_a_weights_enum():
    for model, weights in SUPPORTED_MODELS.items():
        assert weights.endswith("_Weights"), model
