"""torchvision COCO pretrained person detector.

**왜 torchvision인가.**

라이선스 때문이다. 지금 사람 검출에서 가장 흔한 선택은 Ultralytics YOLO
계열인데, Ultralytics는 코드와 pretrained weight 모두 AGPL-3.0이고 상용
제품에 쓰려면 별도 Enterprise 라이선스를 사야 한다(공식 저장소 명시).
밥플래너는 서비스로 나갈 제품이므로, 연구 단계에서 AGPL 모델로 숫자를
만들어 두면 나중에 그 숫자를 그대로 쓸 수 없다. 결과를 버리게 될 선택은
처음부터 하지 않는다.

torchvision은 BSD-3-Clause이고(공식 저장소 LICENSE 확인), 함께 배포되는
COCO pretrained weight도 같은 저장소에서 나온다. 상용 제약이 없다.

**성능은 양보한 것이 맞다.** torchvision이 보고하는 COCO box mAP는
fasterrcnn_mobilenet_v3_large_fpn 32.8, ssdlite320_mobilenet_v3_large 21.3,
fasterrcnn_resnet50_fpn_v2 46.7이다. 최신 YOLO 계열보다 낮다. 이것은 baseline을
낮게 잡은 대가이고, 보고서에 그대로 적는다. 나중에 Apache-2.0인 RT-DETR이나
YOLOX로 갈아끼울 수 있게 detector 경계를 따로 둔 이유가 이것이다.

**사람 클래스 번호를 코드에 박지 않는다.** weight 메타데이터의 categories에서
"person"의 위치를 찾는다. 모델을 바꿨을 때 조용히 다른 클래스를 세는 일이
가장 찾기 어려운 종류의 버그다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cctv_counting.detection import Detection
from cctv_counting.geometry import BoundingBox, GeometryError


class DetectorLoadError(RuntimeError):
    """detector를 준비할 수 없다."""


#: 확인하고 쓰는 모델만 허용한다. 임의의 이름을 getattr로 부르면 detection이
#: 아닌 것까지 불려 와서 엉뚱한 출력 모양으로 터진다.
SUPPORTED_MODELS: dict[str, str] = {
    "fasterrcnn_mobilenet_v3_large_fpn": "FasterRCNN_MobileNet_V3_Large_FPN_Weights",
    "fasterrcnn_mobilenet_v3_large_320_fpn": (
        "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights"
    ),
    "fasterrcnn_resnet50_fpn_v2": "FasterRCNN_ResNet50_FPN_V2_Weights",
    "ssdlite320_mobilenet_v3_large": "SSDLite320_MobileNet_V3_Large_Weights",
    "retinanet_resnet50_fpn_v2": "RetinaNet_ResNet50_FPN_V2_Weights",
    "fcos_resnet50_fpn": "FCOS_ResNet50_FPN_Weights",
}


def resolve_device(requested: str) -> str:
    """auto면 CUDA가 되는지 실제로 확인하고 고른다."""
    if requested not in {"auto", "cpu", "cuda"}:
        raise DetectorLoadError(f"알 수 없는 device입니다: {requested}")
    try:
        import torch
    except ModuleNotFoundError as error:
        raise DetectorLoadError(
            "torch가 필요합니다. pip install -e \".[cctv]\""
        ) from error

    if requested == "cpu":
        return "cpu"
    available = bool(torch.cuda.is_available())
    if requested == "cuda":
        if not available:
            raise DetectorLoadError("device=cuda로 요청했지만 CUDA를 쓸 수 없습니다.")
        return "cuda"
    return "cuda" if available else "cpu"


def person_class_id_from(categories: list[str | None]) -> int:
    """categories에서 person의 위치를 찾는다."""
    for index, name in enumerate(categories):
        if name == "person":
            return index
    raise DetectorLoadError(
        "이 모델의 클래스 목록에 person이 없습니다. 사람 검출에 쓸 수 없습니다."
    )


@dataclass
class TorchvisionPersonDetector:
    """torchvision detection 모델 하나를 감싼다.

    detect()는 BGR numpy 배열(OpenCV가 주는 그대로)을 받는다. RGB 변환과
    정규화는 여기서 한다. 호출하는 쪽이 색 순서를 신경 쓰게 하면 언젠가
    한 번은 틀린다.
    """

    model_name: str = "fasterrcnn_mobilenet_v3_large_fpn"
    device: str = "auto"
    #: 이 값 아래는 모델 출력 단계에서 바로 버린다. downstream 임계값과
    #: 별개로, 상자 수천 개를 들고 다니지 않기 위한 것이다.
    score_floor: float = 0.05
    max_input_size: int | None = None

    person_class_id: int = field(init=False, default=1)
    resolved_device: str = field(init=False, default="cpu")
    weights_name: str = field(init=False, default="")

    _model: Any = field(init=False, default=None, repr=False)
    _transform: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.model_name not in SUPPORTED_MODELS:
            raise DetectorLoadError(
                f"확인되지 않은 모델입니다: {self.model_name}. "
                f"쓸 수 있는 것: {', '.join(sorted(SUPPORTED_MODELS))}"
            )
        try:
            import torch  # noqa: F401
            from torchvision.models import detection as tv_detection
        except ModuleNotFoundError as error:
            raise DetectorLoadError(
                "torch와 torchvision이 필요합니다. pip install -e \".[cctv]\""
            ) from error

        self.resolved_device = resolve_device(self.device)

        weights_enum = getattr(tv_detection, SUPPORTED_MODELS[self.model_name])
        weights = weights_enum.DEFAULT
        self.weights_name = str(weights)

        categories = list(weights.meta.get("categories", []))
        if not categories:
            raise DetectorLoadError(
                f"{self.model_name}의 클래스 목록을 읽을 수 없습니다."
            )
        self.person_class_id = person_class_id_from(categories)

        builder = getattr(tv_detection, self.model_name)
        model = builder(weights=weights, box_score_thresh=self.score_floor)
        model.eval()
        model.to(self.resolved_device)
        self._model = model
        self._transform = weights.transforms()

    def describe(self) -> dict[str, Any]:
        """실험 metadata에 그대로 들어갈 기록."""
        return {
            "backend": "torchvision",
            "model": self.model_name,
            "weights": self.weights_name,
            "device": self.resolved_device,
            "personClassId": self.person_class_id,
            "maxInputSize": self.max_input_size,
        }

    def detect(self, frame: Any) -> list[Detection]:
        """BGR 배열 한 장에서 상자들을 뽑는다."""
        import torch

        height, width = frame.shape[:2]
        scale = 1.0
        working = frame
        if self.max_input_size is not None:
            longest = max(height, width)
            if longest > self.max_input_size:
                import cv2

                scale = self.max_input_size / longest
                working = cv2.resize(
                    frame,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_LINEAR,
                )

        rgb = working[:, :, ::-1].copy()
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        batch = self._transform(tensor).to(self.resolved_device)

        with torch.inference_mode():
            outputs = self._model([batch])[0]

        boxes = outputs["boxes"].detach().cpu().tolist()
        scores = outputs["scores"].detach().cpu().tolist()
        labels = outputs["labels"].detach().cpu().tolist()

        detections: list[Detection] = []
        for box, score, label in zip(boxes, scores, labels):
            if int(label) != self.person_class_id:
                continue
            x1, y1, x2, y2 = (value / scale for value in box)
            try:
                bbox = BoundingBox(x1, y1, x2, y2)
            except GeometryError:
                # 모델이 아주 가끔 폭이 0인 상자를 낸다. 세지 않으면 그만이다.
                continue
            detections.append(
                Detection(
                    bbox=bbox,
                    confidence=min(1.0, max(0.0, float(score))),
                    class_id=int(label),
                )
            )
        return detections
