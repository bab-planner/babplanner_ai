"""이미지에서 특징을 뽑는다.

두 갈래를 같은 자리에 둔다.

    B1  단순 이미지 통계      밝기·채도·색 히스토그램·경계 밀도
    B2  frozen 사전학습 특징  torchvision 경량 backbone의 임베딩

**전처리는 결정적이다.** random crop이나 flip을 쓰지 않는다. 이유가 둘이다.

1. frozen feature baseline에서는 증강이 할 일이 거의 없다. 특징 추출기를
   학습시키지 않기 때문이다.
2. **before/after 짝 실험에서 두 이미지에 서로 다른 무작위 변환을 걸면
   비교 자체가 깨진다.** 같은 식사의 전후를 보는 것이 목적인데 한쪽만
   뒤집히면 그 관계가 사라진다.

**무거운 import를 함수 안에 둔다.** 이 모듈을 import하는 것만으로 torch가
필요하지 않게 해서, 지표·라벨 테스트가 `.venv`에서 그대로 돈다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: B1이 내는 특징 이름. 순서가 곧 벡터의 순서다.
SIMPLE_FEATURE_NAMES = (
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "mean_saturation",
    "std_saturation",
    "mean_value",
    "std_value",
    "dark_fraction",
    "edge_density",
)

#: 단순 통계를 낼 때 줄이는 크기. 작게 줄여도 색 통계는 거의 보존된다.
SIMPLE_RESIZE = (128, 128)

#: "어둡다"고 볼 밝기 문턱. 0~1 기준. 접시 바깥의 그늘과 빈 칸을 잡는다.
DARK_THRESHOLD = 0.25


class FeatureError(RuntimeError):
    """특징을 뽑을 수 없다."""


def _pillow() -> Any:
    try:
        from PIL import Image
    except ModuleNotFoundError as error:  # pragma: no cover - 환경 의존
        raise FeatureError("Pillow가 필요합니다.") from error
    return Image


def simple_image_features(path: Path) -> list[float]:
    """B1. 손으로 고른 12개 통계.

    **더 늘리지 않는다.** 특징을 계속 더하면 그것 자체가 모델 설계가 되고,
    "단순 baseline"이라는 자리를 잃는다. 여기서 알고 싶은 것은 색과 밝기만으로
    어디까지 되는가이지 특징 공학의 상한이 아니다.
    """
    import numpy as np

    Image = _pillow()
    with Image.open(path) as handle:
        image = handle.convert("RGB").resize(SIMPLE_RESIZE)
        array = np.asarray(image, dtype=np.float32) / 255.0

    red, green, blue = array[..., 0], array[..., 1], array[..., 2]
    maximum = array.max(axis=2)
    minimum = array.min(axis=2)
    # HSV의 채도와 명도. 색 공간 변환 라이브러리를 부르지 않고 정의대로 쓴다.
    saturation = np.where(maximum > 0, (maximum - minimum) / np.maximum(maximum, 1e-6), 0.0)
    value = maximum

    grey = array.mean(axis=2)
    dy = np.abs(np.diff(grey, axis=0)).mean()
    dx = np.abs(np.diff(grey, axis=1)).mean()

    return [
        float(red.mean()),
        float(green.mean()),
        float(blue.mean()),
        float(red.std()),
        float(green.std()),
        float(blue.std()),
        float(saturation.mean()),
        float(saturation.std()),
        float(value.mean()),
        float(value.std()),
        float((value < DARK_THRESHOLD).mean()),
        float((dx + dy) / 2.0),
    ]


#: 조사한 backbone 후보. 둘 다 torchvision 배포 가중치다.
SUPPORTED_BACKBONES = {
    "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
    "resnet18": "ResNet18_Weights",
}


@dataclass
class FrozenBackbone:
    """사전학습 backbone을 특징 추출기로만 쓴다.

    **학습하지 않는다.** W2는 frozen feature baseline까지다. fine-tuning은
    baseline을 확보한 뒤의 후보이고, 먼저 하면 무엇 덕분에 좋아졌는지 알 수
    없다.

    가중치의 조건은 소프트웨어 라이선스와 다르다. torchvision 자체는
    BSD-3-Clause이고, 배포 가중치는 ImageNet으로 학습된 것이다. **연구·평가
    용도로 쓰며, 상용 배포 적격이라고 말하지 않는다.**
    """

    name: str = "mobilenet_v3_small"
    device: str = "cpu"

    _model: Any = None
    _transform: Any = None

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_BACKBONES:
            raise FeatureError(
                f"확인하고 쓰는 backbone만 허용합니다: {sorted(SUPPORTED_BACKBONES)}"
            )

    def load(self) -> "FrozenBackbone":
        import torch
        import torchvision

        weights_enum = getattr(torchvision.models, SUPPORTED_BACKBONES[self.name])
        weights = weights_enum.DEFAULT
        model = getattr(torchvision.models, self.name)(weights=weights)
        # 분류 머리를 떼고 임베딩만 남긴다. resnet 계열은 `fc`, mobilenet
        # 계열은 `classifier`라 이름이 다르다.
        if hasattr(model, "fc"):
            model.fc = torch.nn.Identity()
        if hasattr(model, "classifier"):
            model.classifier = torch.nn.Identity()
        model.eval()
        model.to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self._model = model
        # 가중치가 함께 배포하는 전처리를 그대로 쓴다. 우리가 다시 정하지
        # 않는다 — 사전학습 분포와 어긋나면 특징이 망가진다.
        self._transform = weights.transforms()
        return self

    @property
    def weights_name(self) -> str:
        import torchvision

        return getattr(torchvision.models, SUPPORTED_BACKBONES[self.name]).DEFAULT.name

    def embed(self, paths: list[Path], batch_size: int = 32) -> Any:
        """이미지 목록의 임베딩. (n, d) 배열."""
        import numpy as np
        import torch

        if self._model is None:
            raise FeatureError("load()를 먼저 부르세요.")
        Image = _pillow()

        outputs: list[Any] = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            tensors = []
            for path in chunk:
                with Image.open(path) as handle:
                    tensors.append(self._transform(handle.convert("RGB")))
            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                features = self._model(batch)
            outputs.append(features.flatten(1).cpu().numpy())
        return np.concatenate(outputs, axis=0)


def paired_features(before: Any, after: Any, mode: str) -> Any:
    """before/after 임베딩을 하나의 특징으로 합친다.

    **Siamese 네트워크를 만들지 않는다.** 학습하는 부분이 없으므로 결합은
    벡터 연산으로 충분하고, 복잡한 구조를 먼저 넣으면 개선이 정보 때문인지
    구조 때문인지 갈라 볼 수 없다.

        after           식후만. 운영 API와 같은 입력
        concat          [before, after]
        concat_diff     [before, after, after - before]
        diff            after - before

    같은 backbone, 같은 전처리로 뽑은 임베딩이어야 한다. 그래야 두 갈래의
    차이가 **정보량 차이**만 남는다.
    """
    import numpy as np

    if before.shape != after.shape:
        raise FeatureError(
            f"before/after 임베딩 모양이 다릅니다: {before.shape} vs {after.shape}"
        )
    if mode == "after":
        return after
    if mode == "concat":
        return np.concatenate([before, after], axis=1)
    if mode == "concat_diff":
        return np.concatenate([before, after, after - before], axis=1)
    if mode == "diff":
        return after - before
    raise FeatureError(f"알 수 없는 결합 방식입니다: {mode}")
