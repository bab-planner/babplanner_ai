"""실험 설정.

임계값과 좌표를 코드에 박지 않는다. 카메라가 바뀌면 선도 ROI도 바뀌고,
"신뢰도 0.4가 제일 낫다" 같은 것은 재 보기 전에는 알 수 없다. 설정 파일을
바꿔가며 돌린 것이 곧 실험 기록이 되도록 한다.

pydantic으로 검증한다. 저장소가 이미 pydantic을 쓰고 있고, 잘못된 설정은
영상을 다 돌린 다음이 아니라 시작하기 전에 터져야 한다.

YAML을 읽으려면 pyyaml이 필요하다(`pip install -e ".[cctv]"`). 설정 모델
자체는 dict에서 바로 만들 수 있으므로 core 테스트는 pyyaml 없이도 돈다.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cctv_counting.crossing import InDirection
from cctv_counting.detection import Anchor
from cctv_counting.geometry import Line, NormalizedPoint, NormalizedRect


class ConfigError(ValueError):
    """설정 파일을 읽을 수 없다."""


class Device(str, Enum):
    """어디서 추론할 것인가."""

    #: CUDA가 되면 GPU, 안 되면 CPU.
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class DetectorBackend(str, Enum):
    """어떤 detector 구현을 쓸 것인가.

    지금은 하나뿐이다. 값이 하나여도 열거형으로 두는 이유는, 설정 파일에
    무엇을 썼는지가 실험 기록에 남아야 하기 때문이다.
    """

    TORCHVISION = "torchvision"


class CountingMethod(str, Enum):
    """occupancy를 무엇으로 셀 것인가.

    A3에서 추가했다. A2까지는 tracker.enabled 하나로 두 가지(A, C)만
    골랐는데, 그러면 "추적의 이득이 연결에서 왔는가 확인에서 왔는가"를
    분해할 수 없다. 가운데 항을 명시적으로 둔다.
    """

    #: A. 프레임별 상자 수. 시간 정보를 전혀 쓰지 않는다.
    DETECTION_ONLY = "detection_only"

    #: B. 직전 프레임과의 겹침으로 연속 관측만 확인한다. 연결도 신원도 없다.
    TEMPORAL_CONFIRMATION = "temporal_confirmation"

    #: C. 추적. 이것만 입장/퇴장을 셀 수 있다.
    TRACKING = "tracking"


class Strict(BaseModel):
    """오타를 막는다. 모르는 키가 오면 터진다."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DetectorConfig(Strict):
    backend: DetectorBackend = DetectorBackend.TORCHVISION

    #: torchvision.models.detection 의 builder 이름.
    model: str = "fasterrcnn_mobilenet_v3_large_fpn"

    #: 이 아래 상자는 버린다. 실험 축의 하나다.
    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    device: Device = Device.AUTO

    #: None이면 weight 메타데이터에서 사람 클래스 번호를 찾는다. 번호를
    #: 추측해서 박아 두면 모델을 바꿨을 때 엉뚱한 클래스를 센다.
    person_class_id: int | None = None

    #: 긴 변을 이 길이로 줄여서 넣는다. None이면 원본 그대로.
    max_input_size: int | None = Field(default=None, ge=64)


class ConfirmationConfig(Strict):
    """B baseline 설정. counting_method가 temporal_confirmation일 때만 쓴다."""

    #: 이만큼 연속으로 보인 상자만 센다. 1이면 detection-only와 같아진다.
    min_observations: int = Field(default=3, ge=1)
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)


class TrackerConfig(Strict):
    #: False면 detection-only baseline이다. 입장 인원은 셀 수 없다.
    enabled: bool = True
    high_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    low_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    low_iou_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    max_age: int = Field(default=30, ge=1)
    min_hits: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> TrackerConfig:
        if self.low_threshold > self.high_threshold:
            raise ValueError(
                "low_threshold가 high_threshold보다 클 수 없습니다: "
                f"{self.low_threshold} > {self.high_threshold}"
            )
        return self


class RoiConfig(Strict):
    """관심 영역. 0~1 정규화 좌표다."""

    enabled: bool = False
    x1: float = Field(default=0.0, ge=0.0, le=1.0)
    y1: float = Field(default=0.0, ge=0.0, le=1.0)
    x2: float = Field(default=1.0, ge=0.0, le=1.0)
    y2: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> RoiConfig:
        if self.enabled and (self.x2 <= self.x1 or self.y2 <= self.y1):
            raise ValueError(
                f"ROI가 뒤집혔습니다: ({self.x1}, {self.y1}) ~ ({self.x2}, {self.y2})"
            )
        return self

    def to_rect(self) -> NormalizedRect | None:
        if not self.enabled:
            return None
        return NormalizedRect(self.x1, self.y1, self.x2, self.y2)


class LineConfig(Strict):
    """입구 가상 선. 0~1 정규화 좌표다."""

    enabled: bool = True
    start: tuple[float, float] = (0.10, 0.60)
    end: tuple[float, float] = (0.90, 0.60)

    #: 선의 어느 쪽으로 넘어가는 것을 입장으로 볼 것인가. 카메라 설치
    #: 방향에 달렸으므로 반드시 현장에서 확인하고 정한다.
    in_direction: InDirection = InDirection.NEGATIVE

    #: 상자의 어느 점으로 판정할 것인가. 바닥에 그은 선이면 발쪽이 맞다.
    anchor: Anchor = Anchor.BOTTOM_CENTER

    band: float = Field(default=0.01, ge=0.0, le=1.0)
    min_confirm_distance: float = Field(default=0.03, ge=0.0, le=1.0)
    #: 사건을 낸 뒤 이만큼은 같은 track을 다시 세지 않는다. 프레임 수가
    #: 아니라 초다. 프레임 수로 두면 같은 설정 파일이 10fps 영상과 60fps
    #: 영상에서 서로 다른 정책이 된다.
    cooldown_seconds: float = Field(default=0.5, ge=0.0)

    @model_validator(mode="after")
    def _valid(self) -> LineConfig:
        if not self.enabled:
            return self
        for name in ("start", "end"):
            for value in getattr(self, name):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name}은(는) 0~1 정규화 좌표여야 합니다: {value}")
        if self.start == self.end:
            raise ValueError("counting line의 시작점과 끝점이 같습니다.")
        if self.min_confirm_distance < self.band:
            raise ValueError(
                "min_confirm_distance는 band 이상이어야 합니다: "
                f"{self.min_confirm_distance} < {self.band}"
            )
        return self

    def to_line(self) -> Line | None:
        if not self.enabled:
            return None
        return Line(
            NormalizedPoint(self.start[0], self.start[1]),
            NormalizedPoint(self.end[0], self.end[1]),
        )


class ReliabilityConfig(Strict):
    """detector 신뢰도만 본다.

    장면이 얼마나 붐비는지는 여기 없다. 혼잡도와 신뢰도는 다른 축이고,
    백엔드의 CROWDED 등급과 이름이 겹치며, 몇 명부터 붐비는지는 좌석 수에
    달렸는데 그 임계값이 아직 없다. 조건 태그는 평가 metadata 쪽에서 다룬다.
    """

    min_mean_confidence: float = Field(default=0.45, ge=0.0, le=1.0)


class WindowConfig(Strict):
    """부가 출력인 시간 창 통계."""

    enabled: bool = False
    seconds: float = Field(default=3.0, gt=0.0)


class OutputConfig(Strict):
    annotated_video: bool = True
    observations_csv: bool = True
    summary_json: bool = True
    windows_csv: bool = False


class CctvConfig(Strict):
    """실험 하나의 전체 설정."""

    name: str = "unnamed"
    description: str = ""
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    roi: RoiConfig = Field(default_factory=RoiConfig)
    line: LineConfig = Field(default_factory=LineConfig)
    confirmation: ConfirmationConfig = Field(default_factory=ConfirmationConfig)
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    window: WindowConfig = Field(default_factory=WindowConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    #: 지금 pipeline에 난수는 없다. 그래도 기록에 남겨 둔다. 나중에 무엇이
    #: 들어와도 이 값이 실험 metadata에 이미 있다.
    seed: int = 0

    #: 매 N번째 프레임만 처리한다. 1이면 전부.
    frame_stride: int = Field(default=1, ge=1)

    @property
    def counting_method(self) -> CountingMethod:
        """이 설정이 실제로 쓰는 계수 방법.

        tracker.enabled와 confirmation을 따로 두는 대신 여기서 하나로
        읽는다. 실험 기록에 "무엇으로 셌는가"가 한 값으로 남아야 한다.
        """
        if self.tracker.enabled:
            return CountingMethod.TRACKING
        if self.confirmation.min_observations > 1:
            return CountingMethod.TEMPORAL_CONFIRMATION
        return CountingMethod.DETECTION_ONLY

    @property
    def counts_entries(self) -> bool:
        """이 설정이 입장 인원을 실제로 셀 수 있는가.

        선만 그어 두고 추적을 끄면 셀 수 없다. 그 조합은 금지가 아니라
        baseline A다. 같은 선을 놓고 A와 B를 비교해야 추적이 무엇을
        바꾸는지 보이기 때문에 허용한다. 다만 그 경우 entries/exits는
        0이 아니라 None으로 나간다.
        """
        return self.line.enabled and self.tracker.enabled


def load_config(path: Path) -> CctvConfig:
    """설정 파일을 읽는다. .yaml/.yml 또는 .json."""
    if not path.exists():
        raise ConfigError(f"설정 파일이 없습니다: {path}")

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as error:  # pragma: no cover - 환경 의존
            raise ConfigError(
                "YAML 설정을 읽으려면 pyyaml이 필요합니다. "
                'pip install -e ".[cctv]"'
            ) from error
        payload = yaml.safe_load(text)
    else:
        raise ConfigError(f"지원하지 않는 설정 형식입니다: {path.suffix}")

    if not isinstance(payload, dict):
        raise ConfigError(f"설정 파일의 최상위가 매핑이 아닙니다: {path}")
    return CctvConfig.model_validate(payload)


def config_to_dict(config: CctvConfig) -> dict[str, Any]:
    """실험 metadata에 그대로 실을 수 있는 형태."""
    return config.model_dump(mode="json")
