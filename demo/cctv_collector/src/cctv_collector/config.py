"""수집기 설정.

두 종류가 있고, 둘을 섞지 않는다.

    카메라 설정   YAML 파일. site_id, device_id, ROI, 선, 표본 간격, 임계값.
                  비밀이 아니다. 저장소에 예시를 둔다.

    접속 설정     환경 변수. 백엔드 주소와 내부 서비스 토큰.
                  **비밀이다. 파일에도 소스에도 두지 않는다.**

**연구 임계값을 여기에 옮겨 적지 않는다.** occupancy 임계값은 A6가 동결한
기록(`final_test_freeze.json`)에서 읽는다. 복사해 두면 언젠가 한쪽만
바뀌고, 그때 시연에 나가는 숫자는 연구가 검증한 숫자가 아니게 된다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cctv_counting.crossing import InDirection
from cctv_counting.detection import Anchor
from cctv_counting.dualpath import SceneGeometry, TaskPolicy, TrackerKind
from cctv_counting.geometry import Line, NormalizedPoint, NormalizedRect

from cctv_collector.congestion_policy import Calibration, CountThresholdPolicy


class CollectorConfigError(ValueError):
    """설정이 성립하지 않는다.

    메시지에 토큰 값을 절대 싣지 않는다. 설정 오류 로그가 비밀을 흘리는
    가장 흔한 경로다.
    """


#: 백엔드 주소. 기본값을 두지 않는다. production hostname을 코드에 박으면
#: 실수로 시연 데이터가 운영 DB에 들어간다.
BACKEND_BASE_URL_ENV = "BAPPLANNER_BACKEND_BASE_URL"

#: `/internal/v1/**` 이 받는 토큰. Spring이 **받는** 쪽이고,
#: FastAPI가 받는 `AI_SERVICE_TOKEN`과는 다른 비밀 값이다.
INTERNAL_SERVICE_TOKEN_ENV = "BAPPLANNER_INTERNAL_SERVICE_TOKEN"

#: 백엔드 `InternalServiceProperties.MIN_TOKEN_LENGTH`와 같은 값.
MIN_SERVICE_TOKEN_LENGTH = 32

#: A6가 동결한 기록. 채택된 정책 이름과 임계값이 여기 있다.
RESEARCH_FREEZE_PATH = Path(
    "research/cctv_counting/experiments/a6_continuity/final_test_freeze.json"
)

#: 연구를 과장하지 않는 이름. "production"이나 "validated"를 쓰지 않는다.
DEFAULT_MODEL_VERSION = "cctv-fasterrcnn-mobilenetv3-demo-v1"

#: 1초에 한 번. 영상이 25fps라고 25번 POST하지 않는다. 혼잡도는 초 단위로
#: 바뀌는 값이 아니고, 백엔드는 관측마다 두 테이블에 쓴다.
DEFAULT_OBSERVATION_INTERVAL_SECONDS = 1.0

AGGREGATIONS = ("latest", "median")


@dataclass(frozen=True)
class BackendSettings:
    """백엔드 접속 설정. **환경 변수에서만 온다.**"""

    base_url: str
    #: repr에서 뺀다. 예외 메시지나 로그에 객체가 실려도 토큰이 나오지 않는다.
    token: str

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise CollectorConfigError(f"{BACKEND_BASE_URL_ENV} 이(가) 비어 있습니다.")
        if not self.base_url.startswith(("http://", "https://")):
            raise CollectorConfigError(
                f"{BACKEND_BASE_URL_ENV} 은(는) http:// 또는 https:// 로 "
                "시작해야 합니다."
            )
        if len(self.token.strip()) < MIN_SERVICE_TOKEN_LENGTH:
            # 값도 길이도 적지 않는다. 길이를 적으면 그것도 단서가 된다.
            raise CollectorConfigError(
                f"{INTERNAL_SERVICE_TOKEN_ENV} 은(는) 최소 "
                f"{MIN_SERVICE_TOKEN_LENGTH}자여야 합니다."
            )

    def __repr__(self) -> str:
        return f"BackendSettings(base_url={self.base_url!r}, token=<redacted>)"


def load_backend_settings(
    environ: dict[str, str] | None = None,
    base_url_override: str | None = None,
) -> BackendSettings:
    """환경 변수에서 접속 설정을 읽는다.

    base_url만 CLI로 덮어쓸 수 있다. **토큰은 CLI 인자로 받지 않는다.**
    명령줄 인자는 프로세스 목록과 셸 히스토리에 남는다.
    """
    env = os.environ if environ is None else environ
    base_url = base_url_override or env.get(BACKEND_BASE_URL_ENV, "")
    if not base_url:
        raise CollectorConfigError(
            f"{BACKEND_BASE_URL_ENV} 이(가) 없습니다. --backend-url 로 줄 수도 "
            "있습니다."
        )
    token = env.get(INTERNAL_SERVICE_TOKEN_ENV, "")
    if not token:
        raise CollectorConfigError(
            f"{INTERNAL_SERVICE_TOKEN_ENV} 이(가) 없습니다. 환경 변수로만 "
            "주입합니다."
        )
    return BackendSettings(base_url=base_url.rstrip("/"), token=token)


@dataclass(frozen=True)
class ObservationSettings:
    """얼마나 자주, 무엇으로 대표값을 낼 것인가."""

    interval_seconds: float = DEFAULT_OBSERVATION_INTERVAL_SECONDS
    #: latest = 창의 마지막 표본. median = 창 표본의 중앙값.
    #: 둘 다 연구 결과를 변형하지 않는 단순 정책이다. 결과를 좋아 보이게
    #: 하는 적응형 평활은 두지 않는다.
    aggregation: str = "latest"

    def __post_init__(self) -> None:
        if not self.interval_seconds > 0:
            raise CollectorConfigError(
                f"interval_seconds는 0보다 커야 합니다: {self.interval_seconds}"
            )
        if self.aggregation not in AGGREGATIONS:
            raise CollectorConfigError(
                f"aggregation은 {AGGREGATIONS} 중 하나여야 합니다: "
                f"{self.aggregation}"
            )


@dataclass(frozen=True)
class CameraConfig:
    """카메라 한 대에 대한 전부.

    **ROI와 선은 카메라마다 다르다.** A6에서 확인한 것이 이것이다 — 카메라를
    바꿨는데 ROI를 그대로 쓰면 통과 사건의 23%가 계수 영역 밖에서 일어난다.
    그래서 이 값들에 기본값을 두지 않고 설정 파일에서 반드시 받는다.
    """

    site_id: int
    device_id: str
    frame_width: int
    frame_height: int
    roi: NormalizedRect | None
    line: Line
    in_direction: InDirection
    anchor: Anchor
    band: float
    min_confirm_distance: float
    cooldown_seconds: float
    observation: ObservationSettings
    congestion_policy: CountThresholdPolicy | None
    model_version: str | None

    def scene(self) -> SceneGeometry:
        return SceneGeometry(
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            line=self.line,
            in_direction=self.in_direction,
            roi=self.roi,
            anchor=self.anchor,
            band=self.band,
            min_confirm_distance=self.min_confirm_distance,
            cooldown_seconds=self.cooldown_seconds,
        )

    def describe(self) -> dict[str, Any]:
        """기록에 남길 형태. 비밀 값이 없다."""
        return {
            "siteId": self.site_id,
            "deviceId": self.device_id,
            "frameSize": [self.frame_width, self.frame_height],
            "roi": None
            if self.roi is None
            else [self.roi.x1, self.roi.y1, self.roi.x2, self.roi.y2],
            "line": {
                "start": [self.line.start.x, self.line.start.y],
                "end": [self.line.end.x, self.line.end.y],
                "inDirection": self.in_direction.value,
                "anchor": self.anchor.value,
            },
            "observation": {
                "intervalSeconds": self.observation.interval_seconds,
                "aggregation": self.observation.aggregation,
            },
            "congestionPolicy": None
            if self.congestion_policy is None
            else self.congestion_policy.describe(),
            "modelVersion": self.model_version,
        }


def adopted_research_policy(
    freeze_path: Path | None = None,
) -> tuple[TaskPolicy, str]:
    """A6가 **채택한** 정책을 동결 기록에서 읽는다.

    복사본을 두지 않는 이유가 있다. A6는 개선 tracker(`P2_motion`)를 만들고도
    동결한 주 지표가 개선되지 않아 **채택하지 않았다.** 기본값은 여전히
    `P0_baseline`이다. 그 판단이 기록 파일 안에 있으므로, 여기서 그 파일을
    읽으면 시연이 연구 결론과 어긋날 수 없다.

    (정책 이름, TaskPolicy)를 돌려준다.
    """
    path = RESEARCH_FREEZE_PATH if freeze_path is None else freeze_path
    if not path.exists():
        raise CollectorConfigError(
            f"연구 동결 기록이 없습니다: {path}. 저장소 루트에서 실행하세요."
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    name = record.get("baseline")
    if not name:
        raise CollectorConfigError(f"{path}에 baseline 정책 이름이 없습니다.")
    raw = record.get("policies", {}).get(name)
    if raw is None:
        raise CollectorConfigError(f"{path}에 정책 {name}이(가) 없습니다.")

    policy = TaskPolicy(
        occupancy_threshold=raw["occupancyThreshold"],
        tracker_high_threshold=raw["trackerHighThreshold"],
        tracker_low_threshold=raw["trackerLowThreshold"],
        tracker=TrackerKind(raw["tracker"]),
        iou_threshold=raw["iouThreshold"],
        low_iou_threshold=raw["lowIouThreshold"],
        max_age_frames=raw["maxAgeFrames"],
        max_age_seconds=raw["maxAgeSeconds"],
        min_hits=raw["minHits"],
        velocity_smoothing=raw["velocitySmoothing"],
    )
    return policy, name


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise CollectorConfigError(f"{where}에 {key}가 없습니다.")
    return mapping[key]


def _unit(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CollectorConfigError(f"{name}은(는) 숫자여야 합니다: {value!r}")
    if not 0.0 <= float(value) <= 1.0:
        raise CollectorConfigError(f"{name}은(는) 0~1 정규화 좌표여야 합니다: {value}")
    return float(value)


def load_camera_config(path: Path) -> CameraConfig:
    """카메라 설정 YAML(또는 JSON)을 읽는다."""
    if not path.exists():
        raise CollectorConfigError(f"카메라 설정 파일이 없습니다: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as error:  # pragma: no cover - 환경 의존
            raise CollectorConfigError(
                'YAML 설정을 읽으려면 pyyaml이 필요합니다. pip install -e ".[cctv]"'
            ) from error
        payload = yaml.safe_load(text)
    else:
        raise CollectorConfigError(f"지원하지 않는 설정 형식입니다: {path.suffix}")
    if not isinstance(payload, dict):
        raise CollectorConfigError(f"설정 파일의 최상위가 매핑이 아닙니다: {path}")
    return camera_config_from_dict(payload, source=str(path))


def camera_config_from_dict(
    payload: dict[str, Any], source: str = "설정"
) -> CameraConfig:
    """이미 읽어 온 매핑에서 설정을 만든다. 테스트가 이쪽을 쓴다."""
    for forbidden in ("token", "service_token", "serviceToken", "password", "secret"):
        if forbidden in payload:
            raise CollectorConfigError(
                f"{source}에 비밀 값처럼 보이는 키가 있습니다: {forbidden}. "
                "토큰은 환경 변수로만 줍니다."
            )

    site_id = _require(payload, "site_id", source)
    if isinstance(site_id, bool) or not isinstance(site_id, int) or site_id <= 0:
        raise CollectorConfigError(f"site_id는 양의 정수여야 합니다: {site_id!r}")

    device_id = _require(payload, "device_id", source)
    if not isinstance(device_id, str) or not device_id.strip():
        raise CollectorConfigError(f"device_id가 비어 있습니다: {device_id!r}")

    frame = payload.get("frame", {})
    width = int(frame.get("width", 0))
    height = int(frame.get("height", 0))
    if width <= 0 or height <= 0:
        raise CollectorConfigError(
            f"{source}에 frame.width / frame.height 가 필요합니다. "
            "ROI와 선이 정규화 좌표이므로 프레임 크기를 알아야 합니다."
        )

    roi_raw = payload.get("roi")
    roi: NormalizedRect | None = None
    if roi_raw is not None and roi_raw.get("enabled", True):
        roi = NormalizedRect(
            _unit(_require(roi_raw, "x1", f"{source}.roi"), "roi.x1"),
            _unit(_require(roi_raw, "y1", f"{source}.roi"), "roi.y1"),
            _unit(_require(roi_raw, "x2", f"{source}.roi"), "roi.x2"),
            _unit(_require(roi_raw, "y2", f"{source}.roi"), "roi.y2"),
        )

    line_raw = _require(payload, "line", source)
    start = _require(line_raw, "start", f"{source}.line")
    end = _require(line_raw, "end", f"{source}.line")
    if len(start) != 2 or len(end) != 2:
        raise CollectorConfigError(f"{source}.line 의 start/end 는 [x, y] 입니다.")
    line = Line(
        NormalizedPoint(_unit(start[0], "line.start.x"), _unit(start[1], "line.start.y")),
        NormalizedPoint(_unit(end[0], "line.end.x"), _unit(end[1], "line.end.y")),
    )
    try:
        in_direction = InDirection(line_raw.get("in_direction", "POSITIVE"))
        anchor = Anchor(line_raw.get("anchor", Anchor.BOTTOM_CENTER.value))
    except ValueError as error:
        raise CollectorConfigError(f"{source}.line: {error}") from error

    band = float(line_raw.get("band", 0.01))
    min_confirm = float(line_raw.get("min_confirm_distance", 0.03))
    cooldown = float(line_raw.get("cooldown_seconds", 0.5))
    if min_confirm < band:
        raise CollectorConfigError(
            "min_confirm_distance는 band 이상이어야 합니다: "
            f"{min_confirm} < {band}"
        )

    observation_raw = payload.get("observation", {})
    observation = ObservationSettings(
        interval_seconds=float(
            observation_raw.get(
                "interval_seconds", DEFAULT_OBSERVATION_INTERVAL_SECONDS
            )
        ),
        aggregation=str(observation_raw.get("aggregation", "latest")),
    )

    policy_raw = payload.get("congestion_policy")
    congestion_policy: CountThresholdPolicy | None = None
    if policy_raw is not None:
        kind = policy_raw.get("type", "count_thresholds")
        if kind != "count_thresholds":
            raise CollectorConfigError(
                f"{source}.congestion_policy.type 은 count_thresholds 만 "
                f"지원합니다: {kind}"
            )
        try:
            calibration = Calibration(
                policy_raw.get("calibration", Calibration.DEMO_ONLY_UNCALIBRATED.value)
            )
        except ValueError as error:
            raise CollectorConfigError(
                f"{source}.congestion_policy.calibration: {error}"
            ) from error
        congestion_policy = CountThresholdPolicy(
            comfortable_max=int(
                _require(policy_raw, "comfortable_max", f"{source}.congestion_policy")
            ),
            normal_max=int(
                _require(policy_raw, "normal_max", f"{source}.congestion_policy")
            ),
            crowded_max=int(
                _require(policy_raw, "crowded_max", f"{source}.congestion_policy")
            ),
            calibration=calibration,
        )

    model_version = payload.get("model_version", DEFAULT_MODEL_VERSION)
    if model_version is not None and not str(model_version).strip():
        raise CollectorConfigError("model_version이 공백입니다.")

    return CameraConfig(
        site_id=site_id,
        device_id=device_id.strip(),
        frame_width=width,
        frame_height=height,
        roi=roi,
        line=line,
        in_direction=in_direction,
        anchor=anchor,
        band=band,
        min_confirm_distance=min_confirm,
        cooldown_seconds=cooldown,
        observation=observation,
        congestion_policy=congestion_policy,
        model_version=None if model_version is None else str(model_version).strip(),
    )
