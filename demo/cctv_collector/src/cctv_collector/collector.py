"""영상을 관측으로 바꾸고 백엔드로 보낸다.

**detector는 프레임당 한 번이다.** A6가 검증한 dual-path 구조를 그대로
쓴다(`cctv_counting.dualpath.DualPathCounter`). occupancy 갈래와 entry
갈래가 같은 추론 결과를 나눠 쓰고, 갈라지는 것은 필터 정책뿐이다. 여기서
detector를 다시 만들거나 두 번 부르지 않는다.

**두 개의 시간축을 분리한다.**

    source_video_timestamp_seconds   영상 안에서의 위치. 진단용이다.
    observed_at                      수집기가 그 관측을 만든 실제 시각.

녹화 영상을 재생하는 데모에서 영상 안의 시각을 `observedAt`으로 보내면
안 된다. 백엔드는 늦게 도착한 과거 관측이 현재값을 되돌리지 않도록
`observedAt`을 비교하는데(`CongestionIngestService.updateIfNotOlder`),
2005년에 찍힌 CAVIAR 영상의 시각을 보내면 현재값이 영원히 갱신되지 않는다.
그것은 백엔드가 옳게 동작한 것이고, 보내는 쪽이 틀린 것이다.

**표본을 모아서 보낸다.** 25fps 영상이라고 초당 25번 POST하지 않는다.
혼잡도는 초 단위로 뒤집히는 값이 아니고, 백엔드는 관측 하나마다 이력과
현재값 두 테이블에 쓴다.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from cctv_counting.dualpath import DualPathCounter, TaskPolicy

from cctv_collector.config import CameraConfig
from cctv_collector.congestion_policy import CongestionLevel
from cctv_collector.schemas import CongestionObservation, LocalObservationRecord
from cctv_collector.spring_client import SendResult, SendStatus


class CollectorError(RuntimeError):
    """수집기를 돌릴 수 없다."""


@dataclass(frozen=True)
class Sample:
    """프레임 하나의 결과. 창 안에 모였다가 대표값 하나가 된다."""

    frame_index: int
    video_timestamp_seconds: float
    occupancy_count: int
    entries_total: int
    exits_total: int


def aggregate(samples: list[Sample], how: str) -> int:
    """창 안의 표본을 대표값 하나로 줄인다.

    두 가지만 둔다. 연구가 정의한 occupancy를 변형하지 않는 것이 목적이고,
    숫자를 좋아 보이게 하는 적응형 평활은 넣지 않는다.

        latest   창의 마지막 표본. 가장 최근의 상태다.
        median   창 표본의 중앙값. 한 프레임짜리 튀는 값에 덜 흔들린다.
    """
    if not samples:
        raise CollectorError("빈 창에서 대표값을 만들 수 없습니다.")
    counts = [sample.occupancy_count for sample in samples]
    if how == "latest":
        return counts[-1]
    if how == "median":
        # 짝수 개일 때 중앙 두 값의 평균이 정수가 아닐 수 있다.
        # 인원수는 정수여야 하므로 낮은 쪽 중앙값을 쓴다. 올림으로 사람을
        # 만들어 내지 않는다.
        return int(statistics.median_low(counts))
    raise CollectorError(f"알 수 없는 aggregation입니다: {how}")


@dataclass
class CollectorStats:
    """한 번의 실행 요약. 시연에서 이것만 봐도 무슨 일이 있었는지 안다."""

    frames: int = 0
    detector_calls: int = 0
    observations: int = 0
    sent: int = 0
    send_failed: int = 0
    skipped_dry_run: int = 0
    entries_total: int = 0
    exits_total: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "detectorCalls": self.detector_calls,
            "observations": self.observations,
            "sent": self.sent,
            "sendFailed": self.send_failed,
            "skippedDryRun": self.skipped_dry_run,
            # 아래 둘은 로컬 전용이다. 백엔드로 가지 않는다.
            "entriesTotal": self.entries_total,
            "exitsTotal": self.exits_total,
        }


@dataclass
class Collector:
    """프레임 흐름 -> 관측 -> (선택적) 전송.

    영상 파일을 모른다. 프레임을 주는 쪽이 `run()`에 넘긴다. 그래서 opencv
    없이 가짜 프레임으로 전체 흐름을 그대로 테스트할 수 있다.
    """

    camera: CameraConfig
    policy: TaskPolicy
    detector: Any
    #: None이면 dry-run이다. 백엔드로 아무것도 보내지 않는다.
    sender: Callable[[CongestionObservation], SendResult] | None = None
    #: 관측 시각을 만드는 함수. 테스트가 고정 시계를 넣는다.
    clock: Callable[[], datetime] = field(
        default=lambda: datetime.now(timezone.utc).astimezone()
    )

    stats: CollectorStats = field(default_factory=CollectorStats, init=False)
    _counter: DualPathCounter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.camera.congestion_policy is None and self.sender is not None:
            # 안전장치. cli에서도 막지만 여기서도 막는다. 임계값 없이
            # 단계를 만들려면 코드가 숫자를 지어내야 한다.
            raise CollectorError(
                "혼잡도 임계값이 없으면 백엔드로 보낼 수 없습니다. "
                "카메라 설정에 congestion_policy를 넣거나 dry-run으로 돌리세요."
            )
        self._counter = DualPathCounter(policy=self.policy, scene=self.camera.scene())

    @property
    def detector_calls(self) -> int:
        """detector를 몇 번 불렀는가. 프레임 수와 같아야 한다."""
        return self._counter.detector_calls

    def run(
        self, frames: Iterable[tuple[int, float, Any]]
    ) -> Iterator[LocalObservationRecord]:
        """프레임을 받아 관측 기록을 차례로 낸다.

        frames는 (프레임 번호, 영상 안 시각(초), 프레임)이다.
        `cctv_counting.video.VideoReader.frames()`가 그대로 이 모양이다.
        """
        window: list[Sample] = []
        window_started: float | None = None
        interval = self.camera.observation.interval_seconds

        for frame_index, video_seconds, frame in frames:
            result = self._counter.process_frame(
                frame_index, video_seconds, frame, self.detector
            )
            self.stats.frames += 1
            self.stats.detector_calls = self._counter.detector_calls
            sample = Sample(
                frame_index=frame_index,
                video_timestamp_seconds=video_seconds,
                occupancy_count=result.occupancy_count,
                entries_total=result.entries_total,
                exits_total=result.exits_total,
            )
            window.append(sample)
            if window_started is None:
                window_started = video_seconds

            if video_seconds - window_started >= interval:
                yield self._emit(window)
                window = []
                window_started = None

        if window:
            # 마지막 조각이 창보다 짧아도 버리지 않는다. 짧은 데모 영상에서
            # 관측이 하나도 안 나가는 일을 막는다.
            yield self._emit(window)

        self.stats.detector_calls = self._counter.detector_calls

    def _emit(self, window: list[Sample]) -> LocalObservationRecord:
        people_count = aggregate(window, self.camera.observation.aggregation)
        last = window[-1]

        level: CongestionLevel | None = None
        if self.camera.congestion_policy is not None:
            level = self.camera.congestion_policy.level_for(people_count)

        # **관측 시각은 지금이다.** 영상 안의 시각이 아니다.
        observed_at = self.clock()
        self.stats.observations += 1
        self.stats.entries_total = last.entries_total
        self.stats.exits_total = last.exits_total

        result = SendResult(status=SendStatus.SKIPPED_DRY_RUN)
        if self.sender is not None and level is not None:
            observation = CongestionObservation(
                site_id=self.camera.site_id,
                device_id=self.camera.device_id,
                people_count=people_count,
                level=level,
                observed_at=observed_at,
                model_version=self.camera.model_version,
            )
            result = self.sender(observation)

        if result.status is SendStatus.SUCCESS:
            self.stats.sent += 1
        elif result.status is SendStatus.BACKEND_SEND_FAILED:
            self.stats.send_failed += 1
        else:
            self.stats.skipped_dry_run += 1

        return LocalObservationRecord(
            video_timestamp_seconds=last.video_timestamp_seconds,
            observed_at=observed_at,
            site_id=self.camera.site_id,
            device_id=self.camera.device_id,
            people_count=people_count,
            level=level,
            samples=len(window),
            entries_total=last.entries_total,
            exits_total=last.exits_total,
            send_status=result.status.value,
            http_status=result.http_status,
        )
