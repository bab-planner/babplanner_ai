"""영상 읽기와 쓰기.

여기서만 OpenCV를 쓴다. import를 함수 안에 두어, 이 모듈을 import하는 것
자체로는 opencv가 필요하지 않게 한다. core 로직 테스트가 무거운 의존성
없이 도는 것이 그 목적이다.

**원본을 덮어쓰지 않는다.** 결과 파일 경로가 입력 경로와 같으면 시작 전에
터진다. 몇 분짜리 CCTV 영상 원본을 실수로 덮어쓰면 되돌릴 방법이 없다.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class VideoError(RuntimeError):
    """영상을 다룰 수 없다."""


def _cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise VideoError(
            "영상 처리에는 opencv가 필요합니다. pip install -e \".[cctv]\""
        ) from error
    return cv2


@dataclass(frozen=True)
class VideoInfo:
    """영상의 기본 정보."""

    width: int
    height: int
    fps: float
    frame_count: int | None

    @property
    def duration_seconds(self) -> float | None:
        """컨테이너가 프레임 수를 안 알려주면 알 수 없다.

        모르는 것을 0으로 두면 "0초짜리 영상"이 된다. None으로 남긴다.
        """
        if self.frame_count is None or self.fps <= 0:
            return None
        return self.frame_count / self.fps


class VideoReader:
    """영상 파일을 프레임 단위로 읽는다."""

    def __init__(self, path: Path, frame_stride: int = 1) -> None:
        if frame_stride < 1:
            raise VideoError(f"frame_stride는 1 이상이어야 합니다: {frame_stride}")
        if not path.exists():
            raise VideoError(f"영상 파일이 없습니다: {path}")

        cv2 = _cv2()
        self._path = path
        self._stride = frame_stride
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise VideoError(f"영상을 열 수 없습니다: {path}")

        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        raw_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            self._capture.release()
            raise VideoError(f"프레임 크기를 읽을 수 없습니다: {path}")
        if fps <= 0 or fps != fps:
            # 일부 컨테이너는 fps를 0이나 NaN으로 준다. 시각 계산에 그대로
            # 쓰면 timestamp가 전부 깨지므로 여기서 걸러 낸다.
            fps = 0.0

        self.info = VideoInfo(
            width=width,
            height=height,
            fps=fps,
            frame_count=raw_count if raw_count > 0 else None,
        )

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()

    def frames(self) -> Iterator[tuple[int, float, Any]]:
        """(원본 프레임 번호, 초, 프레임)을 차례로 낸다.

        시각은 컨테이너가 알려주는 위치를 우선 쓴다. 가변 프레임률 영상에서
        번호 나누기 fps로 계산하면 뒤로 갈수록 어긋난다.
        """
        cv2 = _cv2()
        index = 0
        while True:
            position_ms = self._capture.get(cv2.CAP_PROP_POS_MSEC)
            ok, frame = self._capture.read()
            if not ok:
                break
            if index % self._stride == 0:
                if position_ms and position_ms > 0:
                    timestamp = position_ms / 1000.0
                elif self.info.fps > 0:
                    timestamp = index / self.info.fps
                else:
                    timestamp = 0.0
                yield index, timestamp, frame
            index += 1


class AnnotatedVideoWriter:
    """겹쳐 그린 영상을 새 파일로 쓴다."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        source: Path | None = None,
    ) -> None:
        if source is not None and path.resolve() == source.resolve():
            raise VideoError(f"원본을 덮어쓸 수 없습니다: {path}")
        if width <= 0 or height <= 0:
            raise VideoError(f"프레임 크기가 올바르지 않습니다: {width}x{height}")

        cv2 = _cv2()
        path.parent.mkdir(parents=True, exist_ok=True)
        # fps를 모르면 재생만 이상해지고 분석에는 영향이 없으므로 기본값을 쓴다.
        output_fps = fps if fps > 0 else 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, output_fps, (width, height))
        if not self._writer.isOpened():
            raise VideoError(f"출력 영상을 만들 수 없습니다: {path}")
        self._path = path

    def __enter__(self) -> AnnotatedVideoWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def write(self, frame: Any) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
