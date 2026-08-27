"""영상 한 편을 돌려 결과 파일을 만든다.

    .venv\\Scripts\\python.exe research\\cctv_counting\\scripts\\run_video.py ^
      --input C:\\path\\to\\demo.mp4 ^
      --config research\\cctv_counting\\configs\\demo.yaml ^
      --output-dir outputs\\cctv_demo

영상 경로는 반드시 인자로 받는다. 저장소 어딘가에 있는 영상을 찾아서 알아서
쓰지 않는다. CCTV 영상에는 사람이 찍혀 있고, 어느 영상을 처리할지는 사람이
정해야 한다.

만드는 것:

    annotated.mp4      겹쳐 그린 영상 (원본은 건드리지 않는다)
    observations.csv   프레임별 관측
    summary.json       영상 전체 요약
    metadata.json      재현에 필요한 기록 (설정, 모델, 장치, 버전)
    windows.csv        시간 창 통계 (설정에서 켰을 때만)
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cctv_counting.config import CctvConfig, ConfigError, config_to_dict, load_config
from cctv_counting.observations import FrameObservation, write_observations
from cctv_counting.pipeline import CountingPipeline
from cctv_counting.summary import build_summary, write_json
from cctv_counting.windows import aggregate_windows

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _package_versions() -> dict[str, str]:
    """실제로 불러온 버전을 기록한다. 설치 목록이 아니라 실행된 것이다."""
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("torch", "torchvision", "cv2", "numpy"):
        try:
            module = __import__(name)
        except ModuleNotFoundError:
            continue
        versions[name] = getattr(module, "__version__", "unknown")
    return versions


def _build_detector(config: CctvConfig):
    from cctv_counting.detectors.torchvision_detector import TorchvisionPersonDetector

    return TorchvisionPersonDetector(
        model_name=config.detector.model,
        device=config.detector.device.value,
        max_input_size=config.detector.max_input_size,
    )


def _write_windows(path: Path, observations: list[FrameObservation], seconds: float) -> None:
    rows = aggregate_windows(observations, seconds)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "start_seconds",
                "end_seconds",
                "frame_count",
                "mean_occupancy",
                "median_occupancy",
                "max_occupancy",
                "entries_in_window",
                "low_reliability_ratio",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    f"{row.start_seconds:.3f}",
                    f"{row.end_seconds:.3f}",
                    row.frame_count,
                    f"{row.mean_occupancy:.4f}",
                    f"{row.median_occupancy:.4f}",
                    row.max_occupancy,
                    "" if row.entries_in_window is None else row.entries_in_window,
                    f"{row.low_reliability_ratio:.4f}",
                ]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="처리할 영상 파일")
    parser.add_argument("--config", type=Path, required=True, help="실험 설정 파일")
    parser.add_argument("--output-dir", type=Path, required=True, help="결과를 쓸 폴더")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="설정 파일의 device를 덮어쓴다",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="설정 파일의 detector confidence_threshold를 덮어쓴다",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="앞에서 이만큼만 처리한다. 빠른 확인용",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"영상 파일이 없습니다: {args.input}", file=sys.stderr)
        return EXIT_USAGE

    try:
        config = load_config(args.config)
    except (ConfigError, ValueError) as error:
        print(f"설정을 읽을 수 없습니다: {error}", file=sys.stderr)
        return EXIT_USAGE

    overrides: dict[str, object] = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.confidence is not None:
        overrides["confidence_threshold"] = args.confidence
    if overrides:
        config = config.model_copy(
            update={"detector": config.detector.model_copy(update=overrides)}
        )

    from cctv_counting.video import AnnotatedVideoWriter, VideoError, VideoReader

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / "annotated.mp4"

    try:
        detector = _build_detector(config)
    except RuntimeError as error:
        print(f"detector를 준비할 수 없습니다: {error}", file=sys.stderr)
        return EXIT_FAILED

    print(f"detector: {detector.describe()}")

    observations: list[FrameObservation] = []
    started = time.perf_counter()

    try:
        with VideoReader(args.input, frame_stride=config.frame_stride) as reader:
            info = reader.info
            print(f"영상: {info.width}x{info.height} @ {info.fps:.2f}fps")

            pipeline = CountingPipeline(
                detector=detector,
                config=config,
                frame_width=info.width,
                frame_height=info.height,
            )

            writer = None
            if config.output.annotated_video:
                from cctv_counting.annotate import draw_frame

                writer = AnnotatedVideoWriter(
                    annotated_path,
                    width=info.width,
                    height=info.height,
                    fps=info.fps / config.frame_stride if info.fps > 0 else 0.0,
                    source=args.input,
                )

            last_timestamp = 0.0
            try:
                for frame_index, timestamp, frame in reader.frames():
                    if args.max_frames is not None and len(observations) >= args.max_frames:
                        break
                    result = pipeline.process_frame(frame_index, timestamp, frame)
                    observations.append(result.observation)
                    last_timestamp = timestamp
                    if writer is not None:
                        writer.write(
                            draw_frame(
                                frame,
                                result,
                                config,
                                width=info.width,
                                height=info.height,
                            )
                        )
            finally:
                if writer is not None:
                    writer.close()
    except VideoError as error:
        print(f"영상 처리 실패: {error}", file=sys.stderr)
        return EXIT_FAILED

    elapsed = time.perf_counter() - started
    if not observations:
        print("프레임을 한 장도 읽지 못했습니다.", file=sys.stderr)
        return EXIT_FAILED

    duration = info.duration_seconds
    if duration is None:
        duration = last_timestamp

    summary = build_summary(
        video=args.input.name,
        observations=observations,
        duration_seconds=duration,
        processing_fps=(len(observations) / elapsed) if elapsed > 0 else None,
        naive_line_band_detections=pipeline.naive_line_band_detections,
    )

    if config.output.observations_csv:
        write_observations(output_dir / "observations.csv", observations)
    if config.output.summary_json:
        write_json(output_dir / "summary.json", summary.to_dict())
    if config.output.windows_csv and config.window.enabled:
        _write_windows(output_dir / "windows.csv", observations, config.window.seconds)

    write_json(
        output_dir / "metadata.json",
        {
            "runAt": datetime.now(timezone.utc).isoformat(),
            "input": str(args.input),
            "config": config_to_dict(config),
            "configPath": str(args.config),
            "detector": detector.describe(),
            "packages": _package_versions(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "seed": config.seed,
            "elapsedSeconds": elapsed,
        },
    )

    print()
    print(f"프레임 {summary.frames_processed}장, {elapsed:.1f}초")
    print(f"처리 속도  {summary.processing_fps:.2f} fps" if summary.processing_fps else "")
    print(f"최대 동시  {summary.peak_occupancy}명")
    print(f"평균 동시  {summary.average_occupancy:.2f}명")
    if summary.entries_total is None:
        print("입장 인원  N/A (추적을 껐으므로 셀 수 없음)")
        if summary.naive_line_band_detections is not None:
            print(
                f"  선 근처 상자 누적 {summary.naive_line_band_detections}개 "
                "(순진한 대리 지표. 입장 인원이 아님)"
            )
    else:
        print(f"입장 인원  {summary.entries_total}명")
        print(f"퇴장 인원  {summary.exits_total}명")
    print(f"낮은 신뢰 프레임 비율 {summary.low_reliability_frame_ratio:.1%}")
    print(f"\n결과: {output_dir}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
