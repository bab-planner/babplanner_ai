"""수집기 실행 진입점.

**기본은 dry-run이다.** 그냥 돌렸는데 서비스 DB에 값이 쌓이면 안 된다.
백엔드로 보내려면 `--send`를 명시해야 한다.

    :: 아무것도 보내지 않고 화면으로만 본다
    .venv-cctv\\Scripts\\python.exe -m cctv_collector ^
      --video demo.mp4 --config demo\\cctv_collector\\cameras\\demo_camera.example.yaml

    :: 실제로 보낸다. 토큰과 주소는 환경 변수에서만 온다.
    set BAPPLANNER_BACKEND_BASE_URL=http://127.0.0.1:8080
    set BAPPLANNER_INTERNAL_SERVICE_TOKEN=...
    .venv-cctv\\Scripts\\python.exe -m cctv_collector ^
      --video demo.mp4 --config ...yaml --send

**토큰을 인자로 받지 않는다.** 명령줄은 프로세스 목록과 셸 히스토리에 남는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cctv_collector.collector import Collector, CollectorError
from cctv_collector.config import (
    CollectorConfigError,
    adopted_research_policy,
    load_backend_settings,
    load_camera_config,
)
from cctv_collector.congestion_policy import Calibration
from cctv_collector.spring_client import SpringCongestionClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cctv_collector",
        description="CCTV 영상에서 혼잡도 관측을 만들어 Spring으로 보낸다.",
    )
    parser.add_argument(
        "--video", type=Path, required=True, help="입력 영상 파일 경로"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="카메라 설정 YAML/JSON"
    )
    parser.add_argument("--site-id", type=int, help="설정의 site_id를 덮어쓴다")
    parser.add_argument("--device-id", help="설정의 device_id를 덮어쓴다")
    parser.add_argument(
        "--send",
        action="store_true",
        help="백엔드로 실제 전송한다. 명시하지 않으면 dry-run이다.",
    )
    parser.add_argument(
        "--backend-url",
        help="백엔드 주소. 없으면 BAPPLANNER_BACKEND_BASE_URL 을 쓴다.",
    )
    parser.add_argument("--jsonl", type=Path, help="관측 기록을 JSONL로 저장")
    parser.add_argument(
        "--limit-seconds",
        type=float,
        help="영상 앞부분 이만큼만 처리한다. 시연 리허설용.",
    )
    parser.add_argument("--device", default="auto", help="detector 장치 (auto/cpu/cuda)")
    parser.add_argument(
        "--preview-frame",
        type=int,
        help="이 번호의 프레임에 ROI/선을 그려 저장하고 끝낸다. 카메라 보정용.",
    )
    parser.add_argument(
        "--preview-out",
        type=Path,
        # outputs/ 는 .gitignore가 막는다. 시연 산출물이 실수로 커밋되지 않게.
        default=Path("outputs/preview.png"),
        help="--preview-frame 저장 경로",
    )
    parser.add_argument("--quiet", action="store_true", help="관측 줄을 찍지 않는다")
    return parser


def _apply_overrides(camera, site_id: int | None, device_id: str | None):
    from dataclasses import replace

    changes = {}
    if site_id is not None:
        changes["site_id"] = site_id
    if device_id is not None:
        changes["device_id"] = device_id
    return replace(camera, **changes) if changes else camera


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        camera = _apply_overrides(
            load_camera_config(args.config), args.site_id, args.device_id
        )
        policy, policy_name = adopted_research_policy()
    except CollectorConfigError as error:
        print(f"설정 오류: {error}", file=sys.stderr)
        return 2

    # **live 전송은 임계값이 있어야 한다.** 없으면 거부한다.
    # 여기서 조용히 기본 임계값을 만들어 넣으면, 화면에 뜬 "여유"가 무엇을
    # 근거로 한 말인지 아무도 설명할 수 없다.
    if args.send and camera.congestion_policy is None:
        print(
            "혼잡도 임계값이 없어 전송을 거부합니다. 카메라 설정에 "
            "congestion_policy를 넣거나 dry-run으로 돌리세요.",
            file=sys.stderr,
        )
        return 2

    if args.preview_frame is not None:
        return _preview(args, camera)

    sender = None
    client: SpringCongestionClient | None = None
    if args.send:
        try:
            settings = load_backend_settings(base_url_override=args.backend_url)
        except CollectorConfigError as error:
            print(f"설정 오류: {error}", file=sys.stderr)
            return 2
        client = SpringCongestionClient(settings=settings)
        client.__enter__()
        sender = client.send

    from cctv_counting.detectors.torchvision_detector import TorchvisionPersonDetector
    from cctv_counting.video import VideoError, VideoReader

    print(f"채택된 연구 정책: {policy_name}  (occupancy conf >= {policy.occupancy_threshold})")
    print(f"모드: {'LIVE SEND' if args.send else 'DRY RUN'}")
    if camera.congestion_policy is not None:
        calibration = camera.congestion_policy.calibration
        if calibration is Calibration.DEMO_ONLY_UNCALIBRATED:
            print(
                "주의: 혼잡도 임계값이 보정되지 않은 시연용 값입니다 "
                "(DEMO_ONLY_UNCALIBRATED). 실제 식당 운영 기준이 아닙니다."
            )

    detector = TorchvisionPersonDetector(device=args.device)
    handle = None
    try:
        reader = VideoReader(args.video)
    except VideoError as error:
        print(f"영상 오류: {error}", file=sys.stderr)
        if client is not None:
            client.close()
        return 2

    if (reader.info.width, reader.info.height) != (
        camera.frame_width,
        camera.frame_height,
    ):
        print(
            f"주의: 영상 크기 {reader.info.width}x{reader.info.height} 가 설정의 "
            f"{camera.frame_width}x{camera.frame_height} 와 다릅니다. "
            "ROI와 선은 정규화 좌표라 동작은 하지만, 보정한 카메라가 맞는지 "
            "확인하세요."
        )

    if args.jsonl:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        handle = args.jsonl.open("w", encoding="utf-8")

    collector = Collector(
        camera=camera, policy=policy, detector=detector, sender=sender
    )
    try:
        frames = reader.frames()
        if args.limit_seconds is not None:
            frames = _limited(frames, args.limit_seconds)
        for record in collector.run(frames):
            if not args.quiet:
                level = record.level.value if record.level else "(정책 없음)"
                print(
                    f"  video {record.video_timestamp_seconds:7.2f}s  "
                    f"관측 {record.observed_at.isoformat(timespec='seconds')}  "
                    f"인원 {record.people_count:>3}  {level:<14} "
                    f"입장 {record.entries_total} 퇴장 {record.exits_total}  "
                    f"{record.send_status}"
                )
            if handle is not None:
                handle.write(
                    json.dumps(record.to_json(), ensure_ascii=False) + "\n"
                )
    except CollectorError as error:
        print(f"수집 오류: {error}", file=sys.stderr)
        return 1
    finally:
        reader.close()
        if handle is not None:
            handle.close()
        if client is not None:
            client.close()

    stats = collector.stats
    print("\n요약")
    for key, value in stats.as_dict().items():
        print(f"  {key:16} {value}")
    if stats.frames and stats.detector_calls != stats.frames:
        print(
            f"  경고: detector 호출({stats.detector_calls})이 프레임 수"
            f"({stats.frames})와 다릅니다.",
            file=sys.stderr,
        )
        return 1
    return 0 if stats.send_failed == 0 else 1


def _limited(frames, limit_seconds: float):
    for frame_index, video_seconds, frame in frames:
        if video_seconds > limit_seconds:
            return
        yield frame_index, video_seconds, frame


def _preview(args, camera) -> int:
    """프레임 하나에 ROI와 선을 그려 저장한다.

    복잡한 GUI를 만들지 않는다. 카메라를 새로 달았을 때 "선이 어디에
    그어졌는가"만 눈으로 확인할 수 있으면 된다.
    """
    try:
        import cv2
    except ModuleNotFoundError:
        print('opencv가 필요합니다. pip install -e ".[cctv]"', file=sys.stderr)
        return 2

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        print(f"영상을 열 수 없습니다: {args.video}", file=sys.stderr)
        return 2
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.preview_frame)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        print(f"프레임 {args.preview_frame}을(를) 읽을 수 없습니다.", file=sys.stderr)
        return 2

    height, width = frame.shape[:2]
    if camera.roi is not None:
        cv2.rectangle(
            frame,
            (int(camera.roi.x1 * width), int(camera.roi.y1 * height)),
            (int(camera.roi.x2 * width), int(camera.roi.y2 * height)),
            (0, 200, 0),
            1,
        )
    cv2.line(
        frame,
        (int(camera.line.start.x * width), int(camera.line.start.y * height)),
        (int(camera.line.end.x * width), int(camera.line.end.y * height)),
        (0, 0, 220),
        2,
    )
    args.preview_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.preview_out), frame)
    print(f"저장했습니다: {args.preview_out}  (초록 = ROI, 빨강 = counting line)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
