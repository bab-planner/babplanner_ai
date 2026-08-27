"""CCTV 기반 비식별 혼잡도/이용 인원 분석 (AI-1).

이 패키지가 하는 일과 하지 않는 일:

    한다     프레임 안에 사람 상자가 몇 개인가 (occupancy)
    한다     입구 선을 IN 방향으로 몇 명이 넘었는가 (entry count)
    한다     그 관측을 얼마나 믿을 만한가 (reliability signal)

    안 한다  얼굴 검출, 얼굴 특징 추출, 얼굴 대조
    안 한다  같은 사람을 다른 영상/다른 날에 다시 알아보기 (re-identification)
    안 한다  이름, 학번, 전화번호, 회원 정보와 연결하기

track_id는 영상 한 세션 안에서만 유효한 임시 번호이고 파일로 나가지 않는다.

무거운 의존성(torch, opencv)은 detectors/, video.py, annotate.py 안에만
있다. 나머지는 표준 라이브러리와 pydantic만 쓴다.
"""

from cctv_counting.detection import Detection, PersonDetector
from cctv_counting.observations import FrameObservation
from cctv_counting.reliability import ReliabilityFlag
from cctv_counting.tracking import TrackedPerson

__all__ = [
    "Detection",
    "FrameObservation",
    "PersonDetector",
    "ReliabilityFlag",
    "TrackedPerson",
]
