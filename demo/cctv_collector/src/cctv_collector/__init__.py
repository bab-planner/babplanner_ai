"""CCTV 수집기 — 영상에서 혼잡도 관측을 만들어 백엔드로 보낸다.

**연구가 아니다.** 여기서 임계값을 고르거나 모델을 바꾸지 않는다. A1~A6가
동결한 결과를 그대로 읽어 쓰고, 이 패키지가 더하는 것은 셋뿐이다.

    1. 표본 추출     프레임마다가 아니라 정해진 간격으로 관측을 만든다
    2. 운영 정책     인원수를 혼잡도 단계로 바꾼다 (AI 출력이 아니다)
    3. 전송          Spring 내부 수집 경로로 보낸다

**계수 로직을 복사하지 않는다.** detector도 tracker도 dual-path도
`cctv_counting`의 것을 그대로 쓴다. 복사본을 만들면 연구가 검증한 것과
시연에 나가는 것이 서로 다른 코드가 된다.

이 패키지의 정책/전송 로직은 torch와 opencv 없이 import된다. 영상 디코딩과
추론만 무거운 의존성을 쓰고, 그 import는 함수 안에 있다.
"""

from cctv_collector.collector import Collector, CollectorError
from cctv_collector.congestion_policy import (
    Calibration,
    CongestionLevel,
    CountThresholdPolicy,
)
from cctv_collector.schemas import CongestionObservation, LocalObservationRecord
from cctv_collector.spring_client import SendStatus, SpringCongestionClient

__all__ = [
    "Calibration",
    "Collector",
    "CollectorError",
    "CongestionLevel",
    "CongestionObservation",
    "CountThresholdPolicy",
    "LocalObservationRecord",
    "SendStatus",
    "SpringCongestionClient",
]
