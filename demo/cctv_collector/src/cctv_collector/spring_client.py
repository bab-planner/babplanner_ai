"""Spring 내부 수집 경로로 관측을 보낸다.

    POST {base}/internal/v1/congestion-observations
    X-Service-Token: <BAPPLANNER_INTERNAL_SERVICE_TOKEN>
    -> 202 Accepted, 본문 없음

**토큰을 절대 로그에 남기지 않는다.** HTTP 오류를 다룰 때 가장 쉽게 새는
경로가 "요청 객체를 통째로 찍는" 코드다. 그래서 이 모듈은 예외에서 요청
헤더를 꺼내지 않고, 오류 메시지를 직접 만든다.

**추론 실패와 전송 실패를 구분한다.** 백엔드가 죽어 있어서 POST가 실패한
것을 "AI가 실패했다"로 적으면, 나중에 로그를 읽는 사람이 모델을 의심한다.
둘은 다른 사건이고 다른 사람이 고친다.

**재시도는 유한하다.** 데모에서 백엔드가 죽어 있으면 영상 처리가 통째로
멈추면 안 된다. 짧은 타임아웃 + 최대 1회 재시도로 끝내고, 실패는 실패로
기록한 뒤 다음 프레임으로 간다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cctv_collector.config import BackendSettings
from cctv_collector.schemas import CongestionObservation

#: 백엔드 `ServiceTokenAuthenticationFilter.SERVICE_TOKEN_HEADER`.
SERVICE_TOKEN_HEADER = "X-Service-Token"

#: 백엔드 `InternalCongestionController`의 경로.
INGEST_PATH = "/internal/v1/congestion-observations"

#: 백엔드는 본문 없이 202를 준다.
EXPECTED_STATUS = 202


class SendStatus(str, Enum):
    """전송 결과. **추론 결과가 아니다.**"""

    SUCCESS = "SUCCESS"
    #: 백엔드가 받지 못했다. 추론은 정상이었다.
    BACKEND_SEND_FAILED = "BACKEND_SEND_FAILED"
    #: dry-run이라 보내지 않았다. 실패가 아니다.
    SKIPPED_DRY_RUN = "SKIPPED_DRY_RUN"


@dataclass(frozen=True)
class SendResult:
    status: SendStatus
    http_status: int | None = None
    #: 사람이 읽을 오류 요약. **토큰도 헤더도 들어가지 않는다.**
    error: str | None = None
    attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.status is SendStatus.SUCCESS


@dataclass
class SpringCongestionClient:
    """관측 하나를 동기로 보낸다.

    **동기다.** 큐를 두지 않는 것이 이 데모의 backpressure 전략이다. 백엔드가
    느리면 다음 관측이 늦게 나갈 뿐, 메모리에 관측이 무한히 쌓이지 않는다.
    표본 간격이 1초이고 타임아웃이 그보다 짧으므로 영상 처리가 밀리지 않는다.

    connect_timeout: 연결까지. 백엔드가 아예 없으면 여기서 끝난다.
    read_timeout:    응답까지. 백엔드가 받고 멈춰 있어도 여기서 끝난다.
    max_retries:     추가 시도 횟수. 0이면 한 번만 보낸다.
    """

    settings: BackendSettings
    connect_timeout: float = 3.0
    read_timeout: float = 5.0
    max_retries: int = 1
    #: 테스트가 httpx.MockTransport를 넣는 자리. 실제 실행에서는 None이다.
    transport: Any = field(default=None, repr=False)

    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.connect_timeout <= 0 or self.read_timeout <= 0:
            raise ValueError("타임아웃은 0보다 커야 합니다.")
        if self.max_retries < 0:
            raise ValueError("max_retries는 0 이상이어야 합니다.")

    @property
    def url(self) -> str:
        return f"{self.settings.base_url}{INGEST_PATH}"

    def __enter__(self) -> SpringCongestionClient:
        self._client = self._build_client()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _build_client(self) -> Any:
        import httpx

        return httpx.Client(
            timeout=httpx.Timeout(
                connect=self.connect_timeout,
                read=self.read_timeout,
                write=self.read_timeout,
                pool=self.connect_timeout,
            ),
            transport=self.transport,
        )

    def send(self, observation: CongestionObservation) -> SendResult:
        """관측 하나를 보낸다. 예외를 밖으로 던지지 않는다.

        전송 실패로 영상 처리를 멈추지 않는다. 실패는 결과값으로 돌려주고,
        부르는 쪽이 기록한다.
        """
        import httpx

        client = self._client
        owned = client is None
        if owned:
            client = self._build_client()

        payload = observation.to_payload()
        headers = {
            SERVICE_TOKEN_HEADER: self.settings.token,
            "Content-Type": "application/json",
        }

        attempts = 0
        last_error = "전송을 시도하지 못했습니다."
        last_status: int | None = None
        try:
            for attempt in range(self.max_retries + 1):
                attempts = attempt + 1
                try:
                    response = client.post(self.url, json=payload, headers=headers)
                except httpx.TimeoutException:
                    last_error = (
                        f"백엔드 응답 시간 초과 (connect {self.connect_timeout}s / "
                        f"read {self.read_timeout}s)"
                    )
                    continue
                except httpx.TransportError as error:
                    # 예외 객체를 그대로 문자열로 만들지 않는다. httpx의
                    # 일부 예외는 request를 들고 있고, 그 안에 헤더가 있다.
                    last_error = f"백엔드에 연결하지 못했습니다: {type(error).__name__}"
                    continue

                last_status = response.status_code
                if response.status_code == EXPECTED_STATUS:
                    return SendResult(
                        status=SendStatus.SUCCESS,
                        http_status=response.status_code,
                        attempts=attempts,
                    )
                if 400 <= response.status_code < 500:
                    # 우리 payload나 토큰이 잘못됐다. 다시 보내도 같다.
                    return SendResult(
                        status=SendStatus.BACKEND_SEND_FAILED,
                        http_status=response.status_code,
                        error=_client_error_message(response.status_code),
                        attempts=attempts,
                    )
                last_error = f"백엔드 오류 응답: HTTP {response.status_code}"
        finally:
            if owned and client is not None:
                client.close()

        return SendResult(
            status=SendStatus.BACKEND_SEND_FAILED,
            http_status=last_status,
            error=last_error,
            attempts=attempts,
        )


def _client_error_message(status: int) -> str:
    """4xx를 사람이 읽을 문장으로. **토큰 값은 넣지 않는다.**"""
    if status in (401, 403):
        return (
            f"HTTP {status}. 내부 서비스 토큰이 거부되었습니다. "
            "환경 변수를 확인하세요(값은 로그에 남기지 않습니다)."
        )
    if status == 400:
        return f"HTTP {status}. 요청 본문이 백엔드 계약과 맞지 않습니다."
    if status == 404:
        return (
            f"HTTP {status}. 경로 또는 siteId를 찾을 수 없습니다. "
            "백엔드 주소와 site_id를 확인하세요."
        )
    return f"HTTP {status}."
