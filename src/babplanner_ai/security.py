"""서비스 간 인증.

사용자 인증이 아니다. 서비스 백엔드만 이 서비스를 부를 수 있게 하는
공유 토큰이다.
"""

from __future__ import annotations

import hmac

from fastapi import Depends, Header, HTTPException, status

from babplanner_ai.config import Settings, get_settings

SERVICE_TOKEN_HEADER = "X-Service-Token"


def tokens_match(presented: str | None, expected: str) -> bool:
    """토큰을 상수 시간으로 비교한다.

    길이나 앞자리가 얼마나 맞았는지가 응답 시간으로 새지 않게 한다.
    """
    if presented is None:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


def require_service_token(
    x_service_token: str | None = Header(default=None, alias=SERVICE_TOKEN_HEADER),
    settings: Settings = Depends(get_settings),
) -> None:
    """토큰이 맞지 않으면 401.

    없는 것과 틀린 것을 같은 응답으로 돌려준다. 어느 쪽인지 알려주면
    유효한 헤더 이름을 찾는 데 쓸 수 있다. 제시된 값도, 기대하는 값도
    응답에 담지 않는다.
    """
    if not tokens_match(x_service_token, settings.service_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="서비스 토큰이 올바르지 않습니다.",
        )
