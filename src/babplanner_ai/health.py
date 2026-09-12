"""프로세스 상태 확인.

서버 프로세스가 요청을 받을 수 있는지만 답한다. 모델을 쓸 수 있는지는
/internal/v1/ai/demand-forecast/status 가 답한다. 둘을 섞으면 모델 파일 하나가
없다고 컨테이너 오케스트레이터가 서버를 계속 재시작한다.

토큰을 요구하지 않는다. 컨테이너 헬스체크가 부르는 경로이고, 아무 정보도
돌려주지 않는다.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str


@router.get("/health", response_model=HealthResponse, summary="프로세스 상태")
def health() -> HealthResponse:
    return HealthResponse(status="ok")
