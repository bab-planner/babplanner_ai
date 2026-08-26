"""잔반 분석 라우트.

서비스 백엔드만 부른다. 사용자 앱이 직접 부르는 endpoint는 여기 없다.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from babplanner_ai.config import Settings, get_settings
from babplanner_ai.security import require_service_token
from babplanner_ai.waste.estimator import WasteEstimateError, WasteEstimator
from babplanner_ai.waste.fixture_estimator import FixtureWasteEstimator
from babplanner_ai.waste.models import AIWasteResponse
from babplanner_ai.waste.service import InvalidImageError, WasteAnalysisService

log = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/v1/ai/waste", tags=["InternalAI"])


def get_estimator(settings: Settings = Depends(get_settings)) -> WasteEstimator:
    """설정이 고른 분석기를 돌려준다.

    고를 수 있는 값이 하나뿐이어도 분기를 남겨둔다. 실제 모델이 붙을 때
    바뀌는 곳이 여기 하나가 되도록.
    """
    if settings.waste_model_mode.value == "fixture":
        return FixtureWasteEstimator()
    raise RuntimeError(f"알 수 없는 분석기 모드입니다: {settings.waste_model_mode}")


def get_service(
    settings: Settings = Depends(get_settings),
    estimator: WasteEstimator = Depends(get_estimator),
) -> WasteAnalysisService:
    return WasteAnalysisService(estimator, settings.max_image_bytes)


@router.post(
    "/analyze",
    response_model=AIWasteResponse,
    dependencies=[Depends(require_service_token)],
)
async def analyze_waste(
    requestId: str = Form(...),
    menuId: int = Form(...),
    image: UploadFile = File(...),
    service: WasteAnalysisService = Depends(get_service),
) -> AIWasteResponse:
    if not requestId.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="requestId 가 비어 있습니다.",
        )

    content = await image.read()
    try:
        return service.analyze(requestId, menuId, content)
    except InvalidImageError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error
    except WasteEstimateError as error:
        # 분석 자체를 못 한 경우다. 판정으로 위장하지 않는다. 호출자는 이걸
        # 보고 제출을 실패로 끝낸다.
        log.warning("잔반 분석을 수행하지 못했습니다. requestId=%s", requestId)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="분석기를 사용할 수 없습니다.",
        ) from error
