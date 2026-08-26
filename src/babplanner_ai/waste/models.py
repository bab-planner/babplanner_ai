"""잔반 분석 계약 모델.

OpenAPI `AIWasteRequest` / `AIWasteResponse` 와 1:1로 대응한다.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class WasteAnalysisDecision(str, Enum):
    """분석 판정.

    서비스나 모델 장애는 여기에 담지 않는다. 그런 경우는 HTTP 오류로
    돌려주고 호출자가 실패로 처리한다. REJECTED는 분석이 정상적으로
    끝났지만 입력을 승인하지 않았다는 뜻이다.
    """

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class WasteEstimate(BaseModel):
    """분석기가 돌려주는 값.

    지급 관련 필드를 담지 않는다. 무엇을 얼마나 줄지는 분석기가 정할 일이
    아니다.
    """

    decision: WasteAnalysisDecision
    waste_rate: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    model_version: str = Field(min_length=1)


class AIWasteResponse(BaseModel):
    """OpenAPI AIWasteResponse."""

    requestId: str
    decision: WasteAnalysisDecision
    wasteRate: float | None = None
    confidence: float | None = None
    modelVersion: str

    @classmethod
    def of(cls, request_id: str, estimate: WasteEstimate) -> "AIWasteResponse":
        return cls(
            requestId=request_id,
            decision=estimate.decision,
            wasteRate=estimate.waste_rate,
            confidence=estimate.confidence,
            modelVersion=estimate.model_version,
        )
