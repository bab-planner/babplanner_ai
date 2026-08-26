"""잔반 분석 흐름.

라우트는 요청을 받고 응답을 만드는 일만 한다. 검증과 판정은 여기 있다.
"""

from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError

from babplanner_ai.waste.estimator import WasteEstimator
from babplanner_ai.waste.models import AIWasteResponse


class InvalidImageError(ValueError):
    """이미지를 읽을 수 없다. 요청 자체가 처리 불가능하다."""


def _ensure_decodable(image: bytes, max_bytes: int) -> None:
    """이미지가 실제로 이미지인지 확인한다.

    서비스 백엔드도 content-type과 크기를 보지만, 헤더만으로는 내용이
    이미지라는 보장이 없다. 여기서 한 번 더 본다.

    형식은 제한하지 않는다. Pillow가 열 수 있으면 통과다. JPEG만 받겠다는
    식의 새 제약은 만들지 않는다.
    """
    if not image:
        raise InvalidImageError("이미지가 비어 있습니다.")
    if len(image) > max_bytes:
        raise InvalidImageError("이미지가 너무 큽니다.")
    try:
        # verify()는 한 번 쓰면 객체를 다시 못 쓰므로 확인 용도로만 연다.
        with Image.open(io.BytesIO(image)) as decoded:
            decoded.verify()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidImageError("이미지를 해석할 수 없습니다.") from error


class WasteAnalysisService:
    def __init__(self, estimator: WasteEstimator, max_image_bytes: int) -> None:
        self._estimator = estimator
        self._max_image_bytes = max_image_bytes

    def analyze(self, request_id: str, menu_id: int, image: bytes) -> AIWasteResponse:
        """한 건을 판정한다.

        이미지는 이 호출이 끝나면 사라진다. 원본 보관은 서비스 백엔드의
        저장소가 맡고 여기서는 어떤 파일도 남기지 않는다.
        """
        _ensure_decodable(image, self._max_image_bytes)
        estimate = self._estimator.estimate(image, menu_id)
        return AIWasteResponse.of(request_id, estimate)
