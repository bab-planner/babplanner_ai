"""현장에서 쓰는 식별자.

전부 임의값이다. 사람 이름이나 학번을 식별자에 넣으면 그 값이 파일명·경로·
매니페스트를 타고 온 데이터셋에 퍼진다. 나중에 지우려면 전부 다시 만들어야
한다.
"""

from __future__ import annotations

import re
import uuid

#: 식별자에 허용하는 모양. 경로 조각으로 쓰이므로 구분자나 상위 이동이
#: 들어올 수 없게 한다.
ID_PATTERN = re.compile(r"^[a-z]{1,8}_[0-9a-f]{12,32}$")

SAMPLE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def _token(length: int = 12) -> str:
    return uuid.uuid4().hex[:length]


def new_collection_session_id() -> str:
    """수집 회차 하나."""
    return f"cs_{_token()}"


def new_participant_group_id() -> str:
    """참가자 한 사람.

    이 값과 실제 사람을 잇는 표가 필요하면 운영자가 이 저장소 밖의 안전한
    곳에서 따로 관리한다. 여기에도 데이터셋에도 두지 않는다.
    """
    return f"p_{_token()}"


def new_capture_session_id() -> str:
    """한 끼 촬영 회차. 분할이 새지 않게 묶는 단위다."""
    return f"cap_{_token()}"


def new_tare_profile_id() -> str:
    return f"tp_{_token()}"


def new_sample_id() -> str:
    """표본 하나. 이미지 파일명도 이 값을 쓴다."""
    return uuid.uuid4().hex


def is_safe_id(value: str) -> bool:
    return bool(ID_PATTERN.match(value))


def is_safe_sample_id(value: str) -> bool:
    return bool(SAMPLE_ID_PATTERN.match(value))
