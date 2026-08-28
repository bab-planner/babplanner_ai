"""백엔드로 나가는 것과 로컬에만 남는 것.

**이 파일의 목적은 경계를 하나로 모으는 것이다.** 무엇이 서비스 DB로 가고
무엇이 가지 않는지가 여러 곳에 흩어져 있으면, 언젠가 편의를 위해 필드
하나가 조용히 늘어난다. 그 필드가 bbox나 track_id라면 그 순간부터 이
시스템은 익명 계수 시스템이 아니다.

    Spring으로 간다      집계된 인원수, 판정 단계, 관측 시각, 장비 id
    로컬에만 남는다      영상 시각, 통과/퇴장 수, 전송 결과, 표본 수

**입장 인원(entries_total)은 백엔드로 보내지 않는다.** 백엔드 계약에 그것을
받을 자리가 없고, 계약에 없는 필드를 몰래 보내면 보내는 쪽은 쓰이고 있다고
믿고 받는 쪽은 무시한다. 로컬 시연 화면에는 띄운다.

**source를 보내지 않는다.** 백엔드가 `CCTV_AI`로 직접 기록한다. 보내는 쪽이
출처를 정하게 하면 저장된 출처를 신뢰할 수 없다. 계약이 그렇게 되어 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cctv_collector.congestion_policy import CongestionLevel


class SchemaError(ValueError):
    """관측을 만들 수 없다."""


#: 백엔드 `CongestionObservationRequest`가 받는 필드 전부.
#: 여기 없는 이름을 payload에 넣지 않는다. 테스트가 이 목록으로 검사한다.
SPRING_REQUEST_FIELDS = frozenset(
    {
        "siteId",
        "deviceId",
        "peopleCount",
        "level",
        "observedAt",
        "displayMessage",
        "modelVersion",
    }
)

#: 필수 필드. 백엔드의 @NotNull / @NotBlank와 같다.
SPRING_REQUIRED_FIELDS = frozenset(
    {"siteId", "deviceId", "peopleCount", "level", "observedAt"}
)


@dataclass(frozen=True)
class CongestionObservation:
    """Spring으로 보내는 관측 하나.

    observed_at은 **수집 시점의 실제 시각**이다. 재생 중인 영상 안의 시각이
    아니다. 그 이유는 `collector.py`에 적어 두었다.
    """

    site_id: int
    device_id: str
    people_count: int
    level: CongestionLevel
    observed_at: datetime
    model_version: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.site_id, bool) or not isinstance(self.site_id, int):
            raise SchemaError(f"site_id는 정수여야 합니다: {self.site_id!r}")
        if self.site_id <= 0:
            raise SchemaError(f"site_id는 양수여야 합니다: {self.site_id}")
        if not self.device_id or not self.device_id.strip():
            raise SchemaError("device_id가 비어 있습니다.")
        if isinstance(self.people_count, bool) or not isinstance(
            self.people_count, int
        ):
            raise SchemaError(
                f"people_count는 정수여야 합니다: {self.people_count!r}"
            )
        if self.people_count < 0:
            raise SchemaError(
                f"people_count는 0 이상이어야 합니다: {self.people_count}"
            )
        if self.observed_at.tzinfo is None:
            # naive datetime을 보내면 백엔드가 어느 시간대로 읽을지 알 수 없다.
            raise SchemaError(
                "observed_at에 시간대가 없습니다. OffsetDateTime으로 나가야 합니다."
            )
        if self.model_version is not None and not self.model_version.strip():
            # 백엔드가 공백 문자열을 검증 오류로 되돌려준다.
            raise SchemaError("model_version이 공백입니다. 보내지 않거나 값을 주세요.")

    def to_payload(self) -> dict[str, Any]:
        """백엔드 계약 그대로의 JSON 본문.

        `displayMessage`는 넣지 않는다. 지어낼 문구가 없고, 지어낸 문구가
        화면에 뜨면 그것이 무엇을 근거로 한 말인지 아무도 설명할 수 없다.
        """
        payload: dict[str, Any] = {
            "siteId": self.site_id,
            "deviceId": self.device_id,
            "peopleCount": self.people_count,
            "level": self.level.value,
            "observedAt": _iso_offset(self.observed_at),
        }
        if self.model_version is not None:
            payload["modelVersion"] = self.model_version.strip()
        return payload


def _iso_offset(value: datetime) -> str:
    """Java OffsetDateTime이 읽는 형식.

    tz-aware datetime의 isoformat은 항상 오프셋을 붙인다(UTC면 `+00:00`).
    밀리초까지만 쓴다 — 마이크로초는 이 용도에서 의미가 없고, 로그를 눈으로
    비교할 때 길이만 늘린다.
    """
    if value.tzinfo is None:
        raise SchemaError("시간대가 없는 datetime은 보낼 수 없습니다.")
    return value.isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class LocalObservationRecord:
    """로컬 기록 한 줄. **백엔드로 가지 않는다.**

    영상 시각과 관측 시각을 나란히 남긴다. 재생 데모에서 둘이 다르다는 것이
    나중에 로그를 읽는 사람에게 보여야 한다.
    """

    video_timestamp_seconds: float
    observed_at: datetime
    site_id: int
    device_id: str
    people_count: int
    level: CongestionLevel | None
    samples: int
    entries_total: int | None
    exits_total: int | None
    send_status: str
    http_status: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "videoTimestampSeconds": round(self.video_timestamp_seconds, 3),
            "observedAt": _iso_offset(self.observed_at),
            "siteId": self.site_id,
            "deviceId": self.device_id,
            "peopleCount": self.people_count,
            "level": self.level.value if self.level is not None else None,
            "samples": self.samples,
            # 아래 둘은 로컬 전용이다. Spring payload에는 없다.
            "entriesTotal": self.entries_total,
            "exitsTotal": self.exits_total,
            "sendStatus": self.send_status,
            "httpStatus": self.http_status,
        }
