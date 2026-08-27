"""화면 좌표와 도형.

여기서 두 종류의 좌표를 명확히 나눈다.

**정규화 좌표(normalized)** 는 0~1이다. ROI와 counting line처럼 사람이 config에
적는 값은 전부 이쪽이다. 같은 설정 파일을 1920x1080에도 1280x720에도 쓸 수
있어야 하기 때문이다. 카메라를 바꿨더니 선이 엉뚱한 곳에 그려지는 일을 막는다.

**픽셀 좌표(pixel)** 는 detector가 돌려주는 값이다. 프레임 크기를 알아야
의미가 생긴다.

두 좌표를 같은 타입으로 두면 반드시 섞인다. 그래서 타입을 따로 만든다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

#: 부동소수 비교에 쓰는 여유. 좌표 자체의 오차가 아니라 직렬화 오차용이다.
_EPSILON = 1e-9


class GeometryError(ValueError):
    """좌표나 도형이 성립하지 않는다."""


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeometryError(f"{name}이(가) 숫자가 아닙니다: {value!r}")
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise GeometryError(f"{name}이(가) 유한한 숫자가 아닙니다: {value!r}")
    return number


def _unit(name: str, value: float) -> float:
    number = _finite(name, value)
    if not -_EPSILON <= number <= 1.0 + _EPSILON:
        raise GeometryError(f"{name}은(는) 0~1 정규화 좌표여야 합니다: {number}")
    return min(1.0, max(0.0, number))


@dataclass(frozen=True)
class Point:
    """픽셀 좌표의 한 점."""

    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite("x", self.x))
        object.__setattr__(self, "y", _finite("y", self.y))


@dataclass(frozen=True)
class NormalizedPoint:
    """0~1 좌표의 한 점."""

    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _unit("x", self.x))
        object.__setattr__(self, "y", _unit("y", self.y))

    def to_pixels(self, width: int, height: int) -> Point:
        return Point(self.x * width, self.y * height)


@dataclass(frozen=True)
class BoundingBox:
    """픽셀 좌표의 축 정렬 사각형.

    x1 < x2, y1 < y2를 강제한다. detector가 뒤집힌 상자를 주면 그 자리에서
    터지는 편이 낫다. 넓이가 음수인 상자로 IoU를 계산하면 조용히 이상한
    매칭이 나온다.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise GeometryError(
                "너비와 높이가 0보다 커야 합니다: "
                f"({self.x1}, {self.y1}) ~ ({self.x2}, {self.y2})"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return Point((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Point:
        """발이 닿는 지점에 가장 가까운 점.

        바닥에 그은 선을 언제 넘었는지 판단할 때는 몸통 중심보다 이쪽이
        맞다. 사람이 카메라 쪽으로 걸어오면 상자가 세로로 커지는데, 중심점은
        발보다 먼저 선을 넘는다.
        """
        return Point((self.x1 + self.x2) / 2.0, self.y2)

    def translated(self, dx: float, dy: float) -> BoundingBox:
        return BoundingBox(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def iou(self, other: BoundingBox) -> float:
        left = max(self.x1, other.x1)
        top = max(self.y1, other.y1)
        right = min(self.x2, other.x2)
        bottom = min(self.y2, other.y2)
        if right <= left or bottom <= top:
            return 0.0
        intersection = (right - left) * (bottom - top)
        union = self.area + other.area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union


@dataclass(frozen=True)
class NormalizedRect:
    """0~1 좌표의 사각형. ROI가 이것이다."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise GeometryError(
                "ROI의 너비와 높이가 0보다 커야 합니다: "
                f"({self.x1}, {self.y1}) ~ ({self.x2}, {self.y2})"
            )

    def to_pixels(self, width: int, height: int) -> BoundingBox:
        return BoundingBox(
            self.x1 * width, self.y1 * height, self.x2 * width, self.y2 * height
        )

    def contains_normalized(self, point: NormalizedPoint) -> bool:
        return (
            self.x1 - _EPSILON <= point.x <= self.x2 + _EPSILON
            and self.y1 - _EPSILON <= point.y <= self.y2 + _EPSILON
        )


def rect_contains(rect: BoundingBox, point: Point) -> bool:
    """경계선 위는 안쪽으로 본다.

    ROI 경계에 정확히 걸친 사람을 밖으로 밀어내면, 경계 근처에서 사람이
    깜빡이며 occupancy가 흔들린다.
    """
    return (
        rect.x1 - _EPSILON <= point.x <= rect.x2 + _EPSILON
        and rect.y1 - _EPSILON <= point.y <= rect.y2 + _EPSILON
    )


class Side(str, Enum):
    """선을 기준으로 어느 쪽에 있는가.

    POSITIVE/NEGATIVE는 "위/아래"나 "안/밖"이 아니다. 선의 방향
    (start -> end)에 대한 외적 부호일 뿐이다. 어느 쪽이 식당 안인지는
    카메라마다 다르므로 config가 정한다.
    """

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    ON_LINE = "ON_LINE"


@dataclass(frozen=True)
class Line:
    """정규화 좌표의 counting line."""

    start: NormalizedPoint
    end: NormalizedPoint

    def __post_init__(self) -> None:
        if (
            abs(self.start.x - self.end.x) < _EPSILON
            and abs(self.start.y - self.end.y) < _EPSILON
        ):
            raise GeometryError("counting line의 시작점과 끝점이 같습니다.")

    def to_pixels(self, width: int, height: int) -> tuple[Point, Point]:
        return self.start.to_pixels(width, height), self.end.to_pixels(width, height)

    def signed_distance_normalized(self, point: NormalizedPoint) -> float:
        """선까지의 부호 있는 거리(정규화 단위).

        부호가 어느 쪽인지를, 크기가 얼마나 떨어졌는지를 알려준다. 크기를
        쓰는 이유는 jitter 방어 때문이다. 선 바로 위에서 1픽셀 떨리는 것과
        선을 확실히 건너간 것을 구분해야 한다.
        """
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        length = math.hypot(dx, dy)
        cross = dx * (point.y - self.start.y) - dy * (point.x - self.start.x)
        return cross / length

    def side_of(self, point: NormalizedPoint, band: float = 0.0) -> Side:
        """band 안쪽은 ON_LINE으로 본다.

        band=0이면 선 위에 정확히 놓인 점만 ON_LINE이다.
        """
        if band < 0.0:
            raise GeometryError(f"band는 0 이상이어야 합니다: {band}")
        distance = self.signed_distance_normalized(point)
        if abs(distance) <= band + _EPSILON:
            return Side.ON_LINE
        return Side.POSITIVE if distance > 0 else Side.NEGATIVE
