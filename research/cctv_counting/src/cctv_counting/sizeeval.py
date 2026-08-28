"""화면상 사람 크기별 검출 성능.

A4에서 occupancy 오차의 절반과 통과 계수 recall 0.4의 직접 원인이 **검출
누락**이었다. 누락이 작은 사람에 몰려 있는지 보려면 크기를 재야 한다.

**정의는 결과를 보기 전에 고정했다.** 나중에 경계를 옮기면 어떤 분포든
원하는 이야기로 만들 수 있다.

    주 변수   normalized_bbox_height = bbox_height / image_height
    보조      normalized_bbox_area   = bbox_area / (image_width * image_height)

높이를 주 변수로 쓰는 이유는 사람이 서 있는 자세에서 높이가 거리와 가장
곧게 대응하기 때문이다. 넓이는 자세와 가림에 더 흔들린다.

    SMALL    h/H <  0.10
    MEDIUM   0.10 <= h/H < 0.20
    LARGE    h/H >= 0.20

CAVIAR(288px 높이)에서 이것은 대략 29px 미만 / 29~58px / 58px 이상이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from cctv_counting.geometry import BoundingBox


class SizeEvalError(ValueError):
    """크기별 평가를 할 수 없다."""


class SizeBin(str, Enum):
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"


#: 경계. 결과를 보기 전에 정했고 이후 바꾸지 않는다.
SMALL_MAX = 0.10
MEDIUM_MAX = 0.20

#: 정답 상자와 검출 상자를 같은 사람으로 볼 최소 겹침.
MATCH_IOU = 0.5


def normalized_height(bbox: BoundingBox, image_height: int) -> float:
    if image_height <= 0:
        raise SizeEvalError(f"image_height는 양수여야 합니다: {image_height}")
    return bbox.height / image_height


def normalized_area(bbox: BoundingBox, image_width: int, image_height: int) -> float:
    if image_width <= 0 or image_height <= 0:
        raise SizeEvalError(
            f"이미지 크기가 올바르지 않습니다: {image_width}x{image_height}"
        )
    return bbox.area / (image_width * image_height)


def size_bin(normalized_h: float) -> SizeBin:
    """경계값은 위쪽 구간에 넣는다. 0.10은 MEDIUM, 0.20은 LARGE."""
    if normalized_h < 0:
        raise SizeEvalError(f"정규화 높이가 음수입니다: {normalized_h}")
    if normalized_h < SMALL_MAX:
        return SizeBin.SMALL
    if normalized_h < MEDIUM_MAX:
        return SizeBin.MEDIUM
    return SizeBin.LARGE


@dataclass
class SizeRecall:
    """크기 구간 하나의 집계."""

    total: int = 0
    matched: int = 0

    @property
    def recall(self) -> float | None:
        """표본이 없으면 None이다. 0.0으로 두면 "전부 놓쳤다"로 읽힌다."""
        if self.total == 0:
            return None
        return self.matched / self.total


@dataclass
class MatchSummary:
    """한 시퀀스(또는 한 조건)의 검출 성능."""

    frames: int = 0
    gt_total: int = 0
    gt_matched: int = 0
    predictions: int = 0
    false_positives: int = 0
    by_size: dict[SizeBin, SizeRecall] = field(default_factory=dict)
    #: 놓친 정답 상자의 정규화 높이. 크기 분포를 보려고 남긴다.
    missed_heights: list[float] = field(default_factory=list)
    #: 맞힌 정답 상자의 정규화 높이.
    matched_heights: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        for b in SizeBin:
            self.by_size.setdefault(b, SizeRecall())

    @property
    def recall(self) -> float | None:
        return None if self.gt_total == 0 else self.gt_matched / self.gt_total

    @property
    def precision(self) -> float | None:
        """예측이 하나도 없으면 정의되지 않는다."""
        if self.predictions == 0:
            return None
        return (self.predictions - self.false_positives) / self.predictions

    @property
    def false_positives_per_frame(self) -> float | None:
        return None if self.frames == 0 else self.false_positives / self.frames


def match_frame(
    predicted: list[BoundingBox],
    truth: list[BoundingBox],
    iou_threshold: float = MATCH_IOU,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """IoU가 큰 쌍부터 하나씩 확정하는 greedy 1:1 매칭.

    **한 예측이 두 정답에 붙거나 그 반대가 되면 안 된다.** 그러면 recall과
    precision이 동시에 부풀려진다. 동점일 때는 index가 작은 쪽이 이긴다.
    같은 입력이면 항상 같은 결과가 나와야 실험을 재현할 수 있다.

    Hungarian 대신 greedy를 쓴다. IoU 0.5 이상에서 한 정답에 겹치는 예측이
    여럿인 경우가 드물고, 그때도 가장 많이 겹치는 것을 고르는 것이 사람이
    보기에 자연스럽다. 이 선택을 보고서에 적는다.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise SizeEvalError(f"iou_threshold는 0~1이어야 합니다: {iou_threshold}")

    candidates: list[tuple[float, int, int]] = []
    for i, p in enumerate(predicted):
        for j, t in enumerate(truth):
            iou = p.iou(t)
            if iou >= iou_threshold:
                candidates.append((iou, i, j))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_p: set[int] = set()
    used_t: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, i, j in candidates:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        matches.append((i, j))

    return (
        matches,
        [i for i in range(len(predicted)) if i not in used_p],
        [j for j in range(len(truth)) if j not in used_t],
    )


def accumulate(
    summary: MatchSummary,
    predicted: list[BoundingBox],
    truth: list[BoundingBox],
    *,
    image_width: int,
    image_height: int,
    iou_threshold: float = MATCH_IOU,
) -> None:
    """한 프레임의 결과를 집계에 더한다."""
    matches, unmatched_p, unmatched_t = match_frame(predicted, truth, iou_threshold)
    matched_truth = {j for _, j in matches}

    summary.frames += 1
    summary.gt_total += len(truth)
    summary.gt_matched += len(matches)
    summary.predictions += len(predicted)
    summary.false_positives += len(unmatched_p)

    for index, box in enumerate(truth):
        h = normalized_height(box, image_height)
        bucket = summary.by_size[size_bin(h)]
        bucket.total += 1
        if index in matched_truth:
            bucket.matched += 1
            summary.matched_heights.append(h)
        else:
            summary.missed_heights.append(h)


#: 높이 분포를 기록할 때 쓰는 해상도. 0.001은 CAVIAR 288px에서 0.3픽셀이다.
HEIGHT_RESOLUTION = 0.001


def height_histogram(values: list[float]) -> dict[str, int]:
    """높이 목록을 눈금 0.001의 히스토그램으로 접는다.

    상자 하나하나의 높이를 그대로 남기면 실험 기록이 수백 KB가 된다. 저장소에
    두기에 크고, 정작 필요한 것은 분포다. 0.001 눈금은 288px 영상에서 0.3픽셀
    이므로 이 단계에서 주장하는 어떤 수치도 이 해상도에서 달라지지 않는다.

    **평균이나 중앙값을 미리 계산해 두지 않는다.** 여러 시퀀스를 합칠 때
    시퀀스별 중앙값의 평균은 전체 중앙값이 아니기 때문이다. 히스토그램을
    더한 뒤 다시 계산해야 한다.
    """
    out: dict[str, int] = {}
    for v in values:
        key = f"{round(v / HEIGHT_RESOLUTION) * HEIGHT_RESOLUTION:.3f}"
        out[key] = out.get(key, 0) + 1
    return out


def histogram_median(histogram: dict[str, int]) -> float | None:
    """히스토그램에서 중앙값. 표본이 없으면 None."""
    total = sum(histogram.values())
    if total == 0:
        return None
    keys = sorted(float(k) for k in histogram)
    # 짝수 개면 가운데 두 값의 평균. 통계 관행을 따른다.
    targets = (
        [total // 2] if total % 2 else [total // 2 - 1, total // 2]
    )
    picked: list[float] = []
    seen = 0
    for key in keys:
        count = histogram[f"{key:.3f}"]
        while picked.__len__() < len(targets) and targets[len(picked)] < seen + count:
            picked.append(key)
        seen += count
        if len(picked) == len(targets):
            break
    return sum(picked) / len(picked)


def histogram_fraction_below(histogram: dict[str, int], cut: float) -> float | None:
    total = sum(histogram.values())
    if total == 0:
        return None
    return sum(n for k, n in histogram.items() if float(k) < cut) / total


def histogram_min(histogram: dict[str, int]) -> float | None:
    return min((float(k) for k in histogram), default=None)
