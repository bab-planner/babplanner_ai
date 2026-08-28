"""전역 할당(global assignment).

**왜 greedy로는 부족한가.** 기존 tracker는 IoU가 큰 쌍부터 하나씩 확정한다
(`tracking.greedy_match`). 빠르고 결정적이지만 국소적이다. 두 사람이 스쳐
지나가는 순간을 생각하면 이렇게 된다.

        track A ─ det 1   IoU 0.55        track A ─ det 2   IoU 0.50
        track B ─ det 1   IoU 0.52        track B ─ det 2   IoU 0.10

    greedy   A-1(0.55)을 먼저 확정 -> B에게는 det 2(0.10)만 남는다.
             IoU 임계값에 걸려 B는 짝을 못 찾고 끊긴다. 합 0.55.
    전역     A-2(0.50) + B-1(0.52) = 1.02. 둘 다 살아남는다.

급식소 통로처럼 사람이 서로 겹쳐 지나가는 장면에서 이 차이가 track 끊김으로
나타나고, 끊긴 track은 새 번호를 받아 통과 계수를 틀리게 만든다.

**scipy를 쓰지 않는다.** `.venv-cctv`에 scipy가 없고, 이 패키지의 계수
로직은 표준 라이브러리만으로 돌아야 한다는 제약이 A2부터 있다(무거운 의존성
없이 전체를 테스트하기 위해서다). 그래서 Hungarian(Kuhn-Munkres)의
증가경로(shortest augmenting path) 변형을 여기 직접 구현한다. O(n^2 m).

이 파일은 tracker를 모른다. 비용 행렬만 받는다. 그래야 할당 자체를 tracker
없이 테스트할 수 있다.
"""

from __future__ import annotations

from collections.abc import Sequence

INF = float("inf")


class AssignmentError(ValueError):
    """할당을 계산할 수 없다."""


def linear_sum_assignment(
    cost: Sequence[Sequence[float]],
) -> list[tuple[int, int]]:
    """비용 합이 최소가 되는 1:1 짝짓기.

    (row, col) 쌍을 row 오름차순으로 돌려준다. 행과 열 수가 달라도 되고,
    적은 쪽 개수만큼 짝이 나온다.

    **결정적이다.** 같은 비용이 여러 개면 index가 작은 쪽이 이긴다. 실험을
    재현하려면 같은 입력에 항상 같은 출력이 나와야 한다.
    """
    rows = len(cost)
    if rows == 0:
        return []
    cols = len(cost[0])
    for index, row in enumerate(cost):
        if len(row) != cols:
            raise AssignmentError(
                f"{index}번째 행의 길이가 다릅니다: {len(row)} != {cols}"
            )
        for value in row:
            if value != value or value == INF or value == -INF:
                raise AssignmentError(f"비용이 유한하지 않습니다: {value}")
    if cols == 0:
        return []

    # 알고리즘이 행 <= 열을 요구한다. 반대면 뒤집어 풀고 결과를 되돌린다.
    if rows > cols:
        transposed = [[cost[r][c] for r in range(rows)] for c in range(cols)]
        return sorted(
            (r, c) for c, r in linear_sum_assignment(transposed)
        )

    n, m = rows, cols
    # u, v는 잠재값(potential). p[j]는 열 j에 붙은 행(1-based, 0이면 없음).
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 == 0:
                raise AssignmentError("증가경로를 찾지 못했습니다.")
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    return sorted((p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0)


def match_by_iou(
    predicted_boxes: Sequence,
    detection_boxes: Sequence,
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """IoU를 최대화하는 전역 1:1 매칭.

    `tracking.greedy_match`와 **같은 모양의 결과**를 돌려준다. 그래야 두
    tracker를 같은 자리에서 바꿔 끼우고 비교할 수 있다.

    IoU가 임계값 미만인 쌍은 짝으로 인정하지 않는다. 할당 자체는 전역으로
    풀되, 푼 뒤에 임계값으로 거른다. 임계값 미만 쌍에 큰 비용을 주어 푸는
    것과 결과가 같고, 이쪽이 읽기 쉽다.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise AssignmentError(f"iou_threshold는 0~1이어야 합니다: {iou_threshold}")
    if not predicted_boxes or not detection_boxes:
        return [], list(range(len(predicted_boxes))), list(range(len(detection_boxes)))

    ious = [
        [predicted.iou(detection) for detection in detection_boxes]
        for predicted in predicted_boxes
    ]
    cost = [[1.0 - value for value in row] for row in ious]
    pairs = linear_sum_assignment(cost)

    matches = [
        (r, c) for r, c in pairs if ious[r][c] >= iou_threshold and ious[r][c] > 0.0
    ]
    matched_rows = {r for r, _ in matches}
    matched_cols = {c for _, c in matches}
    return (
        matches,
        [i for i in range(len(predicted_boxes)) if i not in matched_rows],
        [j for j in range(len(detection_boxes)) if j not in matched_cols],
    )
