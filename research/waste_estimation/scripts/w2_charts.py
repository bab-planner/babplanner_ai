"""W2 그림. 의존성 없이 SVG를 직접 쓴다.

CCTV 트랙과 같은 방식이다. matplotlib을 들이지 않는다 — 그림 세 장 때문에
연구 환경에 의존성을 더할 이유가 없고, SVG는 텍스트라 저장소에서 diff가
읽힌다.

**음식 사진을 만들지 않는다.** 여기서 나오는 것은 집계 수치뿐이고, 원본
이미지는 재배포 조건과 무관하게 저장소에 들어가지 않는다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w2_charts.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

W, H = 720, 400
PAD_L, PAD_R, PAD_T, PAD_B = 78, 26, 58, 60

FG = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
BAR = "#2b6cb0"
BAR_ALT = "#c05621"
POINT = "#2f855a"

OUT = Path("research/waste_estimation/experiments/w2_external")
ASSETS = Path("reports/assets")


def header(title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="system-ui, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{PAD_L}" y="28" font-size="16" font-weight="600" fill="{FG}">{title}</text>',
        f'<text x="{PAD_L}" y="46" font-size="12" fill="{MUTED}">{subtitle}</text>',
    ]


def distribution_chart(rows: list[dict], path: Path) -> None:
    """1. 정답 잔반률 분포."""
    edges = [i / 10 for i in range(10)]
    counts = [0] * 10
    for row in rows:
        value = float(row["target"])
        counts[min(int(value * 10), 9)] += 1
    top = max(counts) or 1

    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B
    parts = header(
        "외부 벤치마크 정답 분포 (LeFood-Set TEST)",
        f"external_waste_ratio · 표본 {len(rows)}건 · 양 끝에 몰려 있다",
    )
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        y = PAD_T + plot_h * (1 - fraction)
        parts.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" stroke="{GRID}"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" '
            f'text-anchor="end">{int(top * fraction)}</text>'
        )
    width = plot_w / 10
    for index, count in enumerate(counts):
        height = plot_h * count / top
        x = PAD_L + index * width
        y = PAD_T + plot_h - height
        parts.append(
            f'<rect x="{x + 3:.1f}" y="{y:.1f}" width="{width - 6:.1f}" '
            f'height="{height:.1f}" fill="{BAR}" opacity="0.85"/>'
        )
        parts.append(
            f'<text x="{x + width / 2:.1f}" y="{PAD_T + plot_h + 18:.1f}" font-size="10" '
            f'fill="{MUTED}" text-anchor="middle">{edges[index]:.1f}</text>'
        )
        if count:
            parts.append(
                f'<text x="{x + width / 2:.1f}" y="{y - 5:.1f}" font-size="10" '
                f'fill="{FG}" text-anchor="middle">{count}</text>'
            )
    parts.append(
        f'<text x="{PAD_L + plot_w / 2:.1f}" y="{H - 16}" font-size="11" fill="{MUTED}" '
        f'text-anchor="middle">external_waste_ratio</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def mae_chart(rows: list[dict], path: Path) -> None:
    """2. 모델별 TEST MAE."""
    wanted = [
        ("B0 전역 평균", "B0_global_mean", "-", "raw"),
        ("B1 단순 통계 (식후)", "B1_simple", "after", "raw"),
        ("B2 식후만", "B2_frozen", "after", "raw"),
        ("B2 식전+식후", "B2_frozen", "concat", "raw"),
        ("B2 식전+식후+차", "B2_frozen", "concat_diff", "raw"),
    ]
    values = []
    for label, model, arm, output in wanted:
        match = next(
            (
                r
                for r in rows
                if r["model"] == model and r["arm"] == arm and r["output"] == output
            ),
            None,
        )
        if match:
            values.append((label, float(match["mae"])))

    top = max(value for _, value in values) * 1.15
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B
    parts = header(
        "TEST MAE — 낮을수록 좋다",
        "처음 보는 음식 범주 10개 · raw 회귀 출력 (자르지 않음)",
    )
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        y = PAD_T + plot_h * (1 - fraction)
        parts.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" stroke="{GRID}"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" '
            f'text-anchor="end">{top * fraction:.2f}</text>'
        )
    width = plot_w / len(values)
    for index, (label, value) in enumerate(values):
        height = plot_h * value / top
        x = PAD_L + index * width
        y = PAD_T + plot_h - height
        colour = BAR_ALT if index == 0 else BAR
        parts.append(
            f'<rect x="{x + 12:.1f}" y="{y:.1f}" width="{width - 24:.1f}" '
            f'height="{height:.1f}" fill="{colour}" opacity="0.9"/>'
        )
        parts.append(
            f'<text x="{x + width / 2:.1f}" y="{y - 6:.1f}" font-size="12" '
            f'fill="{FG}" text-anchor="middle" font-weight="600">{value:.3f}</text>'
        )
        for line_index, chunk in enumerate(label.split(" ", 1)):
            parts.append(
                f'<text x="{x + width / 2:.1f}" y="{PAD_T + plot_h + 18 + line_index * 14:.1f}" '
                f'font-size="10" fill="{MUTED}" text-anchor="middle">{chunk}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def scatter_chart(rows: list[dict], column: str, path: Path) -> None:
    """3. 정답 대 예측."""
    plot = W - PAD_L - PAD_R
    size = min(plot, H - PAD_T - PAD_B)
    left = PAD_L
    top_y = PAD_T

    parts = header(
        "정답 대 예측 (식후 이미지만)",
        "대각선에 가까울수록 좋다 · 높은 잔반에서 평균 쪽으로 끌린다",
    )
    parts.append(
        f'<rect x="{left}" y="{top_y}" width="{size}" height="{size}" fill="none" stroke="{GRID}"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top_y + size}" x2="{left + size}" y2="{top_y}" '
        f'stroke="{MUTED}" stroke-dasharray="4 4"/>'
    )
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<text x="{left - 8}" y="{top_y + size * (1 - fraction) + 4:.1f}" font-size="11" '
            f'fill="{MUTED}" text-anchor="end">{fraction:.2f}</text>'
        )
        parts.append(
            f'<text x="{left + size * fraction:.1f}" y="{top_y + size + 18:.1f}" font-size="11" '
            f'fill="{MUTED}" text-anchor="middle">{fraction:.2f}</text>'
        )
    for row in rows:
        target = float(row["target"])
        prediction = float(row[column])
        x = left + size * min(max(target, 0.0), 1.0)
        y = top_y + size * (1 - min(max(prediction, 0.0), 1.0))
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{POINT}" opacity="0.5"/>'
        )
    parts.append(
        f'<text x="{left + size / 2:.1f}" y="{H - 12}" font-size="11" fill="{MUTED}" '
        f'text-anchor="middle">정답 external_waste_ratio</text>'
    )
    parts.append(
        f'<text x="16" y="{top_y + size / 2:.1f}" font-size="11" fill="{MUTED}" '
        f'transform="rotate(-90 16 {top_y + size / 2:.1f})" text-anchor="middle">예측</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=OUT)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    args = parser.parse_args(argv)
    args.assets.mkdir(parents=True, exist_ok=True)

    predictions = read_csv(args.results / "test_predictions.csv")
    metrics = read_csv(args.results / "regression_metrics.csv")

    distribution_chart(predictions, args.assets / "waste_w2_distribution.svg")
    mae_chart(metrics, args.assets / "waste_w2_mae.svg")
    scatter_chart(predictions, "B2_after", args.assets / "waste_w2_scatter.svg")

    print(f"그림 3장: {args.assets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
