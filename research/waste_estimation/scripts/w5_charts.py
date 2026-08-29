"""W5 그림. 앞 단계와 같이 의존성 없이 SVG를 직접 쓴다.

세 장이다.

    1. 후보별 전체 MAE 대 고잔반 MAE — W4에서는 이 둘이 맞교환이었다.
    2. 치명적 과소추정 비율 — 이번 선택의 주 기준.
    3. 고잔반 표본의 정답 대 예측 — 평균 수축이 실제로 줄었는지.

원본 음식 사진은 만들지 않는다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w5_charts.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

W, H = 720, 400
PAD_L, PAD_R, PAD_T, PAD_B = 78, 26, 58, 62

FG = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
BAR = "#2b6cb0"
BAR_ALT = "#c05621"
GATE = "#b91c1c"
POINT = "#7b8794"
POINT_ALT = "#2f855a"

OUT = Path("research/waste_estimation/experiments/w5_structured")
ASSETS = Path("reports/assets")

ORDER = (
    "S0_STANDARD_RIDGE",
    "S1_CUMULATIVE_ORDINAL_EXPECTATION",
    "S2_SOFT_HIGH_LOW_MIXTURE",
)
LABELS = {
    "S0_STANDARD_RIDGE": "S0 표준 회귀",
    "S1_CUMULATIVE_ORDINAL_EXPECTATION": "S1 누적 순서형",
    "S2_SOFT_HIGH_LOW_MIXTURE": "S2 부드러운 혼합",
}


def header(title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="system-ui, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{PAD_L}" y="28" font-size="16" font-weight="600" fill="{FG}">{title}</text>',
        f'<text x="{PAD_L}" y="46" font-size="12" fill="{MUTED}">{subtitle}</text>',
    ]


def legend(parts: list[str], entries: list[tuple[str, str]]) -> None:
    x, y = PAD_L, H - 20
    for label, color in entries:
        parts.append(
            f'<rect x="{x}" y="{y - 9}" width="12" height="12" fill="{color}" opacity="0.85"/>'
        )
        parts.append(f'<text x="{x + 18}" y="{y + 1}" font-size="11" fill="{FG}">{label}</text>')
        x += 30 + len(label) * 7.2


def _grid(parts: list[str], top: float, fmt: str) -> tuple[float, float]:
    plot_w, plot_h = W - PAD_L - PAD_R, H - PAD_T - PAD_B
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        y = PAD_T + plot_h * (1 - fraction)
        parts.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" stroke="{GRID}"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" '
            f'text-anchor="end">{format(top * fraction, fmt)}</text>'
        )
    return plot_w, plot_h


def _bars(
    parts: list[str],
    values: list[tuple[str, list[tuple[float, str]]]],
    top: float,
    plot_w: float,
    plot_h: float,
    fmt: str,
) -> None:
    slot = plot_w / len(values)
    for index, (label, series) in enumerate(values):
        base_x = PAD_L + index * slot
        width = (slot - 24) / len(series) - 6
        for offset, (value, color) in enumerate(series):
            x = base_x + 12 + offset * (width + 6)
            height = plot_h * min(value, top) / top
            parts.append(
                f'<rect x="{x:.1f}" y="{PAD_T + plot_h - height:.1f}" width="{width:.1f}" '
                f'height="{height:.1f}" fill="{color}" opacity="0.85"/>'
            )
            parts.append(
                f'<text x="{x + width / 2:.1f}" y="{PAD_T + plot_h - height - 5:.1f}" '
                f'font-size="10" fill="{FG}" text-anchor="middle">{format(value, fmt)}</text>'
            )
        parts.append(
            f'<text x="{base_x + slot / 2:.1f}" y="{PAD_T + plot_h + 18:.1f}" font-size="11" '
            f'fill="{MUTED}" text-anchor="middle">{label}</text>'
        )


def tradeoff_chart(rows: list[dict], path: Path) -> None:
    """1. 전체 MAE 대 고잔반 MAE."""
    bounded = {
        row["candidate"]: row for row in rows if row["output"] == "bounded"
    }
    baseline = float(bounded["S0_STANDARD_RIDGE"]["mae"])
    gate = baseline + 0.02
    top = 0.28

    parts = header(
        "전체 MAE 대 고잔반 MAE (DEVELOPMENT group CV, 자른 예측)",
        "W4에서는 이 둘이 맞교환이었다 · 여기서는 둘 다 내려간다 · 낮을수록 좋다",
    )
    plot_w, plot_h = _grid(parts, top, ".2f")
    _bars(
        parts,
        [
            (
                LABELS[name],
                [
                    (float(bounded[name]["mae"]), BAR),
                    (float(bounded[name]["highMae"]), BAR_ALT),
                ],
            )
            for name in ORDER
        ],
        top,
        plot_w,
        plot_h,
        ".3f",
    )
    gate_y = PAD_T + plot_h * (1 - gate / top)
    parts.append(
        f'<line x1="{PAD_L}" y1="{gate_y:.1f}" x2="{W - PAD_R}" y2="{gate_y:.1f}" '
        f'stroke="{GATE}" stroke-width="1.5" stroke-dasharray="6 4"/>'
    )
    parts.append(
        f'<text x="{W - PAD_R}" y="{gate_y - 6:.1f}" font-size="11" fill="{GATE}" '
        f'text-anchor="end">Gate 1 상한 {gate:.3f}</text>'
    )
    legend(parts, [("전체 MAE", BAR), ("고잔반 MAE", BAR_ALT)])
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def catastrophic_chart(
    development: list[dict], analysis: list[dict], path: Path
) -> None:
    """2. 치명적 과소추정 비율."""
    dev = {row["candidate"]: row for row in development if row["output"] == "bounded"}
    ana = {row["candidate"]: row for row in analysis if row["output"] == "bounded"}
    top = 0.50

    parts = header(
        "치명적 과소추정 비율 — 이번 선택의 주 기준",
        "정답 >= 0.8 이고 정답 − 예측 >= 0.30 · 분모는 고잔반 표본 수 · pooled",
    )
    plot_w, plot_h = _grid(parts, top, ".0%")
    _bars(
        parts,
        [
            (
                LABELS[name],
                [
                    (float(dev[name]["catastrophicRatePooled"]), BAR),
                    (float(ana[name]["catastrophicRatePooled"]), BAR_ALT),
                ],
            )
            for name in ORDER
        ],
        top,
        plot_w,
        plot_h,
        ".1%",
    )
    legend(
        parts,
        [
            ("DEVELOPMENT (90건)", BAR),
            ("W2 분석 집합 (34건, 시험 아님)", BAR_ALT),
        ],
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def high_waste_scatter(rows: list[dict], path: Path) -> None:
    """3. 고잔반 표본의 정답 대 예측."""
    high = [row for row in rows if row["isHighWaste"] == "1"]
    plot = W - PAD_L - PAD_R
    size = min(plot, H - PAD_T - PAD_B)
    left, top_y = PAD_L, PAD_T

    parts = header(
        f"고잔반 표본의 정답 대 예측 (W2 분석 집합, n={len(high)})",
        "대각선에서 아래로 멀수록 과소추정 · 붉은 띠가 치명적 구간 · 위로 올라왔지만 남아 있다",
    )
    parts.append(
        f'<rect x="{left}" y="{top_y}" width="{size}" height="{size}" fill="none" stroke="{GRID}"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top_y + size}" x2="{left + size}" y2="{top_y}" '
        f'stroke="{MUTED}" stroke-dasharray="4 4"/>'
    )
    offset = size * 0.30
    parts.append(
        f'<path d="M{left},{top_y + size} L{left + size},{top_y} '
        f'L{left + size},{top_y + offset:.1f} L{left},{top_y + size} Z" '
        f'fill="{GATE}" opacity="0.07"/>'
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
    for row in high:
        target = min(max(float(row["target"]), 0.0), 1.0)
        before = min(max(float(row["S0_bounded"]), 0.0), 1.0)
        after = min(max(float(row["selected_bounded"]), 0.0), 1.0)
        x = left + size * target
        parts.append(
            f'<line x1="{x:.1f}" y1="{top_y + size * (1 - before):.1f}" '
            f'x2="{x:.1f}" y2="{top_y + size * (1 - after):.1f}" '
            f'stroke="{MUTED}" stroke-width="1" opacity="0.4"/>'
        )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{top_y + size * (1 - before):.1f}" r="3.2" '
            f'fill="{POINT}" opacity="0.55"/>'
        )
        parts.append(
            f'<circle cx="{x:.1f}" cy="{top_y + size * (1 - after):.1f}" r="3.2" '
            f'fill="{POINT_ALT}" opacity="0.65"/>'
        )
    parts.append(
        f'<text x="{left + size / 2:.1f}" y="{H - 34}" font-size="11" fill="{MUTED}" '
        f'text-anchor="middle">정답 external_waste_ratio</text>'
    )
    legend(parts, [("S0 표준 회귀", POINT), ("S2 부드러운 혼합", POINT_ALT)])
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

    development = read_csv(args.results / "candidate_metrics.csv")
    analysis = read_csv(args.results / "analysis_metrics.csv")
    tradeoff_chart(development, args.assets / "waste_w5_tradeoff.svg")
    catastrophic_chart(
        development, analysis, args.assets / "waste_w5_catastrophic.svg"
    )
    high_waste_scatter(
        read_csv(args.results / "catastrophic_recovery.csv"),
        args.assets / "waste_w5_high_waste_scatter.svg",
    )
    print(f"그림 3장: {args.assets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
