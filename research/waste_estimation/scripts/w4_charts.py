"""W4 그림. W2·W3과 같이 의존성 없이 SVG를 직접 쓴다.

세 장이다.

    1. 전체 MAE 대 고잔반 MAE — 맞교환이 한 장에 보여야 한다.
       Gate 1 선을 같이 그린다. 왜 R1이 탈락했는지가 그림에서 읽힌다.
    2. 고잔반 표본의 정답 대 예측 — 평균으로 수축하는 모양을 직접 본다.
    3. raw 대 normalized coverage 이동 — W3이 놓친 자리.

원본 음식 사진은 만들지 않는다. 여기 나오는 것은 집계 수치뿐이다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w4_charts.py
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
POINT = "#2f855a"
POINT_ALT = "#c05621"

OUT = Path("research/waste_estimation/experiments/w4_tail")
ASSETS = Path("reports/assets")

LABELS = {
    "R0_STANDARD": "R0 표준",
    "R1_BALANCED": "R1 구간균형",
    "R2_HIGH_AUX": "R2 보조분류기",
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
    x = PAD_L
    y = H - 20
    for label, color in entries:
        parts.append(
            f'<rect x="{x}" y="{y - 9}" width="12" height="12" fill="{color}" opacity="0.85"/>'
        )
        parts.append(f'<text x="{x + 18}" y="{y + 1}" font-size="11" fill="{FG}">{label}</text>')
        x += 30 + len(label) * 7.2


def tradeoff_chart(rows: list[dict], path: Path) -> None:
    """1. 전체 MAE 대 고잔반 MAE."""
    order = ["R0_STANDARD", "R1_BALANCED", "R2_HIGH_AUX"]
    values = [
        (LABELS[name], float(row["mae"]), float(row["highMae"]))
        for name in order
        for row in rows
        if row["candidate"] == name
    ]
    baseline = next(float(row["mae"]) for row in rows if row["candidate"] == "R0_STANDARD")
    gate = baseline + 0.02
    top = 0.32

    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B
    parts = header(
        "전체 MAE 대 고잔반 MAE (DEVELOPMENT 중첩 group CV)",
        "R1은 꼬리를 줄였지만 전체가 Gate 1을 넘어 탈락했다 · 낮을수록 좋다",
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
    slot = plot_w / len(values)
    for index, (label, overall, tail) in enumerate(values):
        base_x = PAD_L + index * slot
        for offset, (value, color) in enumerate(((overall, BAR), (tail, BAR_ALT))):
            width = slot / 2 - 16
            x = base_x + 12 + offset * (width + 8)
            height = plot_h * value / top
            parts.append(
                f'<rect x="{x:.1f}" y="{PAD_T + plot_h - height:.1f}" width="{width:.1f}" '
                f'height="{height:.1f}" fill="{color}" opacity="0.85"/>'
            )
            parts.append(
                f'<text x="{x + width / 2:.1f}" y="{PAD_T + plot_h - height - 5:.1f}" '
                f'font-size="10" fill="{FG}" text-anchor="middle">{value:.3f}</text>'
            )
        parts.append(
            f'<text x="{base_x + slot / 2:.1f}" y="{PAD_T + plot_h + 18:.1f}" font-size="11" '
            f'fill="{MUTED}" text-anchor="middle">{label}</text>'
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


def high_waste_scatter(rows: list[dict], path: Path) -> None:
    """2. 고잔반 표본의 정답 대 예측."""
    high = [row for row in rows if row["isHighWaste"] == "1"]
    plot = W - PAD_L - PAD_R
    size = min(plot, H - PAD_T - PAD_B)
    left, top_y = PAD_L, PAD_T

    parts = header(
        f"고잔반 표본의 정답 대 예측 (W2 분석 집합, n={len(high)})",
        "대각선에서 아래로 멀수록 과소추정 · 회색 띠가 치명적 과소추정 구간",
    )
    parts.append(
        f'<rect x="{left}" y="{top_y}" width="{size}" height="{size}" fill="none" stroke="{GRID}"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top_y + size}" x2="{left + size}" y2="{top_y}" '
        f'stroke="{MUTED}" stroke-dasharray="4 4"/>'
    )
    # 정답 − 예측 >= 0.30 인 영역
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
    for column, color in (("R0_STANDARD", POINT), ("R1_BALANCED", POINT_ALT)):
        for row in high:
            target = min(max(float(row["target"]), 0.0), 1.0)
            prediction = min(max(float(row[column]), 0.0), 1.0)
            x = left + size * target
            y = top_y + size * (1 - prediction)
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{color}" opacity="0.55"/>'
            )
    parts.append(
        f'<text x="{left + size / 2:.1f}" y="{H - 34}" font-size="11" fill="{MUTED}" '
        f'text-anchor="middle">정답 external_waste_ratio</text>'
    )
    legend(parts, [("R0 표준", POINT), ("R1 구간균형", POINT_ALT)])
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def drift_chart(rows: list[dict], path: Path) -> None:
    """3. raw 대 normalized coverage 이동."""
    targets = [0.9, 0.8, 0.7, 0.5]
    top = 0.14
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B
    parts = header(
        "coverage 이동 — 참조 집합이 159건에서 318건으로 커질 때",
        "|실현 − 목표| · 낮을수록 임계값이 잘 옮겨진다 · W3에서 재지 않았던 값",
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
    slot = plot_w / len(targets)
    for index, coverage in enumerate(targets):
        base_x = PAD_L + index * slot
        for offset, (method, color) in enumerate(
            (("C0_RAW", BAR_ALT), ("C0_NORMALIZED", BAR))
        ):
            row = next(
                r
                for r in rows
                if r["method"] == method
                and abs(float(r["targetCoverage"]) - coverage) < 1e-9
            )
            value = float(row["meanCoverageDrift"])
            width = slot / 2 - 16
            x = base_x + 12 + offset * (width + 8)
            height = plot_h * min(value, top) / top
            parts.append(
                f'<rect x="{x:.1f}" y="{PAD_T + plot_h - height:.1f}" width="{width:.1f}" '
                f'height="{height:.1f}" fill="{color}" opacity="0.85"/>'
            )
            parts.append(
                f'<text x="{x + width / 2:.1f}" y="{PAD_T + plot_h - height - 5:.1f}" '
                f'font-size="10" fill="{FG}" text-anchor="middle">{value:.3f}</text>'
            )
        parts.append(
            f'<text x="{base_x + slot / 2:.1f}" y="{PAD_T + plot_h + 18:.1f}" font-size="11" '
            f'fill="{MUTED}" text-anchor="middle">목표 {coverage:.0%}</text>'
        )
    legend(parts, [("C0_RAW (W3 방식)", BAR_ALT), ("C0_NORMALIZED", BAR)])
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

    tradeoff_chart(
        read_csv(args.results / "candidate_summary.csv"),
        args.assets / "waste_w4_tradeoff.svg",
    )
    high_waste_scatter(
        read_csv(args.results / "failure_cases.csv"),
        args.assets / "waste_w4_high_waste_scatter.svg",
    )
    drift_chart(
        read_csv(args.results / "reliability_transport_summary.csv"),
        args.assets / "waste_w4_coverage_drift.svg",
    )
    print(f"그림 3장: {args.assets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
