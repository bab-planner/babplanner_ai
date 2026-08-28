"""W3 그림. W2와 같이 의존성 없이 SVG를 직접 쓴다.

세 장이다.

    1. 위험-coverage 곡선 — 무작위 대조와 oracle 상한을 같이 그린다.
       oracle은 실행할 수 없는 선이므로 점선으로 두고 이름에 적는다.
    2. 신뢰 점수 대 절대 오차 — 관계가 얼마나 헐거운지 그대로 보인다.
    3. coverage별 고잔반 거절률 — MAE가 내려간 이유가 구성 변화인지 본다.

    .venv-waste\\Scripts\\python.exe research\\waste_estimation\\scripts\\w3_charts.py
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
LINE = "#2b6cb0"
LINE_ALT = "#c05621"
LINE_ORACLE = "#7b8794"
LINE_ANALYSIS = "#6b46c1"
POINT = "#2f855a"

OUT = Path("research/waste_estimation/experiments/w3_selective")
ASSETS = Path("reports/assets")

COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.5]


def header(title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="system-ui, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{PAD_L}" y="28" font-size="16" font-weight="600" fill="{FG}">{title}</text>',
        f'<text x="{PAD_L}" y="46" font-size="12" fill="{MUTED}">{subtitle}</text>',
    ]


def _axes(parts: list[str], top: float, ylabel: str, xlabel: str) -> tuple[float, float]:
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        y = PAD_T + plot_h * (1 - fraction)
        parts.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" stroke="{GRID}"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" '
            f'text-anchor="end">{top * fraction:.2f}</text>'
        )
    for index, coverage in enumerate(COVERAGES):
        x = PAD_L + plot_w * index / (len(COVERAGES) - 1)
        parts.append(
            f'<text x="{x:.1f}" y="{PAD_T + plot_h + 18:.1f}" font-size="11" '
            f'fill="{MUTED}" text-anchor="middle">{coverage:.0%}</text>'
        )
    parts.append(
        f'<text x="{PAD_L + plot_w / 2:.1f}" y="{H - 14}" font-size="11" fill="{MUTED}" '
        f'text-anchor="middle">{xlabel}</text>'
    )
    parts.append(
        f'<text x="16" y="{PAD_T + plot_h / 2:.1f}" font-size="11" fill="{MUTED}" '
        f'transform="rotate(-90 16 {PAD_T + plot_h / 2:.1f})" text-anchor="middle">{ylabel}</text>'
    )
    return plot_w, plot_h


def _series(
    parts: list[str],
    values: list[float | None],
    top: float,
    plot_w: float,
    plot_h: float,
    color: str,
    dashed: bool = False,
) -> None:
    points = []
    for index, value in enumerate(values):
        if value is None:
            continue
        x = PAD_L + plot_w * index / (len(COVERAGES) - 1)
        y = PAD_T + plot_h * (1 - value / top)
        points.append((x, y))
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
    parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
    for x, y in points:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{color}"/>')


def _legend(parts: list[str], entries: list[tuple[str, str, bool]]) -> None:
    x = PAD_L
    y = H - 34
    for label, color, dashed in entries:
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        parts.append(
            f'<line x1="{x}" y1="{y}" x2="{x + 22}" y2="{y}" stroke="{color}" '
            f'stroke-width="2"{dash}/>'
        )
        parts.append(
            f'<text x="{x + 28}" y="{y + 4}" font-size="11" fill="{FG}">{label}</text>'
        )
        x += 34 + len(label) * 6.6


def risk_coverage_chart(dev: list[dict], analysis: list[dict], path: Path) -> None:
    """1. 위험-coverage 곡선."""

    def pick(rows: list[dict], signal: str) -> list[float | None]:
        out: list[float | None] = []
        for coverage in COVERAGES:
            match = next(
                (
                    r
                    for r in rows
                    if r["signal"] == signal and abs(float(r["coverage"]) - coverage) < 1e-9
                ),
                None,
            )
            out.append(None if match is None or not match["risk"] else float(match["risk"]))
        return out

    series = [
        ("DEVELOPMENT OOF · C0_knn3", pick(dev, "C0_knn3"), LINE, False),
        ("무작위 거절", pick(dev, "random_baseline"), LINE_ALT, False),
        ("oracle 상한 (실행 불가)", pick(dev, "oracle_upper_bound"), LINE_ORACLE, True),
        ("W2 분석 집합 (시험 아님)", pick(analysis, "C0_knn3"), LINE_ANALYSIS, False),
    ]
    top = 0.20
    parts = header(
        "위험-coverage 곡선 — 승인 표본 MAE",
        "coverage를 줄이면 승인 MAE가 내려간다 · oracle은 사후에만 그릴 수 있는 선이다",
    )
    plot_w, plot_h = _axes(parts, top, "승인 표본 MAE", "coverage (승인 비율)")
    for _, values, color, dashed in series:
        _series(parts, values, top, plot_w, plot_h, color, dashed)
    _legend(parts, [(label, color, dashed) for label, _, color, dashed in series])
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def scatter_chart(rows: list[dict], path: Path) -> None:
    """2. 신뢰 점수 대 절대 오차."""
    scores = [float(r["C0_knn3"]) for r in rows]
    errors = [float(r["absError"]) for r in rows]
    x_top = max(scores) * 1.05
    y_top = max(errors) * 1.05

    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B
    parts = header(
        "신뢰 점수 대 절대 오차 (DEVELOPMENT OOF, n=%d)" % len(rows),
        "Spearman +0.43 · 방향은 맞지만 흩어짐이 크다 · 낮은 점수에도 큰 오차가 있다",
    )
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        y = PAD_T + plot_h * (1 - fraction)
        parts.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" stroke="{GRID}"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" '
            f'text-anchor="end">{y_top * fraction:.2f}</text>'
        )
        parts.append(
            f'<text x="{PAD_L + plot_w * fraction:.1f}" y="{PAD_T + plot_h + 18:.1f}" '
            f'font-size="11" fill="{MUTED}" text-anchor="middle">{x_top * fraction:.2f}</text>'
        )
    for score, error in zip(scores, errors):
        x = PAD_L + plot_w * score / x_top
        y = PAD_T + plot_h * (1 - error / y_top)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.0" fill="{POINT}" opacity="0.45"/>'
        )
    parts.append(
        f'<text x="{PAD_L + plot_w / 2:.1f}" y="{H - 14}" font-size="11" fill="{MUTED}" '
        f'text-anchor="middle">신뢰 점수 C0_knn3 (높을수록 덜 믿을 만함)</text>'
    )
    parts.append(
        f'<text x="16" y="{PAD_T + plot_h / 2:.1f}" font-size="11" fill="{MUTED}" '
        f'transform="rotate(-90 16 {PAD_T + plot_h / 2:.1f})" text-anchor="middle">절대 오차</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def high_waste_chart(subgroup: list[dict], path: Path) -> None:
    """3. coverage별 고잔반 거절률."""
    values = []
    for coverage in COVERAGES:
        match = next(
            r
            for r in subgroup
            if r["signal"] == "C0_knn3" and abs(float(r["coverage"]) - coverage) < 1e-9
        )
        values.append((float(match["highRejectionRate"]), 1.0 - coverage))

    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B
    parts = header(
        "coverage별 고잔반 표본 거절률",
        "고잔반이 전체보다 빨리 빠진다 · MAE 하락의 일부는 구성 변화다",
    )
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        y = PAD_T + plot_h * (1 - fraction)
        parts.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" stroke="{GRID}"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 8}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" '
            f'text-anchor="end">{fraction:.0%}</text>'
        )
    width = plot_w / len(COVERAGES)
    for index, (high_rate, overall_rate) in enumerate(values):
        x = PAD_L + index * width
        for offset, (rate, color) in enumerate(
            ((high_rate, LINE_ALT), (overall_rate, LINE))
        ):
            bar_w = width / 2 - 10
            bar_x = x + 8 + offset * (bar_w + 6)
            height = plot_h * rate
            parts.append(
                f'<rect x="{bar_x:.1f}" y="{PAD_T + plot_h - height:.1f}" '
                f'width="{bar_w:.1f}" height="{height:.1f}" fill="{color}" opacity="0.85"/>'
            )
            if rate:
                parts.append(
                    f'<text x="{bar_x + bar_w / 2:.1f}" y="{PAD_T + plot_h - height - 5:.1f}" '
                    f'font-size="10" fill="{FG}" text-anchor="middle">{rate:.0%}</text>'
                )
        parts.append(
            f'<text x="{x + width / 2:.1f}" y="{PAD_T + plot_h + 18:.1f}" font-size="11" '
            f'fill="{MUTED}" text-anchor="middle">{COVERAGES[index]:.0%}</text>'
        )
    parts.append(
        f'<text x="{PAD_L + plot_w / 2:.1f}" y="{H - 14}" font-size="11" fill="{MUTED}" '
        f'text-anchor="middle">coverage (승인 비율)</text>'
    )
    _legend(parts, [("고잔반 거절률", LINE_ALT, False), ("전체 거절률", LINE, False)])
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

    risk_coverage_chart(
        read_csv(args.results / "coverage_risk.csv"),
        read_csv(args.results / "analysis_coverage_risk.csv"),
        args.assets / "waste_w3_risk_coverage.svg",
    )
    scatter_chart(
        read_csv(args.results / "oof_predictions.csv"),
        args.assets / "waste_w3_reliability_error.svg",
    )
    high_waste_chart(
        read_csv(args.results / "high_waste_selective.csv"),
        args.assets / "waste_w3_high_waste.svg",
    )
    print(f"그림 3장: {args.assets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
