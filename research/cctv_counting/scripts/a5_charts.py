"""A5 결과 그림. 의존성 없이 SVG를 직접 쓴다.

matplotlib을 새로 들이지 않는다. 그림 두 장 때문에 연구 환경에 의존성을
더할 이유가 없고, SVG는 텍스트라 저장소에 넣어도 diff가 읽힌다.

**원본 프레임이나 사람 이미지는 만들지 않는다.** 개인정보와 재배포 조건
때문이다. 여기서 나오는 것은 집계 수치뿐이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

W, H = 720, 420
PAD_L, PAD_R, PAD_T, PAD_B = 70, 24, 62, 56

FG = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
SERIES = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#b83280"]


def _head(title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="system-ui,-apple-system,Segoe UI,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{PAD_L}" y="24" font-size="15" font-weight="600" fill="{FG}">{title}</text>',
        f'<text x="{PAD_L}" y="40" font-size="11" fill="{MUTED}">{subtitle}</text>',
    ]


def _axes(x_label: str, y_label: str, x_ticks, y_ticks) -> list[str]:
    x0, y0, x1, y1 = PAD_L, H - PAD_B, W - PAD_R, PAD_T
    out = []
    for value, pos in y_ticks:
        out.append(f'<line x1="{x0}" y1="{pos:.1f}" x2="{x1}" y2="{pos:.1f}" stroke="{GRID}"/>')
        out.append(
            f'<text x="{x0 - 8}" y="{pos + 4:.1f}" font-size="10" fill="{MUTED}" '
            f'text-anchor="end">{value}</text>'
        )
    for value, pos in x_ticks:
        out.append(
            f'<text x="{pos:.1f}" y="{y0 + 16}" font-size="10" fill="{MUTED}" '
            f'text-anchor="middle">{value}</text>'
        )
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{FG}" stroke-width="1"/>')
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{FG}" stroke-width="1"/>')
    out.append(
        f'<text x="{(x0 + x1) / 2:.0f}" y="{H - 12}" font-size="11" fill="{FG}" '
        f'text-anchor="middle">{x_label}</text>'
    )
    out.append(
        f'<text x="16" y="{(y0 + y1) / 2:.0f}" font-size="11" fill="{FG}" '
        f'text-anchor="middle" transform="rotate(-90 16 {(y0 + y1) / 2:.0f})">{y_label}</text>'
    )
    return out


def size_recall_chart(bins: list[tuple[str, int, dict[str, float | None]]], out: Path,
                      subtitle: str) -> None:
    """가로축 = 크기 구간, 세로축 = recall. 후보별 막대."""
    x0, y0, x1, y1 = PAD_L, H - PAD_B, W - PAD_R, PAD_T
    names = list(bins[0][2])
    parts = _head("Detection recall by apparent person size", subtitle)

    y_ticks = [(f"{v:.1f}", y0 - (y0 - y1) * v) for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    span = (x1 - x0) / len(bins)
    x_ticks = [(f"{label}  (n={n})", x0 + span * (i + 0.5)) for i, (label, n, _) in enumerate(bins)]
    parts += _axes("normalized bbox height (h / image height)", "recall", x_ticks, y_ticks)

    bar_w = span / (len(names) + 1)
    for i, (_, _, values) in enumerate(bins):
        for j, name in enumerate(names):
            v = values[name]
            if v is None:
                continue
            bx = x0 + span * i + bar_w * (j + 0.5)
            bh = (y0 - y1) * v
            parts.append(
                f'<rect x="{bx:.1f}" y="{y0 - bh:.1f}" width="{bar_w * 0.86:.1f}" '
                f'height="{max(bh, 0.6):.1f}" fill="{SERIES[j % len(SERIES)]}" opacity="0.88"/>'
            )
            if v >= 0.02:
                parts.append(
                    f'<text x="{bx + bar_w * 0.43:.1f}" y="{y0 - bh - 4:.1f}" font-size="9" '
                    f'fill="{FG}" text-anchor="middle">{v:.2f}</text>'
                )
    # 범례는 가용 폭에 맞춰 나눈다. 항목이 늘어도 캔버스 밖으로 나가지 않게.
    per_row = max(1, (x1 - x0) // 150)
    for j, name in enumerate(names):
        col, row = j % per_row, j // per_row
        lx = x0 + col * ((x1 - x0) / per_row)
        ly = y1 - 22 + row * 14
        parts.append(f'<rect x="{lx:.0f}" y="{ly:.0f}" width="10" height="10" '
                     f'fill="{SERIES[j % len(SERIES)]}"/>')
        parts.append(f'<text x="{lx + 15:.0f}" y="{ly + 9:.0f}" font-size="10" '
                     f'fill="{FG}">{name}</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def accuracy_efficiency_chart(points: list[tuple[str, float, float]], out: Path,
                              subtitle: str) -> None:
    """가로축 = 처리 FPS, 세로축 = occupancy macro MAE. 왼쪽 아래가 좋다."""
    x0, y0, x1, y1 = PAD_L, H - PAD_B, W - PAD_R, PAD_T
    fps = [p[1] for p in points]
    mae = [p[2] for p in points]
    fmax = max(fps) * 1.15
    mmax = max(mae) * 1.2
    parts = _head("Accuracy vs efficiency (development set)", subtitle)

    y_ticks = [(f"{mmax * f:.2f}", y0 - (y0 - y1) * f) for f in (0, 0.25, 0.5, 0.75, 1.0)]
    x_ticks = [(f"{fmax * f:.0f}", x0 + (x1 - x0) * f) for f in (0, 0.25, 0.5, 0.75, 1.0)]
    parts += _axes("processing FPS (GPU, higher is better)",
                   "occupancy macro MAE (lower is better)", x_ticks, y_ticks)

    src_x = x0 + (x1 - x0) * (25.0 / fmax)
    parts.append(f'<line x1="{src_x:.1f}" y1="{y1}" x2="{src_x:.1f}" y2="{y0}" '
                 f'stroke="{MUTED}" stroke-width="1" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{src_x + 5:.1f}" y="{y1 + 12}" font-size="10" '
                 f'fill="{MUTED}">source 25 fps</text>')

    for i, (name, f, m) in enumerate(points):
        px = x0 + (x1 - x0) * (f / fmax)
        py = y0 - (y0 - y1) * (m / mmax)
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" '
                     f'fill="{SERIES[i % len(SERIES)]}" opacity="0.9"/>')
        anchor = "end" if px > x1 - 120 else "start"
        dx = -10 if anchor == "end" else 10
        parts.append(f'<text x="{px + dx:.1f}" y="{py + 4:.1f}" font-size="10" '
                     f'fill="{FG}" text-anchor="{anchor}">{name}</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True,
                        help="development_aggregate.json")
    parser.add_argument("--final", type=Path,
                        help="final_test_aggregate.json. 있으면 FINAL 그림도 그린다.")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    agg = json.loads(args.development.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    order = sorted(agg, key=lambda k: agg[k]["macroMae"])
    # 표본 수는 development.json의 크기별 합에서 읽는다. 손으로 적어 두면
    # 데이터가 바뀌었을 때 그림만 옛날 숫자를 말한다.
    raw = json.loads(
        (args.development.parent / "development.json").read_text(encoding="utf-8")
    )
    counts = {b: 0 for b in ("SMALL", "MEDIUM", "LARGE")}
    first = raw["rows"][0]["candidate"]
    for row in raw["rows"]:
        if row["candidate"] != first:
            continue
        for b in counts:
            counts[b] += row["detection"]["bySize"][b]["total"]

    bins = [
        ("SMALL", counts["SMALL"], {k: agg[k]["small"] for k in order}),
        ("MEDIUM", counts["MEDIUM"], {k: agg[k]["medium"] for k in order}),
        ("LARGE", counts["LARGE"], {k: agg[k]["large"] for k in order}),
    ]
    size_recall_chart(
        bins, args.out_dir / "cctv_size_recall.svg",
        "CAVIAR shopping-corridor DEVELOPMENT set, 5 sequences, IoU>=0.5, conf 0.7",
    )
    accuracy_efficiency_chart(
        [(k, agg[k]["fps"], agg[k]["macroMae"]) for k in order],
        args.out_dir / "cctv_accuracy_efficiency.svg",
        "Lower-left is better. RTX 4060 Ti, 384x288 source.",
    )
    made = 2

    if args.final and args.final.exists():
        fin = json.loads(args.final.read_text(encoding="utf-8"))["candidates"]
        forder = sorted(fin, key=lambda k: fin[k]["occupancy"]["macroMae"])
        fbins = [
            (
                b,
                fin[forder[0]]["detection"]["bySize"][b]["gtTotal"],
                {k: fin[k]["detection"]["bySize"][b]["microRecall"] for k in forder},
            )
            for b in ("SMALL", "MEDIUM", "LARGE")
        ]
        size_recall_chart(
            fbins, args.out_dir / "cctv_size_recall_final.svg",
            "CAVIAR INRIA-lobby FINAL TEST, 17 fresh sequences, IoU>=0.5, conf 0.7",
        )
        made += 1

    print(f"그림 {made}장: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
