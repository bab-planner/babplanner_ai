"""합성 식수 데이터 EDA. 시연·보고서용 핵심 그래프만 만든다.

실행:
    python src/eda.py

출력: outputs/eda/*.png, outputs/eda/eda_summary.txt, outputs/eda/correlation_matrix.csv

그래프에 보이는 패턴은 generate_synthetic_data.py가 **정해 넣은** 것이다.
"데이터에서 발견한 사실"로 읽으면 안 된다. 생성 규칙이 의도대로 들어갔는지
확인하는 용도다.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import pandas as pd  # noqa: E402

from common import (  # noqa: E402
    CCTV_FEATURES,
    EDA_DIR,
    NUMERIC_FEATURES,
    SYNTHETIC_NOTICE,
    TARGET,
    TRACK_ROOT,
    load_meal_data,
    temporal_split,
)

# 차트 색. 단일 계열은 slot 1(파랑), 비교 계열은 slot 2(주황).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
NEUTRAL_BAND = "#f0efec"

FOOTNOTE = "합성데이터(synthetic) — 설계된 가정으로 생성한 값이며 실제 식당 데이터가 아님"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": ["Malgun Gothic", "NanumGothic", "sans-serif"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
    })


def save(fig: plt.Figure, path: Path) -> None:
    fig.text(0.01, 0.01, FOOTNOTE, fontsize=8, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  저장: {path.relative_to(TRACK_ROOT).as_posix()}")


def label_bars(ax, bars, values, fmt="{:.0f}") -> None:
    for bar, value in zip(bars, values):
        ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color=INK_2)


def plot_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.arange(50, 575, 15)
    normal = df.loc[df["holiday"] == 0, TARGET]
    holiday = df.loc[df["holiday"] == 1, TARGET]
    ax.hist(normal, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1, label=f"평일 ({len(normal)}일)")
    ax.hist(holiday, bins=bins, color=ORANGE, edgecolor=SURFACE, linewidth=1, label=f"공휴일 ({len(holiday)}일)")
    top = ax.get_ylim()[1] * 1.15
    ax.set_ylim(top=top)
    ax.axvline(normal.mean(), color=INK_2, linewidth=1)
    ax.annotate(f"평일 평균 {normal.mean():.0f}명", (normal.mean(), top),
                xytext=(6, -2), textcoords="offset points", fontsize=9, color=INK_2, va="top")
    ax.set_title("중식 식수 분포")
    ax.set_xlabel("meal_count (명)")
    ax.set_ylabel("일 수")
    ax.legend(loc="upper left")
    save(fig, EDA_DIR / "01_meal_count_distribution.png")


def plot_timeseries(df: pd.DataFrame) -> None:
    splits = temporal_split(df)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    # 방학을 건너 선을 잇지 않는다. 학기마다 따로 그린다.
    semester_id = (df["date"].diff().dt.days > 30).cumsum()
    for _, part in df.groupby(semester_id):
        ax.plot(part["date"], part[TARGET], color=BLUE, linewidth=1.4)
    hol = df[df["holiday"] == 1]
    ax.scatter(hol["date"], hol[TARGET], s=22, color=ORANGE, edgecolor=SURFACE, linewidth=1.2,
               zorder=3, label="공휴일")
    ax.plot([], [], color=BLUE, linewidth=1.4, label="식수")
    ax.axvspan(splits["validation"]["date"].min(), splits["test"]["date"].max(), color=NEUTRAL_BAND, zorder=0)
    for name, part in [("train", splits["train"]), ("validation", splits["validation"]), ("test", splits["test"])]:
        if name != "train":
            ax.axvline(part["date"].min(), color=AXIS, linewidth=1, zorder=1)
        ax.annotate(f"{name} {len(part)}일", (part["date"].min(), 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -4), textcoords="offset points", fontsize=9, color=INK_2, va="top")
    ax.set_title("일별 중식 식수 (방학 제외, 시간순 train / validation / test)")
    ax.set_ylabel("meal_count (명)")
    ax.set_ylim(top=ax.get_ylim()[1] + 30)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2)
    save(fig, EDA_DIR / "02_meal_count_timeseries.png")


def plot_weekday(normal: pd.DataFrame) -> pd.Series:
    means = normal.groupby("weekday")[TARGET].mean().reindex(["월", "화", "수", "목", "금"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(means.index, means.values, color=BLUE, width=0.6)
    label_bars(ax, bars, means.values)
    ax.set_ylim(0, means.max() * 1.12)
    ax.set_title("요일별 평균 식수 (공휴일 제외)")
    ax.set_ylabel("평균 meal_count (명)")
    save(fig, EDA_DIR / "03_weekday_mean.png")
    return means


def plot_semester_week(normal: pd.DataFrame, exam_weeks: set[int]) -> pd.Series:
    means = normal.groupby("semester_week")[TARGET].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [ORANGE if w in exam_weeks else BLUE for w in means.index]
    bars = ax.bar(means.index, means.values, color=colors, width=0.7)
    label_bars(ax, bars, means.values)
    ax.set_ylim(0, means.max() * 1.15)
    ax.set_xticks(means.index)
    ax.legend(handles=[Patch(color=BLUE, label="수업 주차"), Patch(color=ORANGE, label="시험 주차")],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2)
    ax.set_title("학기 주차별 평균 식수 (공휴일 제외)")
    ax.set_xlabel("semester_week")
    ax.set_ylabel("평균 meal_count (명)")
    save(fig, EDA_DIR / "04_semester_week_mean.png")
    return means


def plot_menu_category(normal: pd.DataFrame) -> pd.Series:
    means = normal.groupby("menu_category")[TARGET].mean().sort_values(ascending=False)
    counts = normal["menu_category"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(means.index, means.values, color=BLUE, width=0.6)
    label_bars(ax, bars, means.values)
    ax.set_xticks(range(len(means)), [f"{c}\n(n={counts[c]})" for c in means.index])
    ax.set_ylim(0, means.max() * 1.12)
    ax.set_title("메뉴 카테고리별 평균 식수 (공휴일 제외)")
    ax.set_ylabel("평균 meal_count (명)")
    save(fig, EDA_DIR / "05_menu_category_mean.png")
    return means


def plot_rain(normal: pd.DataFrame) -> pd.DataFrame:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.6]})
    rainy = normal["rainfall"] > 0
    groups = [normal.loc[~rainy, TARGET], normal.loc[rainy, TARGET]]
    means = [g.mean() for g in groups]
    bars = left.bar(["비 없음", "비 옴"], means, color=[BLUE, ORANGE], width=0.55)
    label_bars(left, bars, means)
    left.set_xticks([0, 1], [f"비 없음\n(n={len(groups[0])})", f"비 옴\n(n={len(groups[1])})"])
    left.set_ylim(0, max(means) * 1.12)
    left.set_title("강수 여부별 평균 식수")
    left.set_ylabel("평균 meal_count (명)")

    right.scatter(normal.loc[rainy, "rainfall"], normal.loc[rainy, TARGET], s=18, color=ORANGE,
                  edgecolor=SURFACE, linewidth=0.8, label="비 온 날")
    right.axhline(groups[0].mean(), color=BLUE, linewidth=1.2)
    right.annotate(f"비 없는 날 평균 {groups[0].mean():.0f}명", (1.0, groups[0].mean()),
                   xycoords=("axes fraction", "data"), xytext=(-4, 4), textcoords="offset points",
                   ha="right", fontsize=9, color=INK_2)
    right.set_title("강수량과 식수 (비 온 날)")
    right.set_xlabel("rainfall (mm)")
    right.set_ylabel("meal_count (명)")
    right.grid(axis="x", color=GRID, linewidth=0.7)
    save(fig, EDA_DIR / "06_rainfall.png")
    return pd.DataFrame({"n": [len(g) for g in groups], "mean": means}, index=["비 없음", "비 옴"])


def plot_exam(normal: pd.DataFrame) -> pd.DataFrame:
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = [normal.loc[normal["exam_period"] == 0, TARGET], normal.loc[normal["exam_period"] == 1, TARGET]]
    box = ax.boxplot(groups, tick_labels=[f"시험기간 아님\n(n={len(groups[0])})", f"시험기간\n(n={len(groups[1])})"],
                     widths=0.45, patch_artist=True, medianprops={"color": INK, "linewidth": 1.4},
                     whiskerprops={"color": AXIS}, capprops={"color": AXIS},
                     flierprops={"marker": "o", "markersize": 4, "markerfacecolor": MUTED, "markeredgecolor": SURFACE})
    for patch, color in zip(box["boxes"], [BLUE, ORANGE]):
        patch.set_facecolor(color)
        patch.set_edgecolor(SURFACE)
        patch.set_alpha(0.85)
    for i, g in enumerate(groups, start=1):
        ax.annotate(f"평균 {g.mean():.0f}", (i + 0.25, g.mean()), fontsize=9, color=INK_2, va="center")
    ax.set_title("시험기간 여부별 식수 (공휴일 제외)")
    ax.set_ylabel("meal_count (명)")
    save(fig, EDA_DIR / "07_exam_period.png")
    return pd.DataFrame({"n": [len(g) for g in groups], "mean": [g.mean() for g in groups]},
                        index=["시험기간 아님", "시험기간"])


def plot_correlation(df: pd.DataFrame) -> pd.Series:
    corr = df[NUMERIC_FEATURES + [TARGET]].corr()
    corr.round(3).to_csv(EDA_DIR / "correlation_matrix.csv", encoding="utf-8-sig")
    target_corr = corr[TARGET].drop(TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [BLUE if v >= 0 else RED for v in target_corr.values]
    ax.barh(target_corr.index, target_corr.values, color=colors, height=0.6)
    ax.axvline(0, color=AXIS, linewidth=1)
    for y, v in enumerate(target_corr.values):
        ax.annotate(f"{v:+.2f}", (v, y), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if v >= 0 else "right", va="center", fontsize=8.5, color=INK_2)
    ax.set_xlim(-1, 1)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    for label in ax.get_yticklabels():
        if label.get_text() in CCTV_FEATURES:
            label.set_color(INK)
    ax.set_title("수치형 feature와 meal_count의 상관계수 (전체 행, Pearson)")
    ax.set_xlabel("상관계수")
    save(fig, EDA_DIR / "08_feature_correlation.png")
    return target_corr


def main() -> None:
    setup_style()
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_meal_data()
    normal = df[df["holiday"] == 0]
    exam_weeks = set(df.loc[df["exam_period"] == 1, "semester_week"].unique())

    print(SYNTHETIC_NOTICE)
    print()
    plot_distribution(df)
    plot_timeseries(df)
    weekday = plot_weekday(normal)
    week = plot_semester_week(normal, exam_weeks)
    menu = plot_menu_category(normal)
    rain = plot_rain(normal)
    exam = plot_exam(normal)
    target_corr = plot_correlation(df)

    buf = io.StringIO()
    buf.write(SYNTHETIC_NOTICE + "\n\n")
    buf.write(f"행 수 {len(df)}, 기간 {df['date'].min().date()} ~ {df['date'].max().date()}\n\n")
    buf.write("[meal_count 기술통계]\n" + df[TARGET].describe().round(1).to_string() + "\n\n")
    buf.write("[요일별 평균, 공휴일 제외]\n" + weekday.round(1).to_string() + "\n\n")
    buf.write("[주차별 평균, 공휴일 제외]\n" + week.round(1).to_string() + "\n\n")
    buf.write("[메뉴 카테고리별 평균, 공휴일 제외]\n" + menu.round(1).to_string() + "\n\n")
    buf.write("[강수 여부별, 공휴일 제외]\n" + rain.round(1).to_string() + "\n")
    corr_rain = normal.loc[normal["rainfall"] > 0, ["rainfall", TARGET]].corr().iloc[0, 1]
    buf.write(f"비 온 날 강수량-식수 상관: {corr_rain:+.3f}\n\n")
    buf.write("[시험기간 여부별, 공휴일 제외]\n" + exam.round(1).to_string() + "\n\n")
    holiday_mean = df.loc[df["holiday"] == 1, TARGET].mean()
    buf.write(f"[공휴일] 평균 {holiday_mean:.1f}명 (평일 평균 {normal[TARGET].mean():.1f}명)\n\n")
    buf.write("[meal_count와의 상관계수, 절댓값 순]\n"
              + target_corr.reindex(target_corr.abs().sort_values(ascending=False).index).round(3).to_string()
              + "\n")
    summary = buf.getvalue()
    (EDA_DIR / "eda_summary.txt").write_text(summary, encoding="utf-8")
    print()
    print(summary)


if __name__ == "__main__":
    sys.exit(main())
