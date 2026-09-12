"""합성 중식 식수 데이터 생성기 (시연용 PoC).

**이 파일이 만드는 데이터는 실제 호서대학교 식당 데이터가 아니다.** 아래
CONFIG의 가정으로 만든 시뮬레이션이다. 요인 사이의 관계(비가 오면 식수가
준다, 메뉴 선호도가 높으면 는다 등)는 전부 이 파일이 **정해 넣은** 것이다.
이 데이터로 학습한 모델이 그 관계를 찾아내는 것은 파이프라인이 동작한다는
뜻일 뿐, 실제 인과관계나 실제 예측 성능의 근거가 되지 않는다.

실행:
    python src/generate_synthetic_data.py

출력:
    data/synthetic_meal_demand.csv    학습용 (한 행 = 하루 중식)
    data/synthetic_cctv_daily.csv     CCTV 일일 관측 로그 (당일 값)
    outputs/synthetic_effect_shares.csv  생성 요인별 분산 비중

같은 CONFIG와 seed로 다시 돌리면 같은 데이터가 나온다.
"""

from __future__ import annotations

import datetime as dt
import math
import sys

import numpy as np
import pandas as pd

from common import (
    CCTV_LOG_PATH,
    DATA_DIR,
    HISTORY_FEATURES,
    MEAL_DATA_PATH,
    OUTPUT_DIR,
    SYNTHETIC_NOTICE,
    WEEKDAYS,
    history_features,
)

# ---------------------------------------------------------------------------
# 가정값. 전부 시뮬레이션용이며 실측이나 문헌에서 온 값이 아니다.
# ---------------------------------------------------------------------------
CONFIG: dict = {
    "seed": 20260912,
    # 시연 시점(2026-09-12, 토) 직전 평일까지 만든다. 그래서 "다음 날 예측"을
    # 시연할 때 예측일이 데이터 바로 뒤에 온다. datetime.now()를 쓰지 않는
    # 이유는 재현성이다.
    "end_date": "2026-09-11",
    # (학기, 개강일). 주차는 개강일이 속한 주의 월요일부터 센다.
    # 호서대 실제 학사일정이 아니라 국내 대학의 흔한 형태(3월·9월 개강,
    # 16주)를 흉내 낸 것이다.
    "semesters": [
        (1, "2023-03-02"),
        (2, "2023-09-01"),
        (1, "2024-03-04"),
        (2, "2024-09-02"),
        (1, "2025-03-04"),
        (2, "2025-09-01"),
        (1, "2026-03-03"),
        (2, "2026-09-01"),
    ],
    "weeks_per_semester": 16,
    "exam_weeks": (8, 15),  # 중간·기말고사 주차
    # 학기 중 평일에 걸린 공휴일(근사 목록). 식당은 열지만 기숙사생 위주로
    # 수요가 크게 준다고 가정한다(holiday=1).
    "holidays": {
        "2023-05-05": "어린이날",
        "2023-05-29": "부처님오신날 대체공휴일",
        "2023-06-06": "현충일",
        "2023-09-28": "추석 연휴",
        "2023-09-29": "추석",
        "2023-10-02": "임시공휴일",
        "2023-10-03": "개천절",
        "2023-10-09": "한글날",
        "2024-04-10": "국회의원 선거",
        "2024-05-06": "어린이날 대체공휴일",
        "2024-05-15": "부처님오신날",
        "2024-06-06": "현충일",
        "2024-09-16": "추석 연휴",
        "2024-09-17": "추석",
        "2024-09-18": "추석 연휴",
        "2024-10-01": "국군의 날 임시공휴일",
        "2024-10-03": "개천절",
        "2024-10-09": "한글날",
        "2025-05-05": "어린이날·부처님오신날",
        "2025-05-06": "대체공휴일",
        "2025-06-03": "대통령 선거",
        "2025-06-06": "현충일",
        "2025-10-03": "개천절",
        "2025-10-06": "추석",
        "2025-10-07": "추석 연휴",
        "2025-10-08": "대체공휴일",
        "2025-10-09": "한글날",
        "2026-05-05": "어린이날",
        "2026-05-25": "부처님오신날 대체공휴일",
        "2026-06-03": "지방선거",
    },
    # --- 식수 생성식 -------------------------------------------------------
    "base_demand": 400,
    "weekday_effect": {"월": -10, "화": 5, "수": 15, "목": 5, "금": -25},
    # 학기 초 +, 중반 0, 후반 -. 표에 없는 주차는 0.
    "semester_week_effect": {1: 20, 2: 15, 3: 10, 13: -10, 14: -13, 15: -16, 16: -20},
    "exam_effect_range": (-50, -30),  # 시험일마다 균등분포
    "campus_event_prob": 0.05,
    "campus_event_effect_range": (20, 50),
    "holiday_demand_ratio": 0.30,  # 공휴일은 정상 수요의 30%
    "holiday_noise_sd": 8,
    # 날씨. 기온은 연중 사인 곡선 + AR(1) 잡음. 값은 D-1 예보로 간주한다.
    "temperature": {"annual_mean": 13.0, "amplitude": 14.0, "coldest_doy": 20,
                    "noise_sd": 2.5, "noise_ar": 0.6},
    # 쾌적 구간을 벗어난 1℃마다 -1.8명 (춥거나 더우면 캠퍼스 체류가 준다는 가정)
    "temperature_effect": {"comfort_low": 10.0, "comfort_high": 24.0, "per_degree": -1.8},
    "rain_prob_by_month": {3: 0.22, 4: 0.25, 5: 0.25, 6: 0.33, 9: 0.30,
                           10: 0.20, 11: 0.22, 12: 0.18},
    "rain_amount_gamma": {"shape": 0.8, "scale": 10.0, "min_mm": 0.5},
    # 비 효과 = -(15 + 15 * (1 - exp(-mm / 6)))  → 약 -15 ~ -30
    "rain_effect": {"at_onset": -15.0, "extra_max": -15.0, "scale_mm": 6.0},
    # 습도는 강수·계절과 상관만 있고 식수 생성식에 직접 들어가지 않는다.
    "humidity": {"base": 55.0, "seasonal_amplitude": 12.0, "rain_bump": 20.0,
                 "noise_sd": 7.0},
    # 메뉴. 선호도(0~1)는 메뉴별 기본값에 매일 작은 흔들림을 더한다.
    "menu_category_prob": {"한식": 0.30, "중식": 0.15, "양식": 0.17,
                           "일식": 0.13, "분식": 0.15, "기타": 0.10},
    "menu_category_effect": {"한식": 5, "중식": 15, "양식": 20,
                             "일식": 10, "분식": 0, "기타": -10},
    "menus": {
        "한식": {"제육볶음": 0.85, "불고기": 0.72, "비빔밥": 0.50,
                 "닭갈비": 0.78, "생선구이": 0.28},
        "중식": {"짜장면": 0.72, "짬뽕": 0.62, "탕수육": 0.90,
                 "마파두부": 0.32, "볶음밥": 0.50},
        "양식": {"돈까스": 0.93, "파스타": 0.60, "함박스테이크": 0.72,
                 "오므라이스": 0.45, "카레": 0.55},
        "일식": {"우동": 0.48, "규동": 0.62, "가츠동": 0.80,
                 "소바": 0.30, "일본식카레": 0.52},
        "분식": {"떡볶이": 0.70, "김밥": 0.40, "라면": 0.55,
                 "만두": 0.45, "쫄면": 0.35},
        "기타": {"샐러드": 0.18, "샌드위치": 0.30, "죽": 0.12,
                 "덮밥류 기타": 0.42, "실험 메뉴": 0.25},
    },
    "menu_popularity_noise_sd": 0.04,
    "menu_popularity_effect_scale": 60,  # (선호도 - 0.5) * 60 → 약 ±30명
    "special_menu_prob": 0.08,
    "special_menu_effect": 25,
    # 과거 수요 패턴을 만드는 잠재 요인. 모델은 이것을 직접 보지 못하고
    # 과거 식수(lag, rolling)로만 간접적으로 본다.
    "semester_level_sd": 15,  # 학기마다 달라지는 전체 수준(재학생 수 변화 등)
    "momentum": {"phi": 0.75, "stationary_sd": 24},  # autoregressive_effect
    # CCTV가 부분적으로 관측하는 잠재 요인(캠퍼스 체류 수준). 식수에도
    # 영향을 주고 CCTV 동시 인원에도 반영된다. CCTV가 과거 식수에 없는
    # 정보를 조금 더 갖도록 **설계한** 장치다.
    "campus_presence": {"phi": 0.85, "stationary_sd": 16},
    "noise_sd": 22,
    "capacity": 700,
    # CCTV 당일 관측. 평균 동시 인원 ≈ 식수의 12% + 체류 수준 + 잡음.
    "cctv": {"avg_ratio": 0.12, "presence_weight": 0.35, "noise_sd": 2.5,
             "peak_ratio_range": (1.6, 2.0), "peak_noise_sd": 3.0},
}


# ---------------------------------------------------------------------------
# 학사일정
# ---------------------------------------------------------------------------
def _monday(day: dt.date) -> dt.date:
    return day - dt.timedelta(days=day.weekday())


def semester_windows() -> list[tuple[int, dt.date, dt.date]]:
    """(학기, 개강일, 16주차 금요일)."""
    windows = []
    for semester, start in CONFIG["semesters"]:
        start_day = dt.date.fromisoformat(start)
        last_day = _monday(start_day) + dt.timedelta(weeks=CONFIG["weeks_per_semester"], days=-3)
        windows.append((semester, start_day, last_day))
    return windows


def academic_calendar(day) -> dict | None:
    """날짜의 학기·주차·시험·공휴일. 학기 밖이거나 주말이면 None."""
    day = pd.Timestamp(day).date()
    if day.weekday() >= 5:
        return None
    for semester, start_day, last_day in semester_windows():
        if start_day <= day <= last_day:
            week = (day - _monday(start_day)).days // 7 + 1
            return {
                "semester": semester,
                "semester_week": week,
                "exam_period": int(week in CONFIG["exam_weeks"]),
                "holiday": int(day.isoformat() in CONFIG["holidays"]),
            }
    return None


def build_calendar() -> pd.DataFrame:
    end = dt.date.fromisoformat(CONFIG["end_date"])
    rows = []
    for semester_index, (_, start_day, last_day) in enumerate(semester_windows()):
        day = start_day
        while day <= min(last_day, end):
            info = academic_calendar(day)
            if info is not None:
                rows.append({"date": pd.Timestamp(day), "weekday": WEEKDAYS[day.weekday()],
                             "semester_index": semester_index, **info})
            day += dt.timedelta(days=1)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 요인 생성
# ---------------------------------------------------------------------------
def _ar1(rng: np.random.Generator, n: int, phi: float, stationary_sd: float) -> np.ndarray:
    """정상 분포에서 시작하는 AR(1). 분산이 stationary_sd^2로 유지된다."""
    innovation_sd = stationary_sd * math.sqrt(1 - phi**2)
    out = np.empty(n)
    out[0] = rng.normal(0, stationary_sd)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + rng.normal(0, innovation_sd)
    return out


def add_weather(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    t_cfg, h_cfg = CONFIG["temperature"], CONFIG["humidity"]
    doy = df["date"].dt.dayofyear.to_numpy()
    seasonal = -np.cos(2 * np.pi * (doy - t_cfg["coldest_doy"]) / 365)  # 1월 -1, 7~8월 +1

    temp_noise = np.concatenate([
        _ar1(rng, int(size), t_cfg["noise_ar"], t_cfg["noise_sd"])
        for size in df.groupby("semester_index").size()
    ])
    temperature = t_cfg["annual_mean"] + t_cfg["amplitude"] * seasonal + temp_noise

    month = df["date"].dt.month.to_numpy()
    rain_prob = np.array([CONFIG["rain_prob_by_month"][m] for m in month])
    g_cfg = CONFIG["rain_amount_gamma"]
    is_rain = rng.random(len(df)) < rain_prob
    amount = g_cfg["min_mm"] + rng.gamma(g_cfg["shape"], g_cfg["scale"], len(df))
    rainfall = np.where(is_rain, amount, 0.0)

    humidity = (
        h_cfg["base"] + h_cfg["seasonal_amplitude"] * seasonal
        + h_cfg["rain_bump"] * is_rain + rng.normal(0, h_cfg["noise_sd"], len(df))
    )
    return df.assign(
        temperature=np.round(temperature, 1),
        rainfall=np.round(rainfall, 1),
        humidity=np.clip(np.round(humidity), 15, 100).astype(int),
    )


def add_menu(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    categories = list(CONFIG["menu_category_prob"])
    probs = np.array(list(CONFIG["menu_category_prob"].values()))
    chosen_category, chosen_menu, popularity = [], [], []
    previous_menu = None
    for _ in range(len(df)):
        category = categories[rng.choice(len(categories), p=probs)]
        options = [m for m in CONFIG["menus"][category] if m != previous_menu]
        menu = options[rng.integers(len(options))]
        base = CONFIG["menus"][category][menu]
        chosen_category.append(category)
        chosen_menu.append(menu)
        popularity.append(np.clip(base + rng.normal(0, CONFIG["menu_popularity_noise_sd"]), 0, 1))
        previous_menu = menu

    special = (rng.random(len(df)) < CONFIG["special_menu_prob"]) & (df["holiday"] == 0)
    return df.assign(
        menu_category=chosen_category,
        main_menu=chosen_menu,
        menu_popularity=np.round(popularity, 2),
        special_menu=special.astype(int),
    )


def add_campus_event(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    allowed = (df["holiday"] == 0) & (df["exam_period"] == 0)
    event = (rng.random(len(df)) < CONFIG["campus_event_prob"]) & allowed
    return df.assign(campus_event=event.astype(int))


def generate_meal_count(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """meal_count와, 분석용 요인 분해표(학습에는 쓰지 않는다)를 만든다."""
    n = len(df)
    c = CONFIG
    comp = pd.DataFrame(index=df.index)

    comp["weekday_effect"] = df["weekday"].map(c["weekday_effect"])
    comp["semester_effect"] = df["semester_week"].map(lambda w: c["semester_week_effect"].get(w, 0))
    comp["exam_effect"] = df["exam_period"] * rng.uniform(*c["exam_effect_range"], n)
    comp["campus_event_effect"] = df["campus_event"] * rng.uniform(*c["campus_event_effect_range"], n)

    t = c["temperature_effect"]
    too_cold = np.clip(t["comfort_low"] - df["temperature"], 0, None)
    too_hot = np.clip(df["temperature"] - t["comfort_high"], 0, None)
    comp["temperature_effect"] = t["per_degree"] * (too_cold + too_hot)
    r = c["rain_effect"]
    rain_effect = r["at_onset"] + r["extra_max"] * (1 - np.exp(-df["rainfall"] / r["scale_mm"]))
    comp["rain_effect"] = np.where(df["rainfall"] > 0, rain_effect, 0.0)

    comp["menu_category_effect"] = df["menu_category"].map(c["menu_category_effect"])
    comp["menu_popularity_effect"] = c["menu_popularity_effect_scale"] * (df["menu_popularity"] - 0.5)
    comp["special_menu_effect"] = df["special_menu"] * c["special_menu_effect"]

    # 잠재 요인. 방학을 지나면 이어지지 않으므로 학기마다 새로 시작한다.
    level = rng.normal(0, c["semester_level_sd"], df["semester_index"].nunique())
    comp["semester_level"] = level[df["semester_index"].to_numpy()]
    sizes = df.groupby("semester_index").size()
    comp["autoregressive_effect"] = np.concatenate(
        [_ar1(rng, int(s), c["momentum"]["phi"], c["momentum"]["stationary_sd"]) for s in sizes])
    comp["campus_presence"] = np.concatenate(
        [_ar1(rng, int(s), c["campus_presence"]["phi"], c["campus_presence"]["stationary_sd"]) for s in sizes])

    comp["random_noise"] = rng.normal(0, c["noise_sd"], n)

    structural = c["base_demand"] + comp.drop(columns="random_noise").sum(axis=1)
    normal_day = structural + comp["random_noise"]
    holiday_day = c["holiday_demand_ratio"] * structural + rng.normal(0, c["holiday_noise_sd"], n)
    meal = np.where(df["holiday"] == 1, holiday_day, normal_day)
    meal = np.clip(np.round(meal), 0, c["capacity"]).astype(int)
    return df.assign(meal_count=meal), comp


def generate_cctv_log(df: pd.DataFrame, comp: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """CCTV 당일 관측. 식수와 강하게 상관되지만 같지 않다."""
    cc = CONFIG["cctv"]
    n = len(df)
    avg = (cc["avg_ratio"] * df["meal_count"] + cc["presence_weight"] * comp["campus_presence"]
           + rng.normal(0, cc["noise_sd"], n))
    avg = np.clip(avg, 0, None).round(1)
    peak = avg * rng.uniform(*cc["peak_ratio_range"], n) + rng.normal(0, cc["peak_noise_sd"], n)
    peak = np.maximum(peak, avg + 1).round(1)
    return pd.DataFrame({"date": df["date"], "avg_occupancy": avg, "peak_occupancy": peak})


def add_history_features(df: pd.DataFrame, cctv_log: pd.DataFrame) -> pd.DataFrame:
    """target을 다 만든 뒤에 shift로 과거 feature를 붙인다. 당일 값은 들어가지 않는다."""
    df = df.sort_values("date").reset_index(drop=True)
    cctv = cctv_log.set_index("date").loc[df["date"]].reset_index()
    meal = df["meal_count"]
    return df.assign(
        lag_1=meal.shift(1),
        lag_7=meal.groupby(df["weekday"]).shift(1),
        rolling_mean_7=meal.shift(1).rolling(7, min_periods=7).mean(),
        prev_avg_occupancy=cctv["avg_occupancy"].shift(1),
        prev_peak_occupancy=cctv["peak_occupancy"].shift(1),
        same_weekday_prev_occupancy=cctv["avg_occupancy"].groupby(df["weekday"]).shift(1),
    )


def verify_no_leakage(df: pd.DataFrame, cctv_log: pd.DataFrame) -> None:
    """모든 행의 과거 feature를 '그날 이전 기록만 보는' 함수로 다시 계산해 대조한다."""
    meal_log = df[["date", "weekday", "meal_count"]]
    for _, row in df.iterrows():
        expected = history_features(meal_log, cctv_log, row["date"])
        for name in HISTORY_FEATURES:
            got, want = row[name], expected[name]
            same = (pd.isna(got) and pd.isna(want)) or math.isclose(got, want, abs_tol=1e-9)
            if not same:
                raise AssertionError(f"leakage 검증 실패: {row['date'].date()} {name} {got} != {want}")


def effect_shares(df: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """평일(공휴일 제외) 기준, 생성 요인 묶음별 분산 비중. 모델의 feature 중요도가 아니다."""
    groups = {
        "과거 수요 패턴 (학기 수준 + 자기상관)": ["semester_level", "autoregressive_effect"],
        "요일·학사": ["weekday_effect", "semester_effect", "exam_effect", "campus_event_effect"],
        "메뉴": ["menu_category_effect", "menu_popularity_effect", "special_menu_effect"],
        "날씨": ["temperature_effect", "rain_effect"],
        "캠퍼스 체류 수준 (CCTV가 부분 관측)": ["campus_presence"],
        "noise": ["random_noise"],
    }
    normal = comp[df["holiday"] == 0]
    variances = {name: normal[cols].sum(axis=1).var() for name, cols in groups.items()}
    total = sum(variances.values())
    return pd.DataFrame({
        "component_group": list(variances),
        "std_people": [round(math.sqrt(v), 1) for v in variances.values()],
        "variance_share_pct": [round(100 * v / total, 1) for v in variances.values()],
    })


def main() -> None:
    rng = np.random.default_rng(CONFIG["seed"])

    df = build_calendar()
    df = add_weather(df, rng)
    df = add_menu(df, rng)
    df = add_campus_event(df, rng)
    df, components = generate_meal_count(df, rng)
    cctv_log = generate_cctv_log(df, components, rng)
    shares = effect_shares(df, components)

    df = add_history_features(df, cctv_log)
    verify_no_leakage(df, cctv_log)
    n_before = len(df)
    df = df.dropna(subset=HISTORY_FEATURES).reset_index(drop=True)
    dropped = n_before - len(df)

    for col in ["lag_1", "lag_7"]:
        df[col] = df[col].astype(int)
    df["rolling_mean_7"] = df["rolling_mean_7"].round(2)

    columns = [
        "date", "weekday",
        "semester", "semester_week", "exam_period", "holiday", "campus_event",
        "temperature", "rainfall", "humidity",
        "menu_category", "main_menu", "menu_popularity", "special_menu",
        "lag_1", "lag_7", "rolling_mean_7",
        "prev_avg_occupancy", "prev_peak_occupancy", "same_weekday_prev_occupancy",
        "meal_count",
    ]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = df[columns].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(MEAL_DATA_PATH, index=False, encoding="utf-8-sig")
    cctv_out = cctv_log.assign(date=cctv_log["date"].dt.strftime("%Y-%m-%d"))
    cctv_out.to_csv(CCTV_LOG_PATH, index=False, encoding="utf-8-sig")
    shares.to_csv(OUTPUT_DIR / "synthetic_effect_shares.csv", index=False, encoding="utf-8-sig")

    meal = df["meal_count"]
    normal = df.loc[df["holiday"] == 0, "meal_count"]
    print(SYNTHETIC_NOTICE)
    print()
    print(f"저장: {MEAL_DATA_PATH.relative_to(MEAL_DATA_PATH.parents[1])}")
    print(f"저장: {CCTV_LOG_PATH.relative_to(CCTV_LOG_PATH.parents[1])} (CCTV 당일 관측 로그)")
    print(f"shape: {out.shape}  (lag/rolling 생성으로 앞 {dropped}행 제거)")
    print(f"기간: {df['date'].min().date()} ~ {df['date'].max().date()}")
    per_semester = df.groupby([df["date"].dt.year.rename("year"), "semester"]).size()
    print("학기별 행 수:", ", ".join(f"{y}-{s}학기 {n}" for (y, s), n in per_semester.items()))
    print(f"공휴일 행: {int(df['holiday'].sum())}, 시험기간 행: {int(df['exam_period'].sum())}, "
          f"행사 행: {int(df['campus_event'].sum())}, 특식 행: {int(df['special_menu'].sum())}")
    print()
    print("meal_count 통계")
    print(f"  전체      평균 {meal.mean():.1f}  표준편차 {meal.std():.1f}  최소 {meal.min()}  "
          f"최대 {meal.max()}  중앙값 {meal.median():.0f}")
    print(f"  공휴일 제외 평균 {normal.mean():.1f}  표준편차 {normal.std():.1f}  "
          f"최소 {normal.min()}  최대 {normal.max()}")
    print()
    print("생성 요인별 분산 비중 (공휴일 제외, 모델 성능과 무관한 설계값 점검용)")
    print(shares.to_string(index=False))
    print()
    print("leakage 검증: 모든 행의 lag/rolling/CCTV feature가 '그날 이전 기록'만으로 재현됨 — 통과")


if __name__ == "__main__":
    sys.exit(main())
