"""식수 예측 PoC의 공용 정의: 경로, feature 목록, 분할, 지표, D-1 이력 feature.

이 트랙의 데이터는 전부 합성데이터다. 여기서 나오는 어떤 숫자도 실제
급식소의 예측 성능이 아니다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRACK_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = TRACK_ROOT / "data"
MEAL_DATA_PATH = DATA_DIR / "synthetic_meal_demand.csv"
# CCTV 일일 관측 로그. 학습 CSV에는 이것을 하루 미룬(shift) 값만 들어간다.
# 예측 시점에 "어제의 CCTV"를 만들려면 원본 로그가 따로 있어야 한다.
CCTV_LOG_PATH = DATA_DIR / "synthetic_cctv_daily.csv"

OUTPUT_DIR = TRACK_ROOT / "outputs"
EDA_DIR = OUTPUT_DIR / "eda"
MODEL_DIR = TRACK_ROOT / "models"
MODEL_PATH = MODEL_DIR / "meal_demand_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

SYNTHETIC_NOTICE = (
    "[합성데이터 PoC] 설계된 가정으로 생성한 데이터에서 나온 값이다. "
    "실제 급식소의 식수나 예측 성능을 뜻하지 않는다."
)

WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]
TARGET = "meal_count"

CATEGORICAL_FEATURES = ["weekday", "semester", "menu_category", "main_menu"]

PAST_DEMAND_FEATURES = ["lag_1", "lag_7", "rolling_mean_7"]
ACADEMIC_FEATURES = ["semester", "semester_week", "exam_period", "holiday", "campus_event"]
# D-1 예측 시점에 확보 가능한 날씨 "예보값"이다(실제 날씨가 아니다).
WEATHER_FEATURES = ["temperature", "rainfall", "humidity"]
MENU_FEATURES = ["menu_category", "main_menu", "special_menu"]
CCTV_FEATURES = ["prev_avg_occupancy", "prev_peak_occupancy", "same_weekday_prev_occupancy"]

# 과거 식수와 과거 CCTV. 예측일 이전 기록에서만 만든다.
HISTORY_FEATURES = PAST_DEMAND_FEATURES + CCTV_FEATURES

# Ablation 단계. 각 단계는 앞 단계를 포함한다.
FEATURE_GROUPS: dict[str, list[str]] = {
    "M1": PAST_DEMAND_FEATURES + ["weekday"],
}
FEATURE_GROUPS["M2"] = FEATURE_GROUPS["M1"] + ACADEMIC_FEATURES
FEATURE_GROUPS["M3"] = FEATURE_GROUPS["M2"] + WEATHER_FEATURES + MENU_FEATURES
FEATURE_GROUPS["M4"] = FEATURE_GROUPS["M3"] + CCTV_FEATURES

FEATURE_GROUP_LABELS = {
    "M1": "과거 식수 + 요일",
    "M2": "M1 + 학사일정",
    "M3": "M2 + 날씨 + 메뉴",
    "M4": "M3 + CCTV 과거 이용 패턴",
}

ALL_FEATURES = FEATURE_GROUPS["M4"]
NUMERIC_FEATURES = [f for f in ALL_FEATURES if f not in CATEGORICAL_FEATURES]

SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15}  # 나머지가 test


def load_meal_data(path: Path = MEAL_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"{path} 가 없다. 먼저 `python src/generate_synthetic_data.py` 를 실행한다."
        )
    df = pd.read_csv(path, parse_dates=["date"], encoding="utf-8-sig")
    return df.sort_values("date").reset_index(drop=True)


def load_cctv_log(path: Path = CCTV_LOG_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], encoding="utf-8-sig")
    return df.sort_values("date").reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """날짜순 앞 70% / 다음 15% / 마지막 15%. 무작위로 섞지 않는다."""
    ordered = df.sort_values("date").reset_index(drop=True)
    n = len(ordered)
    n_train = int(round(n * SPLIT_FRACTIONS["train"]))
    n_val = int(round(n * SPLIT_FRACTIONS["validation"]))
    return {
        "train": ordered.iloc[:n_train],
        "validation": ordered.iloc[n_train : n_train + n_val],
        "test": ordered.iloc[n_train + n_val :],
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE가 주 지표다. bias = 평균(예측 - 실제). 양수면 과대 예측(음식이 남는다)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE": float(np.mean(np.abs(error) / y_true) * 100),
        "bias": float(np.mean(error)),
    }


def _last(series: pd.Series) -> float:
    return float(series.iloc[-1]) if len(series) else float("nan")


def history_features(
    meal_log: pd.DataFrame, cctv_log: pd.DataFrame, target_date
) -> dict[str, object]:
    """예측일(target_date) **이전** 기록만으로 과거 식수·CCTV feature를 만든다.

    D-1 시점에 알 수 있는 것만 쓴다. 예측일 당일과 그 이후 행은 먼저 잘라낸다.
    생성기는 같은 값을 shift/rolling으로 만들고, 이 함수로 한 행씩 다시 계산해
    둘이 일치하는지 검사한다(leakage 검증). predict.py도 이 함수를 쓴다.

    - lag_1: 직전 운영일 식수
    - lag_7: 직전 동일 요일 식수 (학기 중에는 7일 전)
    - rolling_mean_7: 직전 7개 운영일 식수 평균
    - prev_*_occupancy: 직전 운영일의 CCTV 평균/최대 동시 인원
    - same_weekday_prev_occupancy: 직전 동일 요일의 CCTV 평균 동시 인원
    """
    target = pd.Timestamp(target_date)
    meals = meal_log[meal_log["date"] < target].sort_values("date")
    cctv = cctv_log[cctv_log["date"] < target].sort_values("date")
    same_day_meals = meals[meals["date"].dt.weekday == target.weekday()]
    same_day_cctv = cctv[cctv["date"].dt.weekday == target.weekday()]

    return {
        "lag_1": _last(meals["meal_count"]),
        "lag_7": _last(same_day_meals["meal_count"]),
        "rolling_mean_7": (
            float(meals["meal_count"].tail(7).mean()) if len(meals) >= 7 else float("nan")
        ),
        "prev_avg_occupancy": _last(cctv["avg_occupancy"]),
        "prev_peak_occupancy": _last(cctv["peak_occupancy"]),
        "same_weekday_prev_occupancy": _last(same_day_cctv["avg_occupancy"]),
        "history_last_date": meals["date"].max() if len(meals) else None,
    }
