"""단일 날짜 중식 식수 예측 CLI (합성데이터 PoC 시연용).

실행:
    python src/predict.py
    python src/predict.py --date 2026-09-15 --menu-category 양식 --main-menu 돈까스 \
        --temperature 24 --rainfall 0 --humidity 60 --semester 2 --semester-week 3 --exam-period 0

- 요일·학기·주차·시험기간·공휴일은 날짜에서 자동으로 채운다(인자로 덮어쓸 수 있다).
- 과거 식수(lag)와 과거 CCTV feature는 예측일 **이전** 합성 이력에서 가져온다.
  --lag-1 등으로 직접 넣을 수도 있다. 예측일 당일이나 이후 기록은 쓰지 않는다.

출력되는 숫자는 합성데이터로 학습한 모델의 값이다. 실제 식수 예측이 아니다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

import joblib
import numpy as np
import pandas as pd

from common import (
    ALL_FEATURES,
    METADATA_PATH,
    MODEL_PATH,
    SYNTHETIC_NOTICE,
    TRACK_ROOT,
    WEEKDAYS,
    history_features,
    load_cctv_log,
    load_meal_data,
)
from generate_synthetic_data import CONFIG, academic_calendar

HISTORY_OPTIONS = {
    "lag_1": "직전 운영일 식수",
    "lag_7": "직전 동일 요일 식수",
    "rolling_mean_7": "직전 7운영일 평균 식수",
    "prev_avg_occupancy": "직전 운영일 CCTV 평균 동시 인원",
    "prev_peak_occupancy": "직전 운영일 CCTV 최대 동시 인원",
    "same_weekday_prev_occupancy": "직전 동일 요일 CCTV 평균 동시 인원",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="다음 날 중식 식수 예측 (합성데이터 PoC 시연)")
    p.add_argument("--date", help="예측일 YYYY-MM-DD. 기본값: 합성 이력 마지막 날의 다음 평일")
    p.add_argument("--menu-category", default="양식", choices=list(CONFIG["menus"]))
    p.add_argument("--main-menu", default="돈까스")
    p.add_argument("--menu-popularity", type=float,
                   help="0~1. 기본값: 메뉴별 기본 선호도(시뮬레이션 가정값)")
    p.add_argument("--special-menu", type=int, choices=[0, 1], default=0)
    p.add_argument("--temperature", type=float, default=24.0, help="예보 기온(℃)")
    p.add_argument("--rainfall", type=float, default=0.0, help="예보 강수량(mm)")
    p.add_argument("--humidity", type=float, default=60.0, help="예보 습도(%%)")
    p.add_argument("--semester", type=int, choices=[1, 2], help="기본값: 날짜에서 계산")
    p.add_argument("--semester-week", type=int, help="기본값: 날짜에서 계산")
    p.add_argument("--exam-period", type=int, choices=[0, 1], help="기본값: 주차에서 계산")
    p.add_argument("--holiday", type=int, choices=[0, 1], help="기본값: 공휴일 목록에서 계산")
    p.add_argument("--campus-event", type=int, choices=[0, 1], default=0)
    history = p.add_argument_group("과거 feature 직접 입력 (생략하면 합성 이력에서 자동 조회)")
    for name, help_text in HISTORY_OPTIONS.items():
        history.add_argument(f"--{name.replace('_', '-')}", type=float, help=help_text)
    return p.parse_args()


def previous_weekday(day: dt.date) -> dt.date:
    day -= dt.timedelta(days=1)
    while day.weekday() >= 5:
        day -= dt.timedelta(days=1)
    return day


def next_weekday(day: dt.date) -> dt.date:
    day += dt.timedelta(days=1)
    while day.weekday() >= 5:
        day += dt.timedelta(days=1)
    return day


def fail(message: str) -> None:
    print(f"오류: {message}", file=sys.stderr)
    raise SystemExit(2)


def main() -> None:
    args = parse_args()
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        fail("저장된 모델이 없다. 먼저 `python src/train.py` 를 실행한다.")
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    meal_log = load_meal_data()
    cctv_log = load_cctv_log()
    last_day = meal_log["date"].max().date()
    target = dt.date.fromisoformat(args.date) if args.date else next_weekday(last_day)
    if target.weekday() >= 5:
        fail(f"{target}는 주말이다. 이 PoC는 학기 중 평일 중식만 다룬다.")

    warnings: list[str] = []

    # --- 학사 정보 -----------------------------------------------------------
    calendar = academic_calendar(target) or {}
    semester = args.semester if args.semester is not None else calendar.get("semester")
    week = args.semester_week if args.semester_week is not None else calendar.get("semester_week")
    if semester is None or week is None:
        fail(f"{target}는 설정된 학기 기간 밖이다. --semester 와 --semester-week 를 지정한다.")
    if calendar and (semester, week) != (calendar["semester"], calendar["semester_week"]):
        warnings.append(f"입력한 학기·주차({semester}학기 {week}주차)가 합성 학사일정 계산값"
                        f"({calendar['semester']}학기 {calendar['semester_week']}주차)과 다르다. 입력값을 썼다.")
    exam = args.exam_period if args.exam_period is not None else int(week in CONFIG["exam_weeks"])
    holiday = args.holiday if args.holiday is not None else int(target.isoformat() in CONFIG["holidays"])

    # --- 메뉴 ---------------------------------------------------------------
    menus = CONFIG["menus"][args.menu_category]
    popularity = args.menu_popularity
    if args.main_menu not in menus:
        warnings.append(f"'{args.main_menu}'은(는) {args.menu_category} 메뉴 목록에 없다. "
                        "모델은 이 메뉴 이름을 모르는 값으로 처리한다.")
        if popularity is None:
            popularity = 0.5
            warnings.append("메뉴 선호도를 알 수 없어 0.5를 썼다. --menu-popularity 로 지정할 수 있다.")
    elif popularity is None:
        popularity = menus[args.main_menu]
    if not 0 <= popularity <= 1:
        fail("--menu-popularity 는 0~1 사이여야 한다.")

    # --- 과거 feature: 예측일 이전 기록만 ------------------------------------
    hist = history_features(meal_log, cctv_log, target)
    sources = {}
    for name in HISTORY_OPTIONS:
        given = getattr(args, name)
        if given is not None:
            hist[name] = given
            sources[name] = "직접 입력"
        else:
            sources[name] = "합성 이력"
        if pd.isna(hist[name]):
            fail(f"{name}을(를) 만들 이력이 부족하다. --{name.replace('_', '-')} 로 지정한다.")

    last_hist = hist["history_last_date"].date() if hist["history_last_date"] is not None else None
    auto_used = any(v == "합성 이력" for v in sources.values())
    expected_prev = previous_weekday(target)
    if auto_used and last_hist is not None and last_hist < expected_prev:
        warnings.append(f"직전 운영일({expected_prev}) 기록이 합성 이력에 없어 가장 최근 기록({last_hist})을 "
                        "과거 feature로 썼다. 실제 운영이라면 D-1 당일의 식수와 CCTV 값을 넣어야 한다 "
                        "(--lag-1, --prev-avg-occupancy 등).")
    if target <= last_day:
        warnings.append("이 날짜는 저장된 모델의 학습 데이터 기간 안에 있다. 예측값을 성능 근거로 쓰면 안 된다.")

    # --- 예측 ---------------------------------------------------------------
    row = {
        "weekday": WEEKDAYS[target.weekday()],
        "semester": semester,
        "semester_week": week,
        "exam_period": exam,
        "holiday": holiday,
        "campus_event": args.campus_event,
        "temperature": args.temperature,
        "rainfall": args.rainfall,
        "humidity": args.humidity,
        "menu_category": args.menu_category,
        "main_menu": args.main_menu,
        "menu_popularity": popularity,
        "special_menu": args.special_menu,
        **{name: hist[name] for name in HISTORY_OPTIONS},
    }
    features = pd.DataFrame([row])[ALL_FEATURES]
    raw = float(model.predict(features)[0])
    prediction = int(round(np.clip(raw, 0, CONFIG["capacity"])))
    q_low, q_high = metadata["synthetic_test_residual_quantiles"]["values"]
    low, high = max(0, round(prediction + q_low)), round(prediction + q_high)

    # --- 출력 ---------------------------------------------------------------
    rain_text = "강수 없음" if args.rainfall == 0 else f"강수 {args.rainfall:g}mm"
    print(SYNTHETIC_NOTICE)
    print()
    print(f"예상 중식 식수: {prediction}명")
    print(f"참고 범위: {low}~{high}명  (합성 test 구간 잔차의 10~90% 분위수를 더한 값. 통계적 신뢰구간이 아니다)")
    print()
    print("입력 조건:")
    print(f"- {target} {WEEKDAYS[target.weekday()]}요일")
    print(f"- {semester}학기 {week}주차")
    print(f"- {args.menu_category} / {args.main_menu} (메뉴 선호도 {popularity:.2f}, 시뮬레이션 가정값)"
          f"{', 특식' if args.special_menu else ''}")
    print(f"- 기온 {args.temperature:g}℃, {rain_text}, 습도 {args.humidity:g}%")
    print(f"- {'시험기간' if exam else '시험기간 아님'}, {'공휴일' if holiday else '공휴일 아님'}, "
          f"{'교내 행사 있음' if args.campus_event else '교내 행사 없음'}")
    print()
    print(f"과거 feature (예측일 이전 기록만 사용, 합성 이력의 최근 기록일 {last_hist}):")
    for name, label in HISTORY_OPTIONS.items():
        print(f"- {name} ({label}): {hist[name]:.1f}  [{sources[name]}]")
    if warnings:
        print()
        print("주의:")
        for w in warnings:
            print(f"- {w}")
    print()
    print(f"모델: {metadata['model_name']} (합성데이터 {metadata['training_rows']}행, "
          f"{metadata['training_period'][0]} ~ {metadata['training_period'][1]}로 학습, "
          f"{MODEL_PATH.relative_to(TRACK_ROOT).as_posix()})")
    print(f"합성 test MAE {metadata['synthetic_test_metrics']['MAE']:.1f}명 — 합성 환경의 수치이며 실제 성능이 아니다.")


if __name__ == "__main__":
    sys.exit(main())
