# AI-3. 다음 날 중식 식수 예측 — 합성데이터 PoC

*Data-efficient Meal Demand Forecasting under Limited Historical Data*

> **이 폴더의 모델은 합성데이터로 학습한 시연용 PoC다.** 실제 호서대학교 식당
> 운영 데이터는 아직 한 건도 없다. 아래 성능 숫자는 우리가 정한 생성 규칙을
> 모델이 얼마나 되찾는지를 잰 값이다. 실제 급식소에서의 예측 정확도가 아니다.

> 본 PoC는 실제 대학 급식 운영 데이터가 충분히 확보되지 않은 개발 초기 단계에서
> 식수 예측 파이프라인의 기술적 구현 가능성을 확인하기 위해 합성데이터를 사용한다.
> 합성데이터 기반 성능은 실제 급식 환경의 예측 성능을 의미하지 않으며, 향후 파일럿
> 운영을 통해 실제 식수·학사일정·날씨·메뉴·CCTV 데이터를 확보한 뒤 기관별 모델을
> 재학습하고 검증할 예정이다.

## 1. 프로젝트 목적

데이터 생성부터 EDA, 시간순 분할 학습·평가, 모델 저장, 단일 날짜 예측까지
**파이프라인이 끝까지 돈다는 것**을 보이는 것이 목적이다. 높은 점수는 목표가
아니다.

실제 데이터가 생기면 첫 단계(합성데이터 생성)만 실제 적재로 바꾼다. 나머지는
그대로 쓴다. 컬럼 계약, leakage 없는 과거 feature 생성, 시간순 분할, baseline
비교, ablation, 모델 저장, 예측 CLI가 여기에 해당한다.

**학습은 여기서, 서빙은 FastAPI AI 서버가 한다.** 저장한 artifact(`models/`)를
FastAPI 서버가 읽어 `POST /internal/v1/ai/demand-forecast/predict`로 답한다(루트
README의 "식수 예측 API"). 서버는 이 폴더의 코드를 import하지 않고 artifact 파일만
쓴다. 과거 feature 계산은 서버 쪽에 같은 규칙으로 따로 구현했고, 두 구현이 같은
값을 내는지 테스트가 대조한다. **Spring Boot는 아직 이 API를 부르지 않는다.**

기존 `src/babplanner_ai/demand/`의 `DemandEstimator`와 `AIDemandRequest` 계약은
바꾸지 않았다(fixture/disabled 그대로). 실데이터 계약 확장은 실제 식수 정답 source가
정해진 뒤 별도 버전에서 한다(루트 README의 "수요 예측: 정답 데이터가 없다").

## 2. 예측 문제 정의

> 날씨 예보, 학사일정, 요일, 메뉴 유형, 과거 식수와 과거 CCTV 이용 패턴으로
> **다음 날 중식 식수 인원**을 예측한다.

| 항목 | 정의 |
|---|---|
| 예측 시점 | D-1. 직전 운영일 중식이 끝난 뒤 |
| 대상 | 다음 운영일 중식 한 끼. 조식·석식은 이번 범위가 아니다 |
| target | `meal_count` — 그날 중식 실제 식수 인원 |
| 행 단위 | 하루 한 행. 학기 중 평일만. 방학은 제외한다 |

D-1에 알 수 없는 값은 입력에 넣지 않는다. 특히 **예측일 당일의 CCTV 혼잡도는
쓰지 않는다.** 그날 점심의 결과이므로 그대로 leakage다. CCTV는 직전 운영일과
직전 동일 요일의 관측만 쓴다.

## 3. 합성데이터를 사용하는 이유

**실제 식수의 정답이 쌓이는 곳이 아직 없다.**

- `congestion_observations.peopleCount`는 특정 시점의 혼잡도 관측이다. 한 끼를
  실제로 먹은 사람 수가 아니다.
- 잔반 제출 건수는 인증에 참여한 사람 수다. 이용자 전체가 아니다.
- 식권·입장 기록 같은 canonical source가 아직 없다.

그렇다고 파이프라인까지 미룰 필요는 없다. 컬럼 계약, 과거 feature를 미래 정보
없이 만드는 방법, 시간순 분할, baseline 비교는 데이터 없이도 만들고 검증할 수
있다.

합성데이터는 정답(생성 규칙)을 알기 때문에 **파이프라인 버그를 잡는 데는 오히려
유리하다.** 예를 들어 leakage가 있으면 성능이 이 환경의 이론적 하한(8절)보다
좋게 나오므로 바로 드러난다.

반대로 합성데이터로는 알 수 없는 것이 있다. 실제 요인의 효과 크기, 실제 예측
성능, 실제 feature 중요도는 전부 생성 규칙이 정한 값이다.

## 4. 데이터 컬럼

`data/synthetic_meal_demand.csv` — 553행 × 21열, 2023-03-13 ~ 2026-09-11.

| 구분 | 컬럼 | 설명 | D-1에 아는 값인가 |
|---|---|---|---|
| 기본 | `date` | 날짜 (YYYY-MM-DD) | |
| | `weekday` | 월~금 | 예 |
| 학사 | `semester` | 1 / 2 | 예 |
| | `semester_week` | 1~16. 개강일이 속한 주의 월요일부터 센다 | 예 |
| | `exam_period` | 8·15주차 = 1 | 예 |
| | `holiday` | 학기 중 평일 공휴일 = 1 | 예 |
| | `campus_event` | 교내 행사 = 1 | 예 (사전 공지) |
| 날씨 | `temperature` | 기온 ℃. **D-1 예보값으로 간주** | 예 |
| | `rainfall` | 강수량 mm. 예보값으로 간주 | 예 |
| | `humidity` | 습도 %. 예보값으로 간주 | 예 |
| 메뉴 | `menu_category` | 한식·중식·양식·일식·분식·기타 | 예 (식단표) |
| | `main_menu` | 대표 메뉴 (카테고리당 5개) | 예 |
| | `menu_popularity` | 0~1 선호도. **시뮬레이션 가정값** | 예 |
| | `special_menu` | 특식 = 1 | 예 |
| 과거 식수 | `lag_1` | 직전 운영일 식수 | 예 |
| | `lag_7` | 직전 동일 요일 식수 (학기 중에는 7일 전) | 예 |
| | `rolling_mean_7` | 직전 7개 운영일 평균 식수 | 예 |
| 과거 CCTV | `prev_avg_occupancy` | 직전 운영일 점심 CCTV 평균 동시 인원 | 예 |
| | `prev_peak_occupancy` | 직전 운영일 점심 CCTV 최대 동시 인원 | 예 |
| | `same_weekday_prev_occupancy` | 직전 동일 요일 CCTV 평균 동시 인원 | 예 |
| target | `meal_count` | 그날 중식 식수 | **아니오 (맞힐 값)** |

`data/synthetic_cctv_daily.csv`는 CCTV의 **당일** 관측 로그(`date`,
`avg_occupancy`, `peak_occupancy`)다. 학습 CSV에는 이 로그를 하루 미룬 값만
들어간다. 로그를 따로 둔 이유는 예측 시점에 "어제의 CCTV"를 만들려면 원본이
필요하기 때문이다. 실제 시스템에서도 CCTV 로그는 식수 기록과 별개의 source다.

## 5. 데이터 생성 가정

모든 가정값은 `src/generate_synthetic_data.py` 맨 위의 `CONFIG` 한 곳에 있다.
**전부 시뮬레이션용으로 정한 값이며 실측이나 문헌에서 온 값이 아니다.**

### 기간

2023년 1학기부터 2026년 2학기 2주차(2026-09-11)까지 학기 중 평일을 쓴다. 학기는
3월·9월 개강에 16주로 두었다. 호서대의 실제 학사일정이 아니라 국내 대학의 흔한
형태를 흉내 낸 것이다.

2024~2026년 여섯 학기만으로는 약 470행이라 요청 범위(500~700행)에 못 미친다.
그래서 2023년 1학기부터 시작했다. 끝을 2026-09-11로 둔 것은 시연 시점(2026-09-12)
직전 평일이기 때문이다. 그러면 "다음 날 예측" 대상이 데이터 바로 뒤에 온다.

| 학기 | 2023-1 | 2023-2 | 2024-1 | 2024-2 | 2025-1 | 2025-2 | 2026-1 | 2026-2 |
|---|---|---|---|---|---|---|---|---|
| 행 수 | 70 | 76 | 80 | 80 | 79 | 80 | 79 | 9 |

2023-1학기는 lag 생성으로 앞 7행이 빠졌다. 공휴일은 제외하지 않고 `holiday=1`로
두었다. 학기 중 평일에 걸린 공휴일(추석, 선거일, 대체공휴일 등의 근사 목록)
30일이 여기에 해당하고, 식당은 열지만 정상 수요의 30%만 온다고 가정했다.

### 식수 생성식

```
meal_count = base_demand(400)
           + weekday_effect + semester_effect + exam_effect + campus_event_effect   # 요일·학사
           + temperature_effect + rain_effect                                        # 날씨
           + menu_category_effect + menu_popularity_effect + special_menu_effect     # 메뉴
           + semester_level + autoregressive_effect                                  # 과거 수요 패턴(잠재)
           + campus_presence                                                         # CCTV가 부분 관측하는 잠재 요인
           + random_noise
공휴일: meal_count = 0.30 × (noise를 뺀 위 합) + N(0, 8)
0 ~ 700명으로 자르고 정수로 반올림
```

| 요인 | 가정 |
|---|---|
| 요일 | 월 -10, 화 +5, 수 +15, 목 +5, 금 -25 |
| 학기 주차 | 1주 +20, 2주 +15, 3주 +10, 4~12주 0, 13주 -10, 14주 -13, 15주 -16, 16주 -20 |
| 시험기간 | 8·15주차에 날마다 -30 ~ -50 균등분포 |
| 교내 행사 | 공휴일·시험일이 아닌 날의 5%, +20 ~ +50 |
| 비 | 비 온 날 -(15 + 15·(1 - e^(-mm/6))) → 약 -15 ~ -30. 강수 확률은 월별 18~33% |
| 기온 | 10~24℃를 벗어난 1℃마다 -1.8 (춥거나 더우면 캠퍼스 체류가 준다는 가정) |
| 습도 | 강수·계절과 상관만 있고 **생성식에는 들어가지 않는다** |
| 메뉴 카테고리 | 한식 +5, 중식 +15, 양식 +20, 일식 +10, 분식 0, 기타 -10 |
| 메뉴 선호도 | 60 × (선호도 - 0.5) → 약 ±30명 |
| 특식 | 8%의 날, +25 |
| 학기 수준 | 학기마다 N(0, 15). 재학생 수 변화 같은 것을 흉내 낸다 |
| 자기상관 | AR(1), φ = 0.75, 정상 표준편차 24. 학기마다 새로 시작 |
| 캠퍼스 체류 수준 | AR(1), φ = 0.85, 정상 표준편차 16. 학기마다 새로 시작 |
| noise | N(0, 22) |

**메뉴 선호도.** 메뉴마다 기본값을 두고, 그 메뉴가 나온 날 N(0, 0.04)만큼만
흔든다. 같은 메뉴가 나올 때마다 선호도가 제멋대로 바뀌지 않게 하려는 것이다.
기본값은 돈까스 0.93, 탕수육 0.90, 제육볶음 0.85부터 죽 0.12, 샐러드 0.18까지다.
실제 선호도 조사 결과가 아니라 **시뮬레이션용 가정이다.**

### CCTV 과거 이용 패턴

CCTV 당일 관측은 이렇게 만든다.

```
avg_occupancy  = 0.12 × meal_count + 0.35 × campus_presence + N(0, 2.5)
peak_occupancy = avg_occupancy × U(1.6, 2.0) + N(0, 3)   (항상 avg보다 크게)
```

식수와 강하게 상관되지만 같지 않다. CCTV가 과거 식수에 없는 정보(캠퍼스 체류
수준)를 조금 더 갖도록 **일부러 설계한** 것이다. 따라서 CCTV를 넣어 성능이
좋아진다 해도 그것은 이 설계 때문이다. 실제 환경에서 CCTV가 도움이 된다는 근거가
아니다. 실제 CCTV 계수의 오차(AI-1에서 잰 통과 계수 recall 0.6~0.8, occupancy
과소 계수)도 여기에는 없다.

### leakage 방지

1. target(`meal_count`)을 모든 날짜에 대해 먼저 다 만든다.
2. 그다음 `shift`와 `rolling`으로 과거 feature를 만든다. `lag_7`은 요일별로
   묶어 `shift(1)` 한 값이다.
3. **모든 행**에 대해 `common.history_features()`로 과거 feature를 다시 계산해
   대조한다. 이 함수는 예측일 이전 행만 남기고 계산한다. 하나라도 다르면
   생성이 실패한다. `predict.py`도 같은 함수를 쓰므로 학습과 예측의 feature
   정의가 어긋나지 않는다.
4. NaN이 생긴 앞 7행은 제거한다.

같은 seed로 다시 돌리면 같은 CSV가 나온다(SHA-256 일치 확인).

### 요인별 영향 비중

아래는 생성 요인 묶음별 분산 비중이다(공휴일 제외, `outputs/synthetic_effect_shares.csv`).
모델의 feature 중요도가 아니라 생성 설계를 점검하는 값이다.

| 요인 묶음 | 표준편차(명) | 분산 비중 | 요청 가이드 |
|---|---|---|---|
| 과거 수요 패턴 (학기 수준 + 자기상관) | 28.3 | 31.4% | 35~40% |
| 요일·학사 | 24.7 | 24.0% | 20~25% |
| 메뉴 | 18.3 | 13.2% | 15~20% |
| 날씨 | 11.2 | 4.9% | 10~15% |
| 캠퍼스 체류 수준 (CCTV가 부분 관측) | 14.1 | 7.8% | 10~15% |
| noise | 21.7 | 18.5% | 나머지 |

**날씨가 가이드보다 낮다.** 비 효과를 -10~-30명 범위에 두고 강수 빈도를 현실적인
수준(학기 중 약 25%)으로 두면, 분산의 10%를 넘기려면 효과를 과장해야 한다. 그렇게
하지 않았다.

설계값은 **모델을 한 번도 학습하기 전에** 한 번 조정했다. 첫 생성의 비중(날씨
4.0%, 체류 수준 5.6%, 메뉴 10.8%)이 가이드와 멀어서, 비 효과 하한·기온 구간·메뉴
선호도 폭·특식 효과·체류 수준 표준편차를 올렸다. 그 뒤로는 바꾸지 않았다. 모델
결과를 보고 생성 규칙을 고치면 "잘 나오는 데이터"를 만드는 것이 되기 때문이다.

## 6. 실행 방법

`research/demand_forecasting/`에서 실행한다. 서비스 환경(`.venv`)에는 pandas와
sklearn이 없고, 넣지도 않는다(`tests/test_no_model_yet.py`).

    cd research/demand_forecasting
    python src/generate_synthetic_data.py   # 1. 합성데이터 생성 + leakage 검증
    python src/eda.py                       # 2. EDA 그래프
    python src/train.py                     # 3. 학습·평가·ablation·모델 저장
    python src/predict.py                   # 4. 다음 운영일 예측

Windows에서 여러 Python이 있으면 `py -3.12 src/...`처럼 부른다.

검증은 이미 패키지가 설치된 시스템 Python 3.12.1에서 했다(버전은
`requirements.txt`). 새 가상환경에 설치하는 절차는 돌려 보지 않았다. 새로 만든다면
이 폴더 안에서 한다. `.venv/`는 어느 깊이에서든 `.gitignore`에 걸린다.

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt

### 예측 CLI

    python src/predict.py --date 2026-09-15 --menu-category 양식 --main-menu 돈까스 ^
        --temperature 24 --rainfall 0 --humidity 60 --semester 2 --semester-week 3 --exam-period 0

| 인자 | 기본값 |
|---|---|
| `--date` | 합성 이력 마지막 날의 다음 평일 (지금은 2026-09-14) |
| `--semester`, `--semester-week` | 날짜에서 계산. 설정된 학기 밖이면 필수 |
| `--exam-period`, `--holiday` | 주차와 공휴일 목록에서 계산 |
| `--menu-category`, `--main-menu` | 양식 / 돈까스 |
| `--menu-popularity` | 메뉴별 기본 선호도(가정값) |
| `--special-menu`, `--campus-event` | 0 |
| `--temperature`, `--rainfall`, `--humidity` | 24, 0, 60 |
| `--lag-1`, `--lag-7`, `--rolling-mean-7`, `--prev-avg-occupancy`, `--prev-peak-occupancy`, `--same-weekday-prev-occupancy` | 예측일 **이전** 합성 이력에서 자동 조회 |

주말이나 설정된 학기 밖의 날짜는 오류로 끝낸다. 직전 운영일 기록이 이력에 없으면
가장 최근 기록을 쓰고 경고한다(예: 9/15 예측인데 9/14 기록이 없는 경우). 학습
데이터 기간 안의 날짜를 넣어도 경고한다.

## 7. 모델 구조

```
ColumnTransformer
  ├─ OneHotEncoder(handle_unknown="ignore"): weekday, semester, menu_category, main_menu
  └─ passthrough: 수치형 15개
→ 회귀 모델
```

| 이름 | 역할 | 설정 |
|---|---|---|
| `lag7_baseline` | baseline 1 | `lag_7` 그대로 |
| `rolling7_baseline` | baseline 2 | `rolling_mean_7` 그대로 |
| `random_forest` | 후보 | 500 trees, min_samples_leaf 3, max_features 0.5 |
| `lightgbm` | 후보 | 400 trees, lr 0.03, num_leaves 15, min_child_samples 10, subsample·colsample 0.8 |
| `lag1_baseline` | 참고 | `lag_1` 그대로. 기존 baseline 계단의 1단계 |
| `ridge` | 참고 | 선형 회귀(α = 1, 수치형 표준화). 선택 대상 아님 |

LightGBM이 없으면 XGBoost, 그것도 없으면 sklearn `HistGradientBoostingRegressor`로
대체한다. 새 패키지를 설치하지 않는다. 이번 환경에는 LightGBM 4.7.0이 있었다.

하이퍼파라미터는 고정값이다. 탐색하지 않았다.

**분할과 학습 순서.** 날짜순으로 앞 70% / 다음 15% / 마지막 15%로 자른다. 무작위로
섞지 않는다.

1. train으로 학습하고 validation으로 잰다. 여기서 RF와 LightGBM 중 MAE가 낮은 쪽을
   고른다.
2. 같은 설정을 train+validation으로 다시 학습해 test를 **한 번** 잰다.
3. 저장하는 최종 모델은 전체 553행으로 다시 학습한 것이다. 다음 날을 예측할 때
   최신 학기를 보게 하려는 것이다. 그래서 저장된 모델 자체에는 test 성능이 없다.
   test 지표는 2번 모델의 것이다(`metadata.json`에도 적었다).

저장물은 `models/meal_demand_model.joblib`(전처리 포함 sklearn Pipeline)과
`models/metadata.json`이다. metadata에는 모델 이름, feature 목록, 범주형 feature의
학습 범주값, target, 학습 시각, `synthetic_data: true`, 분할 기간, 합성 test 지표,
데이터 SHA-256, **모델 파일 SHA-256**, 라이브러리 버전을 적는다. FastAPI 서버는 모델
파일 해시가 metadata와 맞을 때만 로드하므로, 재학습하면 두 파일을 함께 옮긴다.

## 8. 평가 지표

| 지표 | 뜻 |
|---|---|
| **MAE** | **주 지표.** 단위가 사람 수 그대로다 |
| RMSE | 큰 오차에 더 민감하다 |
| MAPE | 공휴일(약 100명)에서 크게 부풀므로 보조로만 본다 |
| bias | 평균(예측 - 실제). 양수는 과대 예측(음식이 남는다), 음수는 과소 예측(사람이 못 먹는다). 비용이 비대칭이므로 부호를 따로 본다 |
| MAE_excl_holiday | 공휴일을 뺀 MAE. baseline이 공휴일에서 크게 틀리는 효과를 분리해 본다 |

**이 합성 환경의 이론적 하한.** 오늘 noise(표준편차 22)와 두 잠재 요인의 새
충격(15.9, 8.4)은 D-1에 알 방법이 없다. 합치면 표준편차 약 28.4명이다. 그래서
전날까지의 잠재 상태를 **완벽히** 안다 해도 공휴일을 뺀 MAE는 약 23명 아래로
내려갈 수 없다. 실제로는 잠재 상태를 잡음 섞인 과거 식수로 추정해야 하므로
하한은 더 높다. 공휴일 제외 MAE가 이보다 낮게 나오면 leakage부터 의심한다.

## 9. Ablation 설명

| 단계 | feature | 수 |
|---|---|---|
| M1 | 과거 식수(`lag_1`, `lag_7`, `rolling_mean_7`) + 요일 | 4 |
| M2 | M1 + `semester`, `semester_week`, `exam_period`, `holiday`, `campus_event` | 9 |
| M3 | M2 + 날씨 3개 + 메뉴 4개 | 16 |
| M4 | M3 + CCTV 과거 이용 패턴 3개 | 19 |

RF와 LightGBM 둘 다 같은 분할과 같은 설정으로 돌린다(`outputs/ablation_metrics.csv`).

**이것은 실제 feature 효과를 입증하는 실험이 아니다.** 설계된 합성환경에서
파이프라인이 각 feature group을 받아 활용하는지 확인하는 점검이다. 합성데이터에서
어떤 group이 도움이 되는지는 생성 규칙이 이미 정해 두었다.

## 10. 결과 예시

아래는 전부 **합성데이터 위의 숫자**다. 실제 성능이 아니다.

분할: train 387행(2023-03-13 ~ 2025-09-02), validation 83행(2025-09-03 ~
2026-03-09), test 83행(2026-03-10 ~ 2026-09-11). 공휴일은 각각 22일, 5일, 3일이다.

### 모델 비교 (`outputs/model_metrics.csv`)

test: train+validation으로 학습해 한 번 평가한 값.

| model | 역할 | MAE | RMSE | MAPE(%) | bias | MAE (공휴일 제외) | validation MAE |
|---|---|---|---|---|---|---|---|
| lag7_baseline | baseline | 56.11 | 85.63 | 22.11 | +3.60 | 48.62 | 78.55 |
| rolling7_baseline | baseline | 47.51 | 69.14 | 19.34 | +2.29 | 39.72 | 60.23 |
| random_forest | 후보 | 32.96 | 40.40 | 10.15 | +8.45 | 32.60 | 28.44 |
| **lightgbm** | **최종 (validation 기준)** | **32.35** | **39.18** | **9.52** | **+10.81** | **32.55** | **26.50** |
| lag1_baseline | 참고 | 64.07 | 94.88 | 24.59 | +0.60 | 55.80 | 46.84 |
| ridge | 참고 | 28.52 | 37.34 | 8.37 | +10.47 | 28.66 | 29.19 |

읽는 법:

- **baseline을 크게 이긴 이유의 상당 부분은 공휴일·시험기간이다.** `lag_7`은
  지난주 공휴일의 급감을 이번 주에 그대로 옮긴다(`outputs/test_predictions.png`에서
  주황 선이 공휴일 1주 뒤에 떨어진다). 공휴일을 빼면 격차가 줄어든다(lag7 48.6 →
  LightGBM 32.6).
- **RF와 LightGBM은 test에서 사실상 같다**(32.96 vs 32.35). validation에서
  LightGBM이 2명 낮아 최종으로 골랐다.
- **참고용 선형 모델(Ridge)이 test에서 가장 낮다.** 생성식이 대부분 더하기로 되어
  있어서 선형 모델이 그 구조에 잘 맞는다. 이것은 합성데이터의 성질이다. 실제
  데이터에서도 선형 모델이 낫다는 뜻이 아니다.
- **학습 모델이 전부 test에서 +8~+11명 과대 예측한다.** test의 대부분인 2026-1학기의
  평균이 383.7명으로 직전 1학기(398.7명)보다 낮다(학기 수준 요인). 모델이 학기
  수준 변화를 과거 식수로 천천히 따라가서 생기는 과대 예측이다. 실제로도 재학생
  수가 바뀌면 같은 일이 생길 수 있으므로 운영 단계에서는 bias를 따로 감시해야 한다.
- validation에서 test로 가며 LightGBM MAE가 26.5에서 32.4로 나빠졌다. 83행짜리
  구간 하나씩이라 이 정도 흔들림은 흔하다.

### Ablation (`outputs/ablation_metrics.csv`)

| 단계 | RF validation | RF test | LightGBM validation | LightGBM test |
|---|---|---|---|---|
| M1 과거 식수 + 요일 | 52.44 | 48.57 | 57.63 | 57.09 |
| M2 + 학사일정 | 33.34 | 33.32 | 32.35 | 36.23 |
| M3 + 날씨 + 메뉴 | 28.11 | 32.46 | 26.08 | 33.61 |
| M4 + CCTV 과거 이용 패턴 | 28.44 | 32.96 | 26.50 | 32.35 |

MAE(명). 파이프라인이 각 group을 쓰는지 확인하는 용도다.

- **M1 → M2가 가장 크다**(약 -15 ~ -25명). 공휴일과 시험기간을 알게 된 효과다.
- **M2 → M3은 validation에서 뚜렷하다**(RF -5.2, LightGBM -6.3). test에서는 작다.
- **M3 → M4(CCTV)는 판정할 수 없다.** RF는 두 구간 모두 0.3~0.5명 나빠졌고,
  LightGBM은 validation에서 0.4명 나빠지고 test에서 1.3명 좋아졌다. CCTV가 과거 식수에 없는 정보를 갖도록 **설계했는데도** 이
  데이터 규모(학습 387~470행)에서는 트리 모델이 그것을 안정적으로 쓰지 못한다.
  즉 이 PoC는 합성 환경에서조차 CCTV의 기여를 보여 주지 못했다. 실제 효과는
  말할 것도 없다.

### 단일 날짜 예측

    $ python src/predict.py --date 2026-09-15 --menu-category 양식 --main-menu 돈까스 \
        --temperature 24 --rainfall 0 --humidity 60 --semester 2 --semester-week 3 --exam-period 0
    [합성데이터 PoC] 설계된 가정으로 생성한 데이터에서 나온 값이다. 실제 급식소의 식수나 예측 성능을 뜻하지 않는다.

    예상 중식 식수: 440명
    참고 범위: 381~477명  (합성 test 구간 잔차의 10~90% 분위수를 더한 값. 통계적 신뢰구간이 아니다)

    입력 조건:
    - 2026-09-15 화요일
    - 2학기 3주차
    - 양식 / 돈까스 (메뉴 선호도 0.93, 시뮬레이션 가정값)
    - 기온 24℃, 강수 없음, 습도 60%
    - 시험기간 아님, 공휴일 아님, 교내 행사 없음

    과거 feature (예측일 이전 기록만 사용, 합성 이력의 최근 기록일 2026-09-11):
    - lag_1 (직전 운영일 식수): 350.0  [합성 이력]
    ...

    주의:
    - 직전 운영일(2026-09-14) 기록이 합성 이력에 없어 가장 최근 기록(2026-09-11)을
      과거 feature로 썼다. 실제 운영이라면 D-1 당일의 식수와 CCTV 값을 넣어야 한다 ...

같은 날(2026-09-14, 돈까스, 맑음, 기본 예측 425명)에서 한 조건만 바꾸면 이렇게
움직였다. 설계한 방향과 맞는지 보는 점검이다.

| 바꾼 조건 | 예측 | 차이 |
|---|---|---|
| 비 25mm, 습도 85% | 402명 | -23 |
| 기타 / 죽 | 374명 | -51 |
| 시험기간 | 389명 | -36 |
| 교내 행사 | 442명 | +17 |
| 공휴일 | 149명 | -276 |

### 그래프

`python src/eda.py`와 `python src/train.py`를 돌리면 생긴다(`outputs/`는 커밋하지 않는다).

| 파일 | 내용 |
|---|---|
| `outputs/eda/01_meal_count_distribution.png` | 식수 분포 (평일/공휴일) |
| `outputs/eda/02_meal_count_timeseries.png` | 일별 식수와 train/validation/test 구간 |
| `outputs/eda/03_weekday_mean.png` | 요일별 평균 |
| `outputs/eda/04_semester_week_mean.png` | 학기 주차별 평균, 시험 주차 표시 |
| `outputs/eda/05_menu_category_mean.png` | 메뉴 카테고리별 평균 |
| `outputs/eda/06_rainfall.png` | 강수 여부·강수량과 식수 |
| `outputs/eda/07_exam_period.png` | 시험기간 여부별 분포 |
| `outputs/eda/08_feature_correlation.png` | 수치형 feature와 식수의 상관계수 |
| `outputs/test_predictions.png` | test 구간 실제 vs 최종 모델 vs lag7 baseline |

모든 그래프 하단에 "합성데이터" 표기가 들어간다. 그래프를 보고서로 옮겨도 이
표기가 따라가게 하려는 것이다. 표로 된 값은 `outputs/eda/eda_summary.txt`와
`outputs/eda/correlation_matrix.csv`에 있다.

## 11. 한계

**데이터**

- 전부 합성데이터다. 요인 사이의 관계와 효과 크기는 `CONFIG`가 정했다. 모델이
  찾아낸 "비가 오면 준다", "시험기간에 준다"는 우리가 넣은 규칙을 되찾은 것이지
  발견이 아니다.
- 메뉴 선호도는 가정값이다. 실제로는 선호도를 잴 source부터 없다. 이 PoC는 그 값을
  "안다"고 가정한다.
- 날씨는 예보 오차가 없는 것으로 두었다. 실제 D-1 예보는 틀리므로 날씨의 쓸모를
  과대평가한다.
- CCTV는 설계된 잠재 요인에서 만들었다. 실제 CCTV의 계수 오차, 카메라 가림, 결측이
  없다.
- 학사일정은 호서대의 실제 일정이 아니고, 공휴일 목록은 근사다.
- 학기 첫 주의 `lag_1`, `lag_7`, `rolling_mean_7`은 지난 학기 말(시험 주간 포함)
  값이다. D-1에 실제로 알 수 있는 값이긴 하지만 품질이 나쁘다.
- 생성 요인 비중이 요청 가이드와 다르다(날씨 4.9%, 5절).

**평가**

- validation과 test가 각각 83행이다. 1~2명 차이는 판정할 수 없다.
- 분할이 하나다. rolling-origin 평가(TimeSeriesSplit)는 하지 않았다.
- 하이퍼파라미터를 탐색하지 않았다.
- 참고 범위는 test 83행 잔차의 분위수를 전체 데이터로 재학습한 모델에 그대로
  더한 것이다. 보정된 예측 구간이 아니다.

**범위**

- 중식만 다룬다. 식당 하나만 가정한다.
- 이 트랙의 핵심 연구 질문은 아직 답하지 않았다. **신규 급식소에서 며칠 정도의 과거
  데이터가 있어야 쓸 만한 예측이 가능한가**, 그리고 과거가 0일인 cold-start에서
  무엇을 낼 것인가다. 이 PoC는 3년치가 있다고 가정한다.
- FastAPI 서버로 서빙은 하지만 Spring Boot는 아직 부르지 않는다. 합성데이터
  모델이므로 `APP_ENV=production`에서는 서버가 로드하지 않는다.

## 12. 향후 실제 데이터 확보 후 계획

1. **실제 식수의 canonical source를 정한다.** 식권·입장 기록 등. 이것이 선행 조건이다.
2. 그때까지 AI-1의 **entry count**가 식수의 대리 지표가 될 수 있는지 검증한다.
   검증 전에 정답으로 쓰면 모델이 카메라의 계수 오차를 학습한다.
3. 파일럿에서 실제 데이터를 쌓는다. 식수, 실제 학사일정, 식단표, D-1 시점의 기상
   **예보값**(관측값이 아니다), CCTV 일일 로그를 모은다.
4. `generate_synthetic_data.py`를 같은 컬럼을 내는 실제 적재 코드로 바꾼다.
   `history_features()`와 leakage 검증은 그대로 쓴다.
5. 메뉴 선호도는 가정값 대신 **예측일 이전 기록만으로** 계산한다(예: 그 메뉴가
   나온 과거 날들의 평균 식수 대비 편차).
6. baseline 계단부터 다시 돈다(전날 → 지난주 같은 요일 → 이동평균 → 같은 요일
   이동평균 → 선형/트리). 그리고 **데이터 양을 축으로** 표를 만든다.

   | 과거 데이터 | 7일 | 14일 | 30일 | 60일 | 90일 |
   |---|---|---|---|---|---|
   | baseline 1 (전날) | | | | | |
   | baseline 4 (요일 이동평균) | | | | | |
   | 모델 | | | | | |

   "며칠부터 모델이 baseline을 이기는가"가 답이다. cold-start는 따로 본다.
7. 기관별로 재학습하고, rolling-origin 평가, bias 감시, 보정된 예측 구간을 붙인다.
8. 그 뒤에 `AIDemandRequest` 계약을 새 버전으로 확장하고 `DemandEstimator`를
   교체한다.

## 폴더 구조

```
research/demand_forecasting/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ common.py                   경로, feature 정의, 분할, 지표, D-1 이력 feature
│  ├─ generate_synthetic_data.py  CONFIG(가정값) + 생성 + leakage 검증
│  ├─ eda.py
│  ├─ train.py
│  └─ predict.py
├─ data/                          (git 제외, 생성물)
│  ├─ synthetic_meal_demand.csv
│  └─ synthetic_cctv_daily.csv
├─ models/                        (git 제외, 생성물)
│  ├─ meal_demand_model.joblib
│  └─ metadata.json
└─ outputs/                       (git 제외, 생성물)
   ├─ model_metrics.csv
   ├─ ablation_metrics.csv
   ├─ test_predictions.csv / .png
   ├─ synthetic_effect_shares.csv
   └─ eda/
```

루트의 `src/`, `data/`가 아니라 이 폴더에 둔 이유가 있다. 루트 `src/`는 FastAPI
추론 서버이고, `tests/test_service_does_not_train.py`가 그곳에 학습 코드와 학습
프레임워크 import가 들어오는 것, 그리고 루트 `data/`에 파일이 생기는 것을 막는다.
이 폴더를 통째로 지워도 서버는 그대로 뜬다(식수 예측만 unavailable/503).

## 관련

- [research/README.md](../README.md)
- [cctv_counting](../cctv_counting/) — entry count가 여기의 정답 후보다
- 루트 README의 "수요 예측: 정답 데이터가 없다"
