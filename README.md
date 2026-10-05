# babplanner-ai

밥플래너 AI 추론 서버. **다음 운영일 중식 식수 인원 예측**의 내부 경계다. 최종 AI 범위는 **CCTV 혼잡도**(영상 수집기가
Backend로 push)와 **식수 인원 예측** 둘이다. 잔반 사진은 AI가 분석하지 않는다. 사용자가 제출하면
Backend가 관리자 승인 대기(PENDING)로 두고, 관리자가 사진을 보고 승인/거절한다.

## 아키텍처

```
Flutter 앱 / 관리자 Web
        ↓
Spring Boot 메인 백엔드      인증·권한, 메뉴·학사·운영 데이터, DB, 비즈니스 로직, 프론트엔드 API
        ↓ 내부 HTTP (X-Service-Token)
FastAPI AI 추론 서버 (이 저장소)   저장된 모델을 로드해 추론만 한다
        ↓
식수 예측 모델 artifact       research/demand_forecasting 에서 학습·저장
```

- **메인 API와 백엔드는 Spring Boot다.** 이 서버는 그것을 대체하지 않는다.
- **이 서버는 AI 추론 전용이다.** 서비스 백엔드(Spring)만 호출한다. 사용자 앱이
  직접 호출하는 endpoint는 없고, 인터넷에 직접 공개하는 구성을 전제로 하지 않는다.
- **사용자 인증·관리자 권한은 Spring이 맡는다.** 여기서는 서비스 간 토큰
  (`X-Service-Token`)으로 호출자가 Spring인지만 확인한다.
- **학습은 `research/demand_forecasting`, 서빙은 이 FastAPI 서버**가 한다. 서버는
  학습 코드를 import하지 않고, 저장된 artifact(joblib + metadata.json)만 읽는다.
- **지금 식수 예측 모델은 합성데이터 기반 PoC다.** 실제 운영에서는 실데이터로
  학습한 모델로 교체한다. 교체는 artifact 파일을 바꾸는 것으로 끝나게 만들어 두었다.
- Docker에서는 `spring-backend`와 `ai-server`를 별도 컨테이너로 두는 구성을 전제로 한다.

## 현재 상태

**실제 데이터로 학습한 모델은 하나도 없다.**

| 기능 | 구현 | 돌려주는 값 |
|---|---|---|
| **식수 예측 (demand-forecast)** | 저장된 LightGBM Pipeline | **합성데이터로 학습한 PoC 모델의 예측값** |
| CCTV 혼잡도 | 영상 수집기(`demo/cctv_collector`) | 사람 수와 혼잡 단계를 Backend로 push. 이 서버를 거치지 않는다 |

이 서버의 runtime 경로는 `/health`와 식수 예측 status/predict 셋뿐이다. 잔반 분석,
일반 수요 예측, 발주 추천, 추천 같은 다른 AI 경로는 두지 않는다. 이를
`tests/test_runtime_ai_scope.py`가 소스로 고정한다.

demand-forecast의 값은 모델이 실제로 계산한 예측이지만, 학습 데이터가 합성이다.
응답의 `model.syntheticData: true`와 `disclaimer`가 그것을 알린다. 이 값을 실제 급식
운영 정확도로 소개하면 안 된다. 그래서 `APP_ENV=production`에서는 합성데이터 모델을
기본적으로 로드하지 않고(status unavailable, predict 503), `DEMAND_FORECAST_MODE=poc`로
**명시했을 때만** 로드한다.

모델을 실데이터 모델로 바꾸는 지점은 artifact 파일(joblib + metadata.json)이다. HTTP
계층과 인증은 그대로 두고 artifact만 교체하면 된다. 이 extra 없이도 서버는 뜨고, 식수
예측만 쓸 수 없다고 답한다.

artifact와 합성 이력은 git에 없다. 처음이면 아래 "식수 예측 PoC artifact 만들기"로
만든다.

### 식수 예측: 정답 데이터가 없다

식수 예측이 맞히려는 값은 **실제 급식 이용 인원**이다. 그런데 지금 시스템 어디에도 그
값이 정답으로 쌓이는 곳이 없다.

- `congestion_observations.peopleCount`는 특정 시점에 관측된 혼잡도 지표다. 한 끼를
  실제로 먹은 사람 수가 아니다. 이 값을 정답으로 쓰면 관측 시각과 방식에 따라 정답이
  흔들린다.
- 잔반 제출 건수는 인증에 참여한 사람 수다. 이용자 전체가 아니다.
- 식권/입장 기록 같은 canonical source는 아직 없다.

**실제 식수 ground truth source가 생기기 전까지 실데이터 모델 학습은 불가능하다.**
그래서 합성데이터 PoC로 파이프라인이 끝까지 도는지 보이는 데까지만 했다. 이 계약은
과거 관측치 배열을 받지 않는다. 과거 feature는 AI 서버가 자기 이력 저장소에서 예측일
**이전** 기록만으로 만든다.

## AI 연구 트랙

서비스 계약(`src/`) 뒤에 들어갈 모델을 만드는 곳은 `research/`다. 이 폴더를
통째로 지워도 FastAPI 서비스는 그대로 뜬다.

| | 트랙 | 상태 |
|---|---|---|
| AI-1 | [CCTV 혼잡도·이용 인원](research/cctv_counting/) | **공개 데이터 cross-sequence 평가 완료.** 급식소 검증은 아직 |
| (보관) | [잔반 분석 연구](research/waste_estimation/) | **서비스 범위 밖.** 과거 연구 기록으로만 남아 있고 runtime 경로는 없다(잔반 인증은 Backend의 관리자 승인) |
| AI-3 | [식수 예측](research/demand_forecasting/) | **합성데이터 PoC.** 저장 모델을 demand-forecast API로 서빙. 실제 식수 데이터 0건이라 실제 모델 없음. Spring Backend가 호출한다 |

AI-1은 영상에서 현재 인원(occupancy)과 입구 통과 인원(entry count)을 따로
낸다. 두 값은 다르다. 점심에 300명이 다녀가도 한 순간의 인원은 40명일 수 있다.

**첫 정량 파일럿을 마쳤다. 다만 매우 작은 표본이다.** 공개 보행자 영상 한 편
(79.5초)을 앞 30초(dev)와 뒤 50초(holdout)로 나누고, 사람이 직접 센 정답으로
채점했다. holdout 평가는 설정을 고른 뒤 **한 번만** 했다.

holdout: Occupancy MAE **0.46명**, 입장 인원 오차 **1**, 퇴장 오차 6.
GPU(RTX 4060 Ti)에서 **31 fps**.

이어서 공개 데이터(CAVIAR, CC BY-SA)의 **서로 다른 영상 13편**으로 다시
검증했다. 영상 단위로 DEV 5편 / TEST 8편을 나누고, 설정을 고정한 뒤 TEST를
한 번만 평가했다.

두 가지가 나왔다.

**혼잡도에는 추적이 필요 없다.** 13편 전부에서 detection-only와 추적의 계수가
같았고, TEST macro MAE 차이는 0.008명(사전 동등 마진 0.25명)이었다. 그래서
**혼잡도는 검출만, 입장 인원은 추적 + 선 통과**로 갈래를 나눴다.

**실패 유형은 도메인에 따라 뒤집힌다.** 실외 영상에서는 통과 계수 오차가
전부 "없는 사건을 만든 것"이었는데, 저해상도 실내 영상에서는 전부
"있는 사건을 놓친 것"이었다. 한쪽만 보고 고쳤다면 다른 쪽에서는 효과가
없었을 것이다.

이어서 **다른 장소**(사무실 로비 17편)와 **다른 카메라**(같은 쇼핑센터의
corridor 시점 12편)에서 각각 한 번씩 더 평가했다. 매번 결과를 보기 전에
설정과 판정 규칙을 동결하고 지문(SHA-256)을 남겼다.

**두 번 다 가설이 기각되었고, 그것이 이 트랙의 성과다.** 해상도를 올리거나
더 큰 detector를 쓰면 누락이 회복될 것이라는 가설은 A5에서 기각되었다
(recall 0.578 → 0.463). 통과 누락이 궤적 연속성 문제라는 A6의 가설은 원인
분해로는 확인되었지만(그 원인의 실패 13건 중 11건 회복), 새 카메라에서
주 지표를 개선하지 못해 **개선된 tracker를 채택하지 않았다.**

**급식소 검증이 아니다.** 공개 데이터의 쇼핑센터와 사무실 로비다. 그리고
지금 수치로 발주를 움직이면 안 된다 — 통과 계수 recall이 0.6~0.8이고
occupancy가 프레임당 1.3명을 덜 센다. 조건과 한계는
[reports/cctv_baseline.md](reports/cctv_baseline.md),
[reports/cctv_a4_cross_video.md](reports/cctv_a4_cross_video.md),
[reports/cctv_a5_small_person.md](reports/cctv_a5_small_person.md),
[reports/cctv_a6_trajectory_continuity.md](reports/cctv_a6_trajectory_continuity.md)에
있다. 여섯 단계를 한 장으로 줄인 것은
[reports/cctv_research_summary.md](reports/cctv_research_summary.md)다.

**연구 트랙은 A6에서 닫았다.** 영상에서 관측을 만들어 백엔드로 보내는
시연 파이프라인은 [demo/cctv_collector/](demo/cctv_collector/)에 있다.
인원수를 혼잡도 **단계**로 바꾸는 규칙은 AI 출력이 아니라 식당별 운영
설정이고, 지금 저장소에 있는 값은 보정되지 않은 시연용이다.

**얼굴을 다루지 않는다.** 사람을 영상 밖에서 다시 알아보지 않고, 회원 정보와
연결하지 않는다. `track_id`는 영상 한 세션 안에서만 유효한 임시 번호이며 파일로
나가지 않는다. 이 경계는 테스트가 매 실행마다 확인한다.

핵심 로직은 추가 설치 없이 `pytest`로 돌지만, 실제 영상을 처리하려면 torch와
opencv가 필요하다. **그것은 이 저장소의 `.venv`가 아니라 별도 환경에 깐다.**
서비스 환경에 수 GB짜리 의존성을 섞지 않기 위해서다.

    py -3.12 -m venv .venv-cctv
    .venv-cctv\Scripts\python.exe -m pip install -e ".[cctv]"

## 실행

### 설치

    python -m venv .venv
    # Windows:      .venv\Scripts\activate
    # macOS/Linux:  source .venv/bin/activate
    python -m pip install -e ".[dev,demand-model]"

`demand-model`은 식수 예측 artifact를 읽는 데 필요한 ML 런타임이다(joblib,
scikit-learn, lightgbm, pandas, numpy). 버전은 artifact를 저장한 환경과 정확히
맞춰 두었다. 이 extra 없이도 서버는 뜨고, 식수 예측만 쓸 수 없다고 답한다.

### 식수 예측 PoC artifact 만들기 (clone 직후)

artifact(모델)와 합성 데이터(`research/demand_forecasting/{models,data,outputs}`)는 git에
없다. clone한 직후에는 아래 순서로 만든다. Python 3.12에서 검증했다.

    # 1. 위 "설치"를 마친 가상환경에서 학습 그래프용 matplotlib만 더한다
    python -m pip install matplotlib

    # 2. 합성 식수/CCTV 이력 생성 (+ leakage 검증)
    python research/demand_forecasting/src/generate_synthetic_data.py

    # 3. 학습·평가·모델 저장 (models/meal_demand_model.joblib, metadata.json)
    python research/demand_forecasting/src/train.py

    # 4. 서버 기동 후 확인 (아래 "기동" 참고)
    #    GET  /health
    #    GET  /internal/v1/ai/demand-forecast/status    -> available: true
    #    POST /internal/v1/ai/demand-forecast/predict

seed가 고정(42)이라 **같은 소스에서는 같은 데이터와 같은 모델이 나온다.** 새 가상환경에서
이 순서로 만든 결과의 합성 CSV 2개, `meal_demand_model.joblib`, 평가 지표 CSV가 기존
artifact와 byte 단위로 일치하는 것을 확인했다(`model_sha256` 동일, status의
`modelVersion`은 이 해시의 앞 12자리). metadata의 `training_date`만 실행 시각이라 다르다.
artifact를 만든 환경과 서버 환경의 라이브러리 버전이 다르면 status `warnings`에 나온다.

서버가 기동할 때마다 학습하지는 않는다. artifact는 한 번 만들어 두고 서버는 읽기만 한다.

### 컨테이너 배포 (Docker)

`Dockerfile`은 두 단계다. build 단계에서 위 2~3을 한 번 실행해 artifact를 만들고, runtime
이미지에는 추론에 필요한 고정 버전 의존성과 그 artifact(모델, metadata, 합성 이력)만
복사한다. matplotlib 같은 학습 도구는 runtime 이미지에 없다. 모델 binary를 git에 넣지
않아도 이미지가 재현된다.

    docker build -t babplanner-ai .
    docker run --rm -p 8000:8000 ^
        -e APP_ENV=production -e AI_SERVICE_TOKEN=<secret, 32자 이상> ^
        -e DEMAND_FORECAST_MODE=poc babplanner-ai

- `AI_SERVICE_TOKEN`과 `APP_ENV`는 이미지에 넣지 않고 실행할 때 준다.
- **`DEMAND_FORECAST_MODE=poc`를 명시해야** production에서 합성 PoC 모델이 서비스된다.
  빼면 status는 `available: false`, predict는 503이다(합성 모델이 운영에 조용히 나가지
  않게 하려는 정책이다).
- 실데이터 모델이 생기면 Dockerfile의 build 단계(generate/train)를 실제 artifact를
  복사하는 것으로 바꾸면 된다. 서버와 API는 그대로다.

### 기동

환경 변수는 셸에만 둔다. 파일로 남기지 않는다.

    APP_ENV=local
    AI_SERVICE_TOKEN=<secret, 32자 이상>

    uvicorn babplanner_ai.main:app --host 127.0.0.1 --port 8001 --reload

컨테이너(`ai-server`)에서는 `--host 0.0.0.0 --port 8000`으로 띄우고 포트를 외부에
공개하지 않는다. Spring 컨테이너만 같은 네트워크에서 부른다.

식수 예측 artifact 위치는 기본값이 `research/demand_forecasting/` 아래이고, 환경
변수로 바꾼다. 컨테이너에서는 마운트한 경로를 준다.

| 변수 | 기본값 |
|---|---|
| `DEMAND_FORECAST_MODEL_PATH` | `research/demand_forecasting/models/meal_demand_model.joblib` |
| `DEMAND_FORECAST_METADATA_PATH` | `research/demand_forecasting/models/metadata.json` |
| `DEMAND_HISTORY_PATH` | `research/demand_forecasting/data/synthetic_meal_demand.csv` |
| `DEMAND_CCTV_LOG_PATH` | `research/demand_forecasting/data/synthetic_cctv_daily.csv` |

### Swagger

`http://127.0.0.1:8001/docs`. demand-forecast 경로는 각 요청의 `X-Service-Token`
칸에 `AI_SERVICE_TOKEN` 값을 넣고 "Try it out"으로 부른다. 요청 예시가 채워져 있다.

운영(`APP_ENV=production`)에서는 이 서비스의 `/docs`와 `/openapi.json`이 닫힌다.

`DEMAND_FORECAST_MODE`(`strict` 기본 | `poc`)는 합성데이터 식수 예측 모델을 운영에서
쓸지 정한다. `strict`에서는 `APP_ENV=production`이면 합성 모델을 로드하지 않고
`poc`를 **명시했을 때만** 로드한다. 합성 PoC 모델임은 status의
`modelKind: "SYNTHETIC_POC"`, `syntheticData: true`와 응답의 `disclaimer`로 계속
드러난다.

## 내부 API

    GET  /health
    GET  /internal/v1/ai/demand-forecast/status
    POST /internal/v1/ai/demand-forecast/predict

`/health`를 뺀 전부가 `X-Service-Token` 헤더를 요구한다. 없거나 틀리면 똑같이
401이다. `/health`는 컨테이너 헬스체크용이고 `{"status": "ok"}` 외에 아무것도
돌려주지 않는다. 모델 파일이 없어도 200이다. 모델 사용 가능 여부는 status가 답한다.

이 서비스는 DB도 외부 네트워크도 쓰지 않고, 아무것도 쓰지(write) 않는다. 보관은
백엔드가 한다. 파일은 식수 예측 artifact와 이력을 기동 후 한 번 **읽기만** 한다.

### 식수 예측 API (demand-forecast)

**합성데이터로 학습한 PoC 모델이다.** 응답 값은 실제 급식 운영 정확도를 뜻하지 않는다.

    POST /internal/v1/ai/demand-forecast/predict
    X-Service-Token: <AI_SERVICE_TOKEN>
    Content-Type: application/json

```json
{
  "date": "2026-09-15",
  "semester": 2,
  "semesterWeek": 3,
  "examPeriod": false,
  "holiday": false,
  "campusEvent": false,
  "temperature": 24.0,
  "rainfall": 0.0,
  "humidity": 60.0,
  "menuCategory": "양식",
  "mainMenu": "돈까스",
  "specialMenu": false
}
```

모든 필드가 필수다. 서버가 빠진 값을 조용히 채우지 않는다. 모르는 필드(예: `lag_1`,
`weekday`, `userId`, 제거된 `menuPopularity`)는 422로 거절한다. 과거 feature를 보내도 쓰지 않는데, 조용히
무시하면 보낸 쪽은 쓰였다고 믿기 때문이다.

| 필드 | 검증 |
|---|---|
| `date` | 평일만. 주말은 422. 요일은 서버가 계산한다 |
| `semester` / `semesterWeek` | 1~2 / 1~16 |
| `temperature` / `rainfall` / `humidity` | -30~45℃ / 0~500mm / 0~100%. **D-1 예보값**이다(실제 당일 날씨가 아니다) |
| `menuCategory` | 한식·중식·양식·일식·분식·기타 |
| `mainMenu` | 1~50자. 모델이 모르는 메뉴면 예측은 하고 `warnings`에 적는다 |

아래는 실제 저장 artifact로 받은 응답이다.

```json
{
  "forecastDate": "2026-09-15",
  "mealType": "LUNCH",
  "predictedMealCount": 425,
  "reference": {
    "previousDay": 405,
    "sameWeekdayPreviousWeek": 405,
    "rolling7DayAverage": 426,
    "historyAsOf": "2026-09-11"
  },
  "conditions": {
    "weekday": "화요일", "semester": 2, "semesterWeek": 3,
    "examPeriod": false, "holiday": false, "campusEvent": false,
    "menuCategory": "양식", "mainMenu": "돈까스", "specialMenu": false,
    "temperature": 24.0, "rainfall": 0.0, "humidity": 60.0
  },
  "model": {
    "name": "lightgbm",
    "version": "04ffceab1186",
    "syntheticData": true,
    "trainedAt": "2026-10-05T23:59:59"
  },
  "warnings": [
    "직전 평일(2026-09-14) 기록이 이력에 없어 가장 최근 기록(2026-09-11)을 기준으로 과거 feature를 만들었습니다."
  ],
  "disclaimer": "합성데이터 기반 PoC 예측값이며 실제 급식 운영 정확도를 의미하지 않습니다."
}
```

- `predictedMealCount`는 0 이상 정수다(0.5는 올림, 음수는 0).
- `model.version`은 모델 파일 SHA-256의 앞 12자리다. 저장한 예측이 어느 모델에서
  나왔는지 가리는 데 쓴다.
- **예측 구간(95% 등)은 주지 않는다.** 통계적으로 보정된 구간이 없다.
- 같은 입력으로 research CLI(`predict.py ...`)를 돌리면 같은
  425명이 나온다.

| 상태 | 뜻 |
|---|---|
| 200 | 예측함 |
| 401 | 서비스 토큰 없음/틀림 |
| 422 | 요청 검증 실패, 또는 예측일 이전 이력이 모자람(과거 7운영일 미만 등) |
| 503 | 모델 artifact나 이력을 쓸 수 없음. `detail`에 이유. 숫자를 지어내지 않는다 |

    GET /internal/v1/ai/demand-forecast/status

```json
{"available": true, "modelLoaded": true, "historyAvailable": true,
 "modelName": "lightgbm", "modelVersion": "54453980807e", "target": "meal_count",
 "syntheticData": true, "trainingPeriod": ["2023-03-13", "2026-09-11"]}
```

artifact가 없으면 서버는 뜬 채로 이렇게 답한다(predict는 503).

```json
{"available": false, "modelLoaded": false, "historyAvailable": true,
 "reason": "model artifact not found"}
```

`reason`에는 파일 경로를 담지 않는다.

### 모델 artifact 로딩

- 기동할 때 **한 번** 로드하고 결과(성공이든 실패든)를 기억한다. 요청마다 파일을
  다시 읽지 않는다. artifact를 바꾸면 서버를 다시 띄운다.
- metadata를 먼저 검증한다. `synthetic_data`가 불리언으로 명시되어 있어야 하고,
  `target`이 `meal_count`여야 하며, feature 목록이 서버가 만들 수 있는 것이어야 한다.
- metadata의 `model_sha256`과 모델 파일 해시가 맞을 때만 연다. joblib은 pickle이라
  여는 순간 코드가 실행될 수 있다. 짝이 안 맞는 파일은 열지 않는다. 경로는 운영자
  설정값이지 요청 값이 아니다.
- 모델이 기억하는 입력 컬럼(`feature_names_in_`)과 metadata `features`를 다시 맞춘다.
  예측 때도 입력 행의 컬럼이 metadata 순서와 같은지 확인한다.
- 학습 때와 라이브러리 버전이 다르면 status의 `warnings`에 적는다.
- **`APP_ENV=production`에서는 합성데이터 모델을 기본적으로 로드하지 않는다.**
  `DEMAND_FORECAST_MODE=poc`로 명시했을 때만 로드한다.

### 과거 feature

모델은 `lag_1`, `lag_7`, `rolling_mean_7`, `prev_avg_occupancy`,
`prev_peak_occupancy`, `same_weekday_prev_occupancy`가 필요하다. **Spring이 계산하지
않는다.** AI 서버가 `DemandHistoryRepository`에서 예측일 이전 기록을 받아 만든다.

- 지금은 `SyntheticDemandHistoryRepository`가 research의 합성 CSV를 읽는다. 실제
  운영에서는 DB나 Spring이 준 이력을 읽는 구현으로 바꾼다.
- 저장소는 `date < 예측일`인 기록만 돌려준다. 계산 함수는 예측일 당일이나 이후 기록이
  하나라도 들어오면 걸러 쓰지 않고 실패한다(503).
- 계산 규칙은 research의 `history_features()`와 같다. 테스트가 합성 학습 CSV 전
  546행에서 lag·CCTV 컬럼을 그대로 재현하는지, 그리고 학습 CSV 행을 모델에 직접
  넣은 예측과 API 예측이 같은지 대조한다.
- 직전 평일 기록이 이력에 없으면 가장 최근 기록을 쓰고 `warnings`에 적는다.

## 테스트

    pytest

## 데이터셋 도구

아직 모델이 없다. 먼저 무엇을 맞히려는 것인지부터 정해 두었다.

정답은 질량비다.

    waste_rate = leftover_edible_mass_g / served_edible_mass_g

사진의 면적 비율도, 사람의 눈대중도 정답으로 쓰지 않는다. 자세한 것은
`docs/waste_ground_truth_protocol.md`.

    python -m babplanner_ai.dataset validate --manifest ... --root ...
    python -m babplanner_ai.dataset summarize --manifest ...
    python -m babplanner_ai.dataset make-splits --manifest ... --out ...

- `docs/waste_ground_truth_protocol.md` — 잔반률의 정의와 측정 절차
- [reports/waste_w1_problem_definition.md](reports/waste_w1_problem_definition.md) — AI-2 문제 정의(W1)
- [reports/waste_w2_external_benchmark.md](reports/waste_w2_external_benchmark.md) — AI-2 외부 벤치마크(W2)
- `docs/waste_dataset_spec.md` — 매니페스트 열, 개인정보 규칙, 분할 기준
- `docs/waste_evaluation_protocol.md` — 지표와 Go/No-Go
- `data/README.md` — 데이터 배치와 커밋 금지 규칙

## 현장 수집 도구

파일럿에서 실제 정답을 모으는 쪽이다. 저울 숫자와 사진을 받아 원시 기록으로
남기고, EXIF를 벗긴 사본을 만들고, 위 계약에 맞는 매니페스트로 내보낸다.

    python -m babplanner_ai.collection init-session --root data/private ...
    python -m babplanner_ai.collection add-sample --root data/private ...
    python -m babplanner_ai.collection qc --root data/private ...
    python -m babplanner_ai.collection export --root data/private ...

원시 기록과 학습용 매니페스트를 나눠 둔다. 잘못 적은 값을 고치거나 정답
정의가 바뀌어도 원래 숫자에서 다시 만들 수 있어야 하기 때문이다.

- `docs/waste_pilot_collection_runbook.md` — 현장 담당자용 절차
- `docs/waste_pilot_checklist.md` — 인쇄용 체크리스트

**아직 실제 파일럿 데이터는 없다.** 저장소에 있는 것은 합성 fixture뿐이다.
