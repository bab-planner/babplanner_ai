# babplanner-ai

밥플래너 AI 서비스. 잔반 분석, 수요 예측, 발주 추천의 내부 경계다.

서비스 백엔드(Spring)만 호출한다. 사용자 앱이 직접 호출하는 endpoint는 없다.

## 현재 상태

**세 기능 모두 실제 모델이 없다.** 지금 있는 것은 프로세스 경계와 계약뿐이다.

| 기능 | 구현 | 돌려주는 값 |
|---|---|---|
| 잔반 분석 | `FixtureWasteEstimator` | 이미지를 판독하지 않고 고정값 |
| 수요 예측 | `FixtureDemandEstimator` | 입력과 무관하게 고정된 인원 수 |
| 발주 추천 | `FixtureProcurementRecommender` | 계획 수량 그대로, 조정률 0 |

이 값들을 분석 결과나 예측으로 읽으면 안 된다. `modelVersion`이
`fastapi-fixture-*`로 시작하면 fixture가 낸 값이다.

교체 지점은 `WasteEstimator` / `DemandEstimator` / `ProcurementRecommender`
셋이다. HTTP 계층과 인증은 그대로 두고 구현만 바꾸면 된다.

### 수요 예측: 정답 데이터가 없다

이건 모델을 아직 안 만들었다는 것과는 다른 문제다.

수요 예측이 맞히려는 값은 **실제 급식 이용 인원**이다. 그런데 지금 시스템
어디에도 그 값이 정답으로 쌓이는 곳이 없다.

- `congestion_observations.peopleCount`는 특정 시점에 관측된 혼잡도 지표다.
  한 끼를 실제로 먹은 사람 수가 아니다. 이 값을 정답으로 쓰면 관측 시각과
  관측 방식에 따라 정답이 흔들린다.
- 잔반 제출 건수는 인증에 참여한 사람 수다. 이용자 전체가 아니다.
- 식권/입장 기록 같은 canonical source는 아직 없다.

그래서 `AIDemandRequest`는 과거 관측치 배열을 받지 않는다. 어느 기간의
무엇을 집계한 값인지 정의된 적이 없는 배열을 계약에 두면, 보내는 쪽은 모델이
그것을 쓰고 있다고 믿고 받는 쪽은 무시하게 된다.

**실제 식수 ground truth source가 생기기 전까지 수요 모델 학습은 불가능하다.**
feature 계약 확장은 그 source가 정해진 뒤 별도 version에서 한다.

발주 추천도 마찬가지다. 계획 발주량의 canonical source가 백엔드에 없어서,
관리자가 요청 본문으로 명시하도록 계약을 정했다.

## AI 연구 트랙

서비스 계약(`src/`) 뒤에 들어갈 모델을 만드는 곳은 `research/`다. 이 폴더를
통째로 지워도 FastAPI 서비스는 그대로 뜬다.

| | 트랙 | 상태 |
|---|---|---|
| AI-1 | [CCTV 혼잡도·이용 인원](research/cctv_counting/) | **공개 데이터 cross-sequence 평가 완료.** 급식소 검증은 아직 |
| AI-2 | [잔반 분석](research/waste_estimation/) | 계획. 정의만 있고 모델 코드 없음 |
| AI-3 | [식수 예측](research/demand_forecasting/) | 계획. 정의만 있고 모델 코드 없음 |

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

환경 변수는 셸에만 둔다. 파일로 남기지 않는다.

    APP_ENV=local
    WASTE_MODEL_MODE=fixture
    DEMAND_MODEL_MODE=fixture
    PROCUREMENT_MODEL_MODE=fixture
    AI_SERVICE_TOKEN=<secret>

    uvicorn babplanner_ai.main:app --host 127.0.0.1 --port 8001

`WASTE_MODEL_MODE`에 기본값은 없다. 운영에서 `fixture`를 고르면 기동에
실패한다. 실제 모델 어댑터가 없는 동안에는 그것이 정상이다.

`DEMAND_MODEL_MODE`와 `PROCUREMENT_MODEL_MODE`의 기본값은 `disabled`다.
`disabled`는 fixture가 아니라 "꺼짐"이며, 호출하면 503이 나간다. 끈 기능이
fixture로 대신 답하면 예측이 없는 배포가 예측을 하고 있는 것처럼 보인다.

## 내부 API

    POST /internal/v1/ai/waste/analyze
    POST /internal/v1/ai/demand/predict
    POST /internal/v1/ai/procurement/recommend

셋 다 `X-Service-Token` 헤더를 요구한다. 없거나 틀리면 똑같이 401이다.
이 서비스는 DB도 파일도 외부 네트워크도 쓰지 않는다. 보관은 백엔드가 한다.

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
