# CCTV Demo Collector

영상에서 혼잡도 관측을 만들어 Spring 백엔드로 보내는 시연용 수집기다.

```
CCTV Video ─→ Shared Person Detector ─┬─→ Occupancy Count ─→ Operational Level Policy ─→ Spring
                                      │
                                      └─→ Entry / Exit Count ─→ 로컬 화면·JSONL only
```

---

## 1. 무엇을 하는가

**연구가 아니다.** 여기서 임계값을 고르거나 모델을 바꾸지 않는다. A1~A6가
동결한 결과를 그대로 읽어 쓰고, 이 패키지가 더하는 것은 셋뿐이다.

| | |
|---|---|
| **표본 추출** | 프레임마다가 아니라 정해진 간격으로 관측 하나를 만든다 |
| **운영 정책** | 인원수를 혼잡도 단계로 바꾼다. **AI 출력이 아니다** |
| **전송** | Spring 내부 수집 경로로 보낸다 |

**계수 로직을 복사하지 않는다.** detector도 tracker도 dual-path도
`research/cctv_counting`의 것을 그대로 import한다. 복사본을 만들면 연구가
검증한 코드와 시연에 나가는 코드가 갈라진다.

**연구 임계값도 복사하지 않는다.** occupancy 신뢰도 임계값은 A6가 동결한
기록(`experiments/a6_continuity/final_test_freeze.json`)의 `baseline` 정책에서
읽는다. A6는 개선 tracker(`MotionAwareTracker`)를 만들고도 동결한 주 지표가
개선되지 않아 **채택하지 않았다.** 그 판단이 기록 파일 안에 있으므로,
수집기가 그 파일을 읽는 한 시연이 연구 결론과 어긋날 수 없다.

## 2. 구조

```
demo/cctv_collector/
├─ README.md
├─ cameras/
│  └─ demo_camera.example.yaml     카메라별 보정값과 임계값 (비밀 없음)
├─ src/cctv_collector/
│  ├─ config.py                    카메라 설정 + 환경 변수 + 채택된 연구 정책
│  ├─ congestion_policy.py         인원수 -> 단계. AI가 아니다
│  ├─ schemas.py                   Spring으로 나가는 것 / 로컬에만 남는 것
│  ├─ spring_client.py             POST, 타임아웃, 유한 재시도, 실패 분류
│  ├─ collector.py                 프레임 -> 표본 -> 관측
│  └─ cli.py                       진입점
└─ tests/
```

정책·계약·전송 로직은 **torch와 opencv 없이 import된다.** 영상 디코딩과
추론만 무거운 의존성을 쓰고 그 import는 함수 안에 있다. 그래서 서비스
환경(`.venv`)의 `pytest` 한 번으로 이 패키지가 전부 검증된다.

## 3. 연구 출력과 운영 정책의 경계

이 구분이 이 패키지에서 가장 중요하다.

| | 누가 정하는가 | 검증되었는가 |
|---|---|---|
| `peopleCount` | 연구 (A1~A6) | 공개 데이터에서 측정했다 |
| `level` | **운영 정책** | **측정된 적 없다** |

"몇 명부터 붐비는가"는 그 식당의 좌석 수와 회전율이 정한다. 100석에서
40명은 한산하고 50석에서는 붐빈다. 모델이 알 수 있는 값이 아니다.

그래서 임계값은 **코드에 없다.** `congestion_policy.py`에 `if count < 10`
같은 줄이 없고, 값은 카메라 설정 파일에서만 온다. 그리고 그 값이 어디서
왔는지가 `calibration` 필드로 함께 저장된다.

```yaml
congestion_policy:
  type: count_thresholds
  calibration: DEMO_ONLY_UNCALIBRATED   # <- 이 값이 기록에 같이 남는다
  comfortable_max: 5
  normal_max: 12
  crowded_max: 20
```

> **저장소에 있는 임계값은 시연을 위해 사람이 임의로 고른 값이다.**
> 학습된 AI 출력이 아니고, 실제 급식소에서 보정한 수용 인원도 아니다.
> 운영 기준으로 인용하면 안 된다.

경계값은 아래 단계에 포함된다. `comfortable_max: 5`면 5명은 COMFORTABLE이다.

## 4. 환경 변수

**비밀은 환경 변수로만 들어온다.** 설정 파일에도, 명령줄 인자에도 두지
않는다. 명령줄은 프로세스 목록과 셸 히스토리에 남는다.

| 변수 | 필수 | 예시 | 비밀 |
|---|---|---|---|
| `BAPPLANNER_BACKEND_BASE_URL` | `--send` 시 | `http://127.0.0.1:8080` | 아니오 |
| `BAPPLANNER_INTERNAL_SERVICE_TOKEN` | `--send` 시 | 32자 이상 | **예** |

**두 토큰을 혼동하지 않는다.**

| | 방향 | 쓰는 곳 |
|---|---|---|
| `BAPPLANNER_INTERNAL_SERVICE_TOKEN` | Spring이 **받는다** | 이 수집기가 `/internal/v1/**` 를 부를 때 |
| `AI_SERVICE_TOKEN` | Spring이 **보낸다** | Spring이 FastAPI의 `/internal/v1/ai/**` 를 부를 때 |

같은 값으로 둘 필요가 없고, 그러지 않는 편이 낫다.

기본 주소를 코드에 두지 않는다. production hostname이 박혀 있으면 실수로
시연 데이터가 운영 DB에 들어간다.

## 5. Dry run

**기본이 dry-run이다.** 그냥 돌렸는데 서비스 DB에 값이 쌓이지 않는다.

```bat
.venv-cctv\Scripts\python.exe -m cctv_collector ^
  --video path\to\demo.mp4 ^
  --config demo\cctv_collector\cameras\demo_camera.example.yaml ^
  --jsonl outputs\demo.jsonl
```

임계값 없이도 돈다. 그 경우 `level`이 비고 인원수만 나온다.

출력 한 줄:

```
video    2.04s  관측 2026-08-28T17:23:10+09:00  인원 2  COMFORTABLE  입장 0 퇴장 0  SKIPPED_DRY_RUN
```

`--limit-seconds 30` 으로 앞부분만 돌려 리허설할 수 있다.

## 6. Live send

```bat
set BAPPLANNER_BACKEND_BASE_URL=http://127.0.0.1:8080
set BAPPLANNER_INTERNAL_SERVICE_TOKEN=...

.venv-cctv\Scripts\python.exe -m cctv_collector ^
  --video path\to\demo.mp4 ^
  --config demo\cctv_collector\cameras\demo_camera.example.yaml ^
  --site-id 1 ^
  --send
```

`--send` 는 명시적 opt-in이다. **임계값이 없으면 전송을 거부한다.**
없는 임계값을 코드가 지어내면, 화면에 뜬 "여유"가 무엇을 근거로 한 말인지
아무도 설명할 수 없다.

보내는 것:

```
POST {base}/internal/v1/congestion-observations
X-Service-Token: <BAPPLANNER_INTERNAL_SERVICE_TOKEN>

{"siteId":1,"deviceId":"demo-cctv-01","peopleCount":8,
 "level":"NORMAL","observedAt":"2026-08-28T12:00:00.000+09:00",
 "modelVersion":"cctv-fasterrcnn-mobilenetv3-demo-v1"}

-> 202 Accepted (본문 없음)
```

`source`를 보내지 않는다. 백엔드가 `CCTV_AI`로 직접 기록한다. 보내는 쪽이
출처를 정하게 하면 저장된 출처를 신뢰할 수 없고, 계약이 그렇게 되어 있다.

### 두 개의 시간축

```
  source_video_timestamp_seconds   영상 안에서의 위치. 로컬 진단용
  observed_at                      수집기가 그 관측을 만든 실제 시각. 이것만 보낸다
```

녹화 영상을 재생하는 데모에서 **영상 안의 시각을 `observedAt`으로 보내면
안 된다.** 백엔드는 늦게 도착한 과거 관측이 현재값을 되돌리지 않도록
`observedAt`을 비교하므로(`CongestionIngestService.updateIfNotOlder`),
2005년에 찍힌 영상의 시각을 보내면 현재값이 영원히 갱신되지 않는다. 그것은
백엔드가 옳게 동작한 것이고 보내는 쪽이 틀린 것이다.

### 표본 간격

25fps 영상이라고 초당 25번 POST하지 않는다. 혼잡도는 초 단위로 뒤집히는
값이 아니고, 백엔드는 관측 하나마다 이력과 현재값 두 테이블에 쓴다.
기본은 1초이고 `observation.interval_seconds`로 바꾼다.

창 안의 여러 표본을 대표값 하나로 줄이는 방법은 둘뿐이다.

```
  latest   창의 마지막 표본
  median   창 표본의 중앙값(낮은 쪽). 정수를 유지하고 사람을 만들어 내지 않는다
```

결과를 좋아 보이게 하는 적응형 평활은 두지 않는다.

## 7. 카메라 보정

**ROI와 counting line은 카메라마다 다르다.** A6에서 확인한 것이 이것이다 —
카메라를 바꿨는데 ROI를 그대로 썼더니 정답 통과 사건의 **23%가 계수 영역
밖에서** 일어났다. tracker를 바꾸는 것보다 이쪽이 영향이 컸다.

그래서 새 카메라를 달면 먼저 눈으로 확인한다.

```bat
.venv-cctv\Scripts\python.exe -m cctv_collector ^
  --video path\to\demo.mp4 ^
  --config demo\cctv_collector\cameras\demo_camera.example.yaml ^
  --preview-frame 100 --preview-out outputs\preview.png
```

초록 사각형이 ROI, 빨간 선이 counting line이다. 사람이 실제로 지나는 곳을
선이 가로지르는지, 발끝이 ROI 안에 들어오는지 본다.

`line.in_direction`(어느 쪽이 입장인가)은 카메라 설치 방향이 정한다.
**현장에서 반드시 확인한다.** 반대로 두면 입장과 퇴장이 통째로 뒤집힌다.

## 8. 보안

- 토큰은 **환경 변수에서만** 읽는다. 설정 파일에 `token` 같은 키가 있으면
  기동을 거부한다.
- 토큰을 CLI 인자로 받지 않는다.
- `BackendSettings.__repr__`이 토큰을 가린다. 객체가 로그에 실려도 새지 않는다.
- HTTP 오류를 다룰 때 요청 객체를 통째로 찍지 않는다. 오류 메시지를 직접
  만들고, 그 안에 헤더가 들어가지 않는다.
- 이 수집기는 브라우저 코드가 아니다. 내부 토큰이 Flutter나 Admin JS에
  들어갈 경로가 없다.

## 9. 개인정보

Spring으로 나가는 것은 **집계된 숫자뿐**이다.

| 나간다 | 나가지 않는다 |
|---|---|
| 인원수, 판정 단계, 관측 시각, 장비 id | 프레임, bbox, track_id, 얼굴, 이름, 학번, 전화번호 |

`schemas.py`가 이 경계를 한 곳에 모으고, 테스트가 payload에 그런 이름이
없는지 매번 확인한다. `device_id`는 사람이 아니라 **카메라**를 가리킨다.

입장/퇴장 인원은 로컬 화면과 JSONL에만 남는다. 이유는 다음 절에 있다.

## 10. 입장 인원을 백엔드에 저장하지 않는 이유

백엔드 계약(`CongestionObservationRequest`)에 그것을 받을 자리가 **없다.**
계약에 없는 필드를 몰래 보내면 보내는 쪽은 쓰이고 있다고 믿고 받는 쪽은
무시한다. 그래서 보내지 않고, 이 Gap을 문서로 남긴다.

그리고 더 근본적인 문제가 있다.

> **line-crossing entry count ≠ 식수(meal attendance) 정답이다.**

같은 사람이 다시 들어오고(재입장), 직원과 방문객이 섞이고, 카메라 사각이
있다. A6 FINAL에서 통과 계수 recall은 0.6~0.8이었다. 이 값을 보정 없이
수요 정답으로 쓰면 실제보다 적게 세고, 그 정답으로 학습한 모델이 다시 적게
발주한다.

쓰려면 두 가지가 먼저 필요하다.

```
  1. meal attendance 집계 도메인       끼니 단위로 누적하고 보관하는 자리
  2. 대조 검증                          수기 계수 / 게이트 / POS 실적과 비교
```

**이번 단계에서 백엔드 계약을 늘리지 않았다.** 검증 없이 필드를 늘리는 것이
문제의 원인이지 해결이 아니다.

## 11. 시연 절차

```
  1. Spring 백엔드를 띄운다
  2. GET /api/v1/congestion/current?siteId=N  으로 현재값을 본다
  3. 수집기를 --send 로 짧게 돌린다
  4. 202 Accepted 를 확인한다
  5. 2번을 다시 호출해 peopleCount / level / observedAt 이 바뀐 것을 본다
  6. source 가 CCTV_AI 로 저장되었는지 본다
```

`site_id`를 **추측하지 않는다.** 실제 DB나 API에서 확인한 값을 쓴다.
`--site-id` 로 설정을 덮어쓸 수 있다.

### 실제로 한 번 통과시켰다 (2026-08-28)

로컬 MySQL + Spring 백엔드를 띄우고 CAVIAR 영상 12초를 재생했다.

```
  보낸 것        12건, 전부 202 Accepted
  detector 호출  301회 / 301프레임        (프레임당 정확히 한 번)

  현재 혼잡도 (before)   peopleCount 42, NORMAL,      source DEV_SEED
  현재 혼잡도 (after)    peopleCount  0, COMFORTABLE, source CCTV_AI

  congestion_observations   12행 추가, device demo-cctv-01,
                            modelVersion cctv-fasterrcnn-mobilenetv3-demo-v1
  congestion_status         siteId 1만 갱신. siteId 2는 DEV_SEED 그대로
```

`source`는 우리가 보내지 않았는데 `CCTV_AI`로 저장되었다. 계약대로 백엔드가
직접 기록한다.

### 영상 시각을 보냈다면 어떻게 되는지도 확인했다

같은 백엔드에 `observedAt`을 **2005년**(녹화 시각에 해당)으로 한 관측을
일부러 보냈다.

```
  HTTP 응답          202 Accepted       <- 성공처럼 보인다
  이력 테이블         행이 남았다
  현재 혼잡도         바뀌지 않았다      <- 2026-08-28 값이 그대로
```

**조용한 실패다.** 매 POST가 202를 받으므로 로그만 보면 정상인데, 앱의
"현재 혼잡도"는 영원히 갱신되지 않는다. 백엔드가 옳게 동작한 것이고
(`updateIfNotOlder`), 보내는 쪽이 틀린 것이다. 그래서 `observedAt`은
반드시 수집 시점의 실제 시각이어야 한다.

### 토큰 확인

잘못된 토큰으로 보내면 401이 오고, 그 결과는 `BACKEND_SEND_FAILED`로
분류된다(추론 실패가 아니다). 오류 메시지에도, 백엔드 로그에도 토큰 값이
남지 않는 것을 확인했다.

## 12. 한계

1. **급식소에서 검증되지 않았다.** A1~A6는 공개 감시 영상(CAVIAR)에서
   평가했다. 배식 줄의 밀집도, 쟁반을 든 자세, 조명, 카메라 높이가 모두 다르다.
2. **혼잡도 임계값이 보정되지 않았다.** 저장소의 값은 시연용이다.
3. **통과 계수 recall이 0.6~0.8이다.** 이 값으로 발주를 움직이면 안 된다.
4. **occupancy가 덜 센다.** A6 FINAL에서 bias가 프레임당 −1.3명이었다.
5. **재생 데모다.** 실시간 RTSP 스트림이 아니다. 스트림에서는 프레임 드랍과
   재연결을 다뤄야 하는데 여기에 없다.
6. **backpressure 전략이 단순하다.** 동기 전송이고 큐가 없다. 백엔드가
   느리면 다음 관측이 늦게 나갈 뿐이지만, 초당 여러 관측을 보내는 설정에서는
   맞지 않는다.
