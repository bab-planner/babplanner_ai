# CCTV Counting Baseline

*밥플래너 AI-1 — CCTV 기반 비식별 식당 혼잡도·이용 인원 분석*

작성 시점: 2026-08-28
상태: **pipeline 완성, 벤치마크 미측정**

> 이 보고서에 성능 수치가 비어 있는 곳은 아직 재지 않은 곳이다. 채워 넣지
> 않았다. 정답 데이터가 없는 상태에서 만든 숫자는 결과가 아니다.

---

## 1. Problem

급식소 운영에 두 개의 숫자가 필요하다.

**지금 붐비는가.** 앱 혼잡도 표시용. 현재 공간 내 인원수.

**오늘 몇 명이 먹었는가.** 식수 예측(AI-3)의 정답이 될 대리 지표. 그런데 지금
시스템 어디에도 실제 식수가 정답으로 쌓이는 곳이 없다.

두 값은 다르다. 점심에 300명이 다녀가도 한 순간의 인원은 40명일 수 있다.
**하나의 "인원 수"로 합치면 둘 다 뜻을 잃는다.**

그래서 과제를 둘로 나눈다.

| | Occupancy Estimation | Entry Counting |
|---|---|---|
| 정의 | 현재 프레임에서 공간 안에 있는 사람 수 | 입구 가상 선을 IN 방향으로 통과한 누적 인원 |
| 필요한 것 | 프레임 하나 | 프레임 사이의 연결(추적) |
| 쓰는 곳 | 실시간 혼잡도 | 한 끼 이용 인원의 대리 지표 |

### 제약

**사람을 식별하지 않는다.** 학생이 밥 먹는 장면을 찍는 카메라다. 필요한 것은
"몇 명"이지 "누가"가 아니다. 얼굴 검출·특징 추출·대조를 하지 않고, 같은 사람을
다른 영상에서 다시 알아보지 않으며, 회원 정보와 연결하지 않는다.

만들지 않은 정보는 유출되지 않고, 잘못 쓰이지 않고, 요구받아도 내줄 것이 없다.

## 2. Research Questions

**RQ1.** 공개 pretrained detector와 tracker만으로, 식당/보행 영상에서
occupancy estimation과 entry counting을 어느 정도까지 수행할 수 있는가?

**RQ2.** 조명, 가림, 밀집도, 카메라 각도 같은 domain condition에 따라 계수
오차가 어떻게 변하는가?

**RQ3.** detector/tracker의 confidence를 이용해 신뢰하기 어려운 관측을 식별할
수 있는가?

RQ3는 이번 단계에서 논문 수준의 uncertainty calibration까지 가지 않는다.
최소한의 신뢰 신호를 만들어 두고, 그것이 쓸모 있는지를 다음 단계에서 잰다.

| | 현재 상태 |
|---|---|
| RQ1 | pipeline 구현 완료. **정량 측정 대기** |
| RQ2 | 조건별 평가 코드 완료. **조건 annotation 있는 데이터 필요** |
| RQ3 | 신호 계산 구현. **flag와 실제 오차의 상관 미측정** |

## 3. Data

| 구분 | 상태 |
|---|---|
| PUBLIC BENCHMARK | 없음. 아직 아무것도 내려받지 않았다 |
| DEMO VIDEO | 없음. 사용자가 명시적으로 제공한 영상이 없다 |
| SYNTHETIC TEST DATA | 있음. 단위 테스트용 상자 좌표. 영상이 아니다 |
| **REAL CAFETERIA DATA** | **Not available** |

### 후보 조사

| 데이터셋 | 목적 | 라이선스 | 판단 |
|---|---|---|---|
| MOT17 | 추적 / 계수 | CC BY-NC-SA 3.0 (비상업) | **첫 벤치마크 후보.** 고정 카메라 시퀀스가 우리 조건에 가장 가깝고, track ID로 entry 정답을 유도할 수 있다 |
| MOT20 | 밀집 계수 | CC BY-NC-SA 3.0 (비상업) | 밀집도가 급식소 피크에 가깝다. 2순위 |
| CrowdHuman | 사람 검출 | 비상업 연구·교육. 재배포 금지 | 밀집·가림 검출 평가용. 정지 이미지라 추적 평가 불가 |
| COCO person | 사람 검출 | annotation CC BY 4.0 / 이미지는 개별 | detector 자신의 학습 분포라 낙관적으로 나온다 |
| PETS2009 | 흐름 계수 | **확인 못 함** | 확인 전까지 후보에서 제외 |

자세한 표는 `research/cctv_counting/datasets/README.md`.

### 비상업 라이선스가 실질적 제약이다

MOT17, MOT20, CrowdHuman 모두 비상업 조건이다. **평가에는 쓸 수 있지만, 이
데이터로 학습한 모델을 상용 서비스에 넣을 수는 없다.** 그래서 이 데이터들은
평가 전용으로 쓰고, fine-tuning은 나중에 상용 가능한 데이터로 한다.

MOT의 라이선스는 검색 결과 기준이고, 조사 시점에 `motchallenge.net` 직접
접속이 되지 않아 **공식 페이지 원문을 확인하지 못했다.** 받기 전에 사이트의
Terms를 직접 읽는다.

### 정답 파일

`datasets/annotations/`에 헤더만 있는 템플릿 두 개. occupancy와 crossing을
분리했고, 둘 다 신원 열이 없다. `condition` 열은 비어 있어도 되고, **비워 두는
것이 지어내는 것보다 낫다.**

## 4. Baselines

### A. Detection only

프레임마다 사람 상자를 세는 것이 곧 occupancy. **입장 인원은 셀 수 없다.**
같은 사람을 프레임 사이에서 잇지 못하면 "넘었다"는 사건 자체가 정의되지 않기
때문이다. `entries_total`은 0이 아니라 빈 값으로 나간다.

대신 **선 근처 상자 수 누적**(`naiveLineBandDetections`)을 기록한다. 한 사람이
선을 지나는 동안 매 프레임 다시 세이므로 실제보다 크게 나온다. **그것이
요점이다.** 이 값과 실제 입장 인원의 비율이 추적이 하는 일의 크기다.

### B. Detection + Tracking

추적으로 프레임을 잇고, 확정된 track 수를 occupancy로, 선 통과 사건을 입장/퇴장으로
센다.

### 묻는 것

**occupancy에서 추적이 도움이 되는가?** 당연하지 않다. 추적은 `min_hits` 때문에
새로 나타난 사람을 늦게 잡고 `max_age` 때문에 나간 사람을 늦게 놓는다. 순간값
정확도만 보면 A가 나을 수도 있다. **재 보기 전에는 모른다.**

**entry counting에서 추적이 얼마나 필요한가?** A는 셀 수 없다. 이것은 측정
결과가 아니라 정의상 그렇다. 크기는 위의 대리 지표로 잰다.

두 설정 파일의 detector / line / ROI 항목은 같다. 단위 테스트가 그것을
고정한다. 그래야 차이를 추적 유무 탓으로 돌릴 수 있다.

### Detector 선택

**torchvision `fasterrcnn_mobilenet_v3_large_fpn`, COCO pretrained.**

라이선스가 결정적이었다. Ultralytics YOLO 계열은 코드와 weight 모두 AGPL-3.0이고
상용에는 별도 Enterprise 라이선스를 요구한다(공식 저장소 명시). 밥플래너는
서비스로 나갈 제품이므로, AGPL 모델로 만든 숫자는 제품의 근거로 쓸 수 없다.
**나중에 버릴 결과는 처음부터 만들지 않는다.**

torchvision은 BSD-3-Clause이고 COCO pretrained weight도 같은 저장소에서 나온다.
AI-2에서 torch가 어차피 필요하다는 점도 맞는다.

**성능은 양보했다.** torchvision 보고 기준 COCO box mAP 32.8. 최신 YOLO 계열보다
낮다. detector 경계를 따로 둔 이유가 이것이고, Apache-2.0인 YOLOX나 RT-DETR로
갈아끼울 때 바꿀 파일은 하나다.

### Tracker 선택

**ByteTrack 방식 2단계 연결** (Zhang et al., ECCV 2022. 공식 구현 MIT).

외형 특징을 쓰지 않아 "사람을 식별하지 않는다"는 제약과 정확히 맞는다.
DeepSORT 계열은 re-id embedding을 쓰는데 그것은 이 프로젝트가 하지 않기로 한
바로 그것이다. 그리고 낮은 신뢰도 상자를 2단계로 구제하므로, 배식 줄에서
사람이 서로를 가려 신뢰도가 잠깐 떨어져도 track이 끊기지 않는다.

**원본과 다른 점:** Kalman filter 대신 등속 외삽, Hungarian 대신 IoU greedy.
둘 다 numpy/scipy 없이 계수 로직을 그대로 단위 테스트하기 위한 선택이다.
**여기 수치를 ByteTrack 논문 성능으로 읽으면 안 된다.** 이 단순화의 대가는
아직 재지 않았다(12절).

## 5. Metrics

### Occupancy

| | |
|---|---|
| 주 | **MAE** — 단위가 사람 수 그대로. "평균 1.8명 틀린다" |
| 보조 | RMSE |
| 보조 | bias = mean(예측 − 실제) |

MAE가 같아도 계속 적게 세는 것과 많이 세는 것은 운영에서 완전히 다르므로
bias를 같이 본다.

**MAPE는 주 지표로 쓰지 않는다.** 식당은 비는 시간이 있고, 실제 인원 0인
구간에서 MAPE는 정의되지 않거나 무한대로 튄다. 0을 빼면 가장 쉬운 구간만
남는다. 어느 쪽이든 숫자가 실제보다 좋아 보인다.

### Entry Count

영상 하나에 `|예측 − 실제|`. 여럿이면 평균. 사건 단위 annotation이 충분해지면
precision/recall/F1로 넘어갈 수 있으나, 그 전에 쓰면 정답이 없는데 있는 것처럼
된다.

### Tracking

ID switch / IDF1 / HOTA는 이번 단계 필수가 아니다. 추적은 목적이 아니라 entry
counting의 수단이고, 제대로 재려면 track 단위 annotation과 `TrackEval` 의존성이
필요하다. **계수가 주 과제라는 점을 유지한다.** 추적 품질은 entry count 오차로
간접 평가한다.

### Runtime

`processing_fps`. `metadata.json`에 장치·해상도·패키지 버전이 같이 남는다.
**다른 환경의 일반 성능으로 옮겨 말하지 않는다.**

## 6. Experimental Setup

### 하드웨어 (개발 환경)

| | |
|---|---|
| OS | Windows 11 |
| Python | 3.12.1 |
| GPU | NVIDIA GeForce RTX 4060 Ti (driver 581.29) |

**이 환경에서 추론을 실행한 적은 없다.** torch/torchvision을 아직 설치하지
않았기 때문이다(11절).

### 설정

`configs/baseline_detection.yaml`, `configs/baseline_tracking.yaml`,
`configs/demo.yaml`. 전부 0~1 정규화 좌표를 쓴다. 해상도가 달라도 같은 자리를
가리켜야 하기 때문이다.

주요 기본값 — **정답 데이터로 검증한 값이 아니라 출발점이다.**

| | 값 | 의미 |
|---|---|---|
| `confidence_threshold` | 0.4 | 실험 축 |
| `high_threshold` | 0.5 | 새 track은 이 이상에서만 |
| `low_threshold` | 0.1 | 그 사이는 2단계 구제 매칭에만 |
| `max_age` | 30 | 30fps 기준 약 1초 |
| `min_hits` | 3 | 1회성 오탐 차단 |
| `band` | 0.01 | 선 위 미세 떨림 무시 |
| `min_confirm_distance` | 0.03 | 얕은 진동 무시 |
| `cooldown_frames` | 15 | 재계수 금지 구간 |

### 재현

실행마다 `metadata.json`을 남긴다. 설정 전체, detector 이름, weight 이름, 장치,
패키지 버전, 실행 시각, seed. 지금 pipeline에 난수는 없지만 seed도 기록한다.

## 7. Results

### Pending benchmark

정답 데이터가 없어 아무것도 측정하지 않았다.

| | |
|---|---|
| Occupancy MAE | **NOT MEASURED** — 정답이 붙은 벤치마크 없음 |
| Occupancy RMSE | **NOT MEASURED** |
| Occupancy bias | **NOT MEASURED** |
| Entry count error | **NOT MEASURED** |
| Processing FPS | **NOT MEASURED** — 실제 추론을 실행하지 않음 |
| Baseline A vs B | **NOT MEASURED** — 코드는 준비됨 |
| Confidence sweep | **NOT MEASURED** — 코드는 준비됨 |

### 측정한 것

정량 성능은 아니지만, 계수 로직의 정확성은 단위 테스트로 검증되어 있다.

| | |
|---|---|
| 신규 테스트 | 208개 |
| 전체 테스트 | 472개 (기존 264 + 신규 208) |
| 결과 | 전부 통과 |
| 실행 조건 | torch, opencv 없이. 가짜 detector로 pipeline 전체를 관통 |

검증되는 것: 상자 0개 → occupancy 0, ROI 포함/제외, IN/OUT 통과, 선 위 정지,
선 접촉 후 복귀, 같은 track의 떨림에서 중복 미발생, cooldown 이후 IN→OUT→IN이
2회 방문, 정규화 좌표의 해상도 독립성, 요약의 최댓값/평균, CSV 스키마,
JSON NaN 거부, 신뢰 flag, 빈 프레임의 신뢰 정보 부재, 잘못된 설정 거부,
없는 영상 파일, MAE/RMSE/bias/입장 오차, 스키마의 개인정보 부재, 계수의 결정성.

**이것은 "코드가 의도대로 동작한다"이지 "실제 영상에서 맞다"가 아니다.** 둘은
다른 주장이다.

## 8. Domain Shift

### 계획

조건별로 나눠 본다.

```
daylight / low-light / sparse / crowded / occlusion / far-distance / oblique-camera
```

| Condition | Samples | Occupancy MAE | Entry Error |
|---|---:|---:|---:|
| (조건 annotation 없음) | 0 | N/A | N/A |

**표가 비어 있는 것이 맞다.** 조건 태그는 데이터셋이 제공했거나 사람이 직접
붙인 것이어야 한다. 조건을 모르는 표본에 "주간"이라고 적으면 표는 채워지지만
그 표는 거짓이다. 빈 태그를 넣으면 코드가 거부한다.

공개 데이터셋 대부분이 이 태그를 제공하지 않는다. MOT는 시퀀스 단위로 실내/실외,
낮/밤 정도의 설명이 있어 시퀀스 전체에 태그를 붙이는 것은 가능하다.

## 9. Failure Cases

### 관찰된 것: 없음

**실제로 영상을 돌린 적이 없으므로 관찰된 실패 사례가 없다.** 여기에 목록을
적어 두면 그것은 결과가 아니라 상상이다.

### 예상되는 것 (예상이며, 결과가 아님)

설계 단계에서 대비한 실패 유형들이다. 실제로 일어나는지, 어떤 순서로 심한지는
돌려 봐야 안다.

| 유형 | 예상 영향 | 대비한 것 |
|---|---|---|
| 밀집 시 상자 병합 | occupancy 과소 | `CROWDED` flag (기본 꺼짐) |
| 가림으로 track 끊김 | 입장 인원 과대 | BYTE 2단계 매칭, `max_age` |
| ID switch | 입장 인원 오차 | IoU + 등속 외삽 |
| 선 근처 떨림 | 입장 인원 과대 | band + 이력 + cooldown 3중 |
| 실제 재입장 | 과소 계수 가능 | cooldown 이후는 별도 방문으로 |
| 오탐(의자, 포스터) | occupancy 과대 | 클래스 필터, `min_hits` |
| ROI 경계 깜빡임 | occupancy 흔들림 | 경계 포함, 한 점 판정 |
| 원거리 소형 상자 | 누락 | `LOW_CONFIDENCE` flag |
| 영상 시작 시 내부 인원 | 입장 과대 | 첫 확정은 사건이 아님 |
| 방향 오설정 | IN/OUT 뒤집힘 | annotated.mp4에 방향 표시 |

**이 표를 결과로 옮기지 않는다.** 실제로 관찰한 것만 위 절로 올린다.

## 10. Demo / Integration

### Demo와 Benchmark를 섞지 않는다

| | Demo | Benchmark |
|---|---|---|
| 무엇 | 우리 영상에서 실제로 도는 모습 | 정답 데이터에서 정량 평가 |
| 결과 | "돌아간다" | MAE, 오차, FPS |
| 현재 | **미실행 — 영상 미제공** | **미측정** |

시연 영상에서 나온 숫자는 성능이 아니다. 정답이 없기 때문이다.

### 실행

    .venv\Scripts\python.exe -m pip install -e ".[cctv]"

    .venv\Scripts\python.exe ^
      research\cctv_counting\scripts\run_video.py ^
      --input <영상 경로> ^
      --config research\cctv_counting\configs\demo.yaml ^
      --output-dir outputs\cctv_demo

**영상 경로는 반드시 직접 준다.** 이 도구는 로컬에 있는 영상을 찾아서 알아서
쓰지 않는다. CCTV 영상에는 사람이 찍혀 있고, 어느 영상을 처리할지는 사람이
정해야 한다.

출력: `annotated.mp4`(상자, 임시 번호, 선, ROI, CURRENT/ENTERED/EXITED, 낮은
신뢰 표시), `observations.csv`, `summary.json`, `metadata.json`. 원본은 덮어쓰지
않는다.

### Spring 연동

| AI-1 출력 | Spring 필드 |
|---|---|
| `occupancy_count` | `peopleCount` — 순간값. entry count가 아니다 |
| 관측 시각 | `observedAt` |
| (고정) | `source` = `CCTV_AI` |
| `reliability_flag` | 계약에 자리 없음 |
| `entries_total` | 계약에 자리 없음 |

**지금 연결하지 않는다.** 백엔드는 `COMFORTABLE / NORMAL / CROWDED /
VERY_CROWDED` 등급을 필요로 하는데, `peopleCount`를 등급으로 바꾸는 임계값이
정해진 적이 없다. 40명이 붐비는지는 좌석 수에 달렸다. 100석에서 40명은 한산하고
50석에서는 붐빈다. **모델이 임계값을 정할 근거가 없다.**

연구 출력은 `people_count`까지다. 등급 변환은 좌석 수 또는 운영 정책 기반
threshold policy가 정해진 뒤 서비스 계층에서 한다. 모델 안에 임의 임계값을
넣으면 정해지지도 않은 정책이 모델 버전에 박힌다.

## 11. Limitations

**정량 성능이 하나도 없다.** 가장 큰 한계다. 지금 있는 것은 "끝까지 도는
pipeline"이지 "얼마나 맞는지 아는 pipeline"이 아니다.

**실제 급식소 데이터가 하나도 없다.** 카메라 각도도 조명도 밀집도도 모른다.
공개 데이터로 잰 숫자가 현장에서 유지될지는 domain shift 실험 전에는 알 수
없다.

**실제 추론을 한 번도 실행하지 않았다.** torch/torchvision을 설치하지 않았다.
CUDA 빌드가 크고(수 GB), FastAPI 서비스가 도는 가상환경에 넣는 것은 되돌리기
번거로운 변경이라 별도 판단으로 남겼다. 그래서 detector 어댑터의 실제 추론
경로는 코드로만 존재하고 실행으로 검증되지 않았다.

**추적 단순화의 대가를 재지 않았다.** Kalman 대신 등속 외삽, Hungarian 대신
greedy. 얼마나 손해인지 모른다.

**밀집이 가장 약할 것으로 보이나 확인되지 않았다.** 예상이지 관찰이 아니다.

**비상업 데이터로 잰 숫자는 상용의 근거가 아니다.** 평가에는 쓰되, 그 숫자를
제품 성능으로 주장하지 않는다.

**congestion level 임계값이 없다.** 10절.

**reliability flag와 실제 오차의 상관이 미측정이다.** RQ3. flag가 켜진 프레임이
실제로 더 많이 틀리는지 확인되지 않았다. 확인되지 않은 신호로 값을 거르면,
멀쩡한 관측을 버리면서 버렸다는 사실조차 모르게 된다.

## 12. Next Experiment

우선순위 순.

**1. 실제 영상 한 편을 끝까지 돌린다.**
`pip install -e ".[cctv]"` → 시연 영상 → `annotated.mp4` 확인. 여기서 처음으로
실제 실패 사례가 나온다. 9절의 "예상"이 "관찰"로 바뀌는 지점이고, FPS도 여기서
처음 나온다. **다른 모든 것보다 먼저다.**

**2. MOT17 고정 카메라 시퀀스 하나로 첫 벤치마크.**
Terms 확인 → 시퀀스 1개 → occupancy 정답은 GT 상자 개수에서, entry 정답은 GT
궤적이 선을 넘는 사건에서 유도. 이것으로 7절의 빈칸 대부분이 채워진다.

**3. Baseline A vs B.**
같은 시퀀스, 같은 detector 설정. occupancy에서 추적이 실제로 도움이 되는지,
entry에서 순진한 대리 지표가 얼마나 부풀려지는지.

**4. Confidence threshold 스윕 (0.2 / 0.4 / 0.6).**
MAE와 bias를 같이 본다. **bias의 부호가 뒤집히는 지점**이 있으면 그것이 가장
볼 만한 결과다.

**5. RQ3 확인.**
`reliability_flag`가 켜진 프레임의 오차와 꺼진 프레임의 오차를 비교한다. 차이가
없으면 이 신호는 쓸모없고, 다른 신호를 찾아야 한다. **없다는 결과도 결과다.**

**6. 추적 단순화의 대가 측정.**
greedy를 Hungarian으로 바꿔 같은 시퀀스를 다시 돌린다. 차이가 작으면 지금
구현을 유지하고, 크면 scipy 의존성을 추가한다.

**7. Domain condition 태그 부착.**
MOT 시퀀스 단위로 실내/실외, 낮/밤, 밀집도를 붙인다. 8절의 표가 채워진다.

---

## 부록. 구현 요약

| | |
|---|---|
| 위치 | `research/cctv_counting/` |
| 핵심 로직 의존성 | 표준 라이브러리 + pydantic |
| 추론 의존성 (선택) | `pip install -e ".[cctv]"` — pyyaml, opencv-python, numpy, torch, torchvision |
| 신규 테스트 | 208개 |
| 전체 테스트 | 472개 통과 |
| 개인정보 검사 | 자동. 매 테스트 실행마다 |

- 상세: `research/cctv_counting/README.md`
- 데이터 조사: `research/cctv_counting/datasets/README.md`
- 평가 절차: `research/cctv_counting/evaluation/README.md`
