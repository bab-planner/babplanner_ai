# W6 — Controlled Babplanner-Domain Pilot

*밥플래너 AI-6. W5(`reports/waste_w5_structured_outputs.md`)에 이어지는 문서.*

작성 시점: 2026-09-01
상태: **`BLOCKED_PENDING_CONTROLLED_DATA`**

> **이 문서에는 성능 숫자가 없다.**
> 통제 파일럿 원본 데이터가 아직 0건이다. 모델 평가를 실행하지 않았고,
> 실행할 수 있는 척도 하지 않았다. 이번 단계의 산출물은 **수집 절차·매니페스트
> 규격·수용 관문·동결된 평가 경로**이며, 데이터가 모이면 그대로 돌린다.

> **합성 이미지를 만들지 않았다.**
> 사진을 자동 생성해서 파일럿인 것처럼 진행하지 않는다. W1에서 세운 규칙이고
> 여기서도 지킨다 — 도구가 돈다는 것과 데이터가 있다는 것은 다른 말이다.

---

## 1. Motivation

W2~W5에서 다섯 단계를 거쳤고, 마지막에 실제로 뭔가가 움직였다.

```
W2   외부 벤치마크 성립          after-only ResNet18 + Ridge
W3   선택적 예측 가능            그러나 고잔반 안에서는 무효
W4   재가중·특징 결합 실패        값이 너무 비싸거나 무변화
W5   출력 구조 성공              DEVELOPMENT 치명 35.6% → 12.2%
```

그런데 이 숫자들이 전부 같은 곳에서 나왔다. **인도네시아 병원 급식 사진.**
밥플래너 현장 표본은 여전히 **0건**이다.

그래서 W6의 질문은 성능이 아니다.

> LeFood에서 만든 frozen S2가 **밥플래너 시연과 비슷한 촬영 조건**에서도
> 최소한의 전이 신호를 보이는가?

핵심은 모델 개선이 아니라 **domain gap 측정**이다.

---

## 2. 왜 LeFood에서 더 튜닝하지 않는가

W5가 잘 됐으니 더 밀어붙이고 싶은 지점이다. 하지 않는 이유가 셋 있다.

**첫째, 남은 여지가 어디 있는지 이미 갈렸다.** W5 §16에서 두 갈래로 좁혔다 —
게이트가 놓치는 표본(구조 쪽)과 게이트가 잡은 표본 안의 잔여 수축(표현 쪽).
어느 쪽을 손대든 **표현을 바꾸거나 입력을 늘리는 일**이고, 그건 LeFood에서
더 짜내는 것과 다른 종류의 작업이다.

**둘째, 같은 집합을 계속 보면 그만큼 낙관 쪽으로 기운다.** W2 TEST 116건은
W3에서 이미 소비 판정을 했고, DEVELOPMENT 398건도 이제 다섯 단계에 걸쳐
방법 선택에 쓰였다. 여섯 번째 선택을 얹으면 그 편향은 측정되지 않는다.

**셋째, 지금 모르는 것 중 가장 큰 것이 도메인이다.** 급식소 카메라, 한국
식판, 한국 메뉴에서 이 모델이 어떻게 되는지 **한 번도 재 본 적이 없다.**
소수점 세 자리를 더 깎는 것보다 이 물음이 훨씬 크다.

**따라서 이번 단계에서 모델 구조를 바꾸지 않는다.** S2 그대로, 임계값 그대로,
게이트 그대로. 파일럿을 보고 파라미터를 조정하는 코드 경로를 아예 두지
않았다.

---

## 3. 통제된 연출 파일럿이란 무엇인가

**이건 사람 대상 실험이 아니다.**

| | |
|---|---|
| 하지 않는 것 | 참가자 모집 |
| 하지 않는 것 | 실제 식사 행동 관찰 |
| 하지 않는 것 | 얼굴 촬영 |
| 하지 않는 것 | 학번·이름·연락처 수집 |
| **하는 것** | 준비한 음식을 **직접 덜어 내서** 잔반 상태를 만든다 |
| **하는 것** | 저울로 재고, 시연에 쓸 조건으로 찍는다 |

그래서 이 표본이 답할 수 있는 것과 없는 것이 갈린다.

| | |
|---|---|
| 답할 수 있다 | 카메라·식판·음식이 바뀔 때 모델이 얼마나 무너지는가 |
| **답할 수 없다** | 학생들이 실제로 얼마나 남기는가 |
| **답할 수 없다** | 메뉴별 실제 잔반률 |

두 번째 칸은 사람의 식사 행동을 봐야 답할 수 있고, 이 파일럿은 그걸 하지
않는다. **보고서에서 "실제로 먹고 남긴 사진"이라고 쓰지 않는다.** 정확한
표현은 `연출된 잔반 상태` / `staged leftover conditions`다.

### 코드가 이 구분을 강제한다

`SampleSource`에 값을 하나 추가했다.

```
PILOT              사람이 실제로 먹고 남긴 것을 관찰한 표본
CONTROLLED_STAGED  사람이 먹지 않았다. 덜어 내서 만든 상태
```

파일럿 매니페스트는 `source=controlled_staged_pilot`만 받는다. 다른 값이
들어오면 그 행을 거절한다. 이름을 하나로 두면 언젠가 "실제 이용자의 식습관
데이터"라고 읽히고, 그건 **하지 않은 관찰을 했다고 말하는 것**이다.

---

## 4. 수집 절차

전체 절차: **`docs/waste_controlled_pilot_runbook.md`**
매니페스트 템플릿: `docs/templates/waste_controlled_pilot_manifest_template.csv`

기존 `docs/waste_pilot_collection_runbook.md`와는 **다른 문서다.** 그쪽은
실제 급식소에서 사람이 먹고 남긴 것을 모으는 절차이고 참가자 번호가 나온다.
이 문서는 그 앞 단계다.

### 목표 규모

```
표본        30 ~ 60         (최소 30)
음식 종류    3 ~ 5
잔반 상태    구간마다 고르게, 특히 >= 0.8을 넉넉히
```

**처음부터 수백 장을 요구하지 않는다.** 반나절에 끝나는 규모로 잡고, 절차가
맞는지부터 확인한다.

### 정답

W1 정의 그대로다. 여기서 새로 만들지 않는다.

```
waste_rate = leftover_edible_mass_g / served_edible_mass_g
```

식기 무게가 섞이지 않게 tare를 쓰거나 빈 식판 무게를 따로 잰다. **이게 이
파일럿에서 가장 흔한 실수다** — 식판 412 g이 섞이면 잔반률이 통째로 틀린다.

### 촬영

식후 사진만 찍는다(현재 production 요청과 같다). 첫 파일럿에서는 촬영 변동을
크게 만들지 않는다 — 거리·각도·배경·조명을 고정한다. 다 흔들어 놓으면 성능이
나빴을 때 무엇 때문인지 알 수 없다. 강건성은 다음 문제다.

### 매니페스트

```
sample_id  source  food_group  capture_session_id  image_file
served_edible_mass_g  leftover_edible_mass_g  waste_rate(선택)
repeat_index  device_model  resolution  orientation  lighting  note
```

**개인 식별 열은 파일 전체를 거절한다.** 열 이름에 name / phone / student /
email / account / user_id / face / address / birth 가 들어가면 도구가
매니페스트를 읽지 않는다. `image_file`은 파일 이름만 받는다 — 절대 경로는 개인
폴더 구조를 그대로 노출하므로 거절한다.

`waste_rate`를 적으면 `leftover / served`와 대조해 0.005 넘게 다를 때 그 행을
거절한다. 옮겨 적기 실수를 잡는 장치다.

원본 사진은 `research/waste_estimation/data/private/w6_pilot/images/`에 두고,
이 경로는 `.gitignore`에 들어 있다. **사진은 저장소에 올라가지 않는다.**

---

## 5. 측정 반복성

**모델 MAE만 적고 이 값을 안 적으면 그건 숨김이다.**

저울과 절차가 만드는 흔들림보다 모델 오차가 작을 수는 없다. 그 바닥을 모르면
"MAE 0.05"가 좋은 값인지 측정 잡음인지 알 수 없다.

절차:

```
1. 표본 5개 이상을 고른다 (구간이 서로 다르면 좋다)
2. 같은 상태를 저울에서 내렸다가 다시 올려 3회 잰다
3. sample_id는 그대로, repeat_index만 0 / 1 / 2로 바꿔 적는다
```

`repeat_index > 0`인 행은 **모델 평가에 쓰지 않는다.** 측정 오차 계산에만
쓴다. 도구가 두 단위로 보고한다.

| | |
|---|---|
| 그램 | mean / std / max−min |
| 잔반률 환산 | 같은 값을 `served`로 나눈 것 |

두 단위가 다 필요한 이유가 있다. **5 g의 흔들림은 300 g 급식에서 0.017이지만
60 g 반찬에서는 0.083이다.** 그램만 보면 작은 반찬에서의 위험이 안 보인다.

평가 스크립트는 이 바닥값을 결과와 나란히 출력한다. 모델 MAE가 이 값과
비슷하면 그 차이는 측정 잡음과 구분되지 않는다.

---

## 6. 데이터셋 감사와 수용 관문

`research/waste_estimation/scripts/w6_pilot_audit.py`

기존 `audit_dataset.py`와 목적이 다르다. 그쪽은 "학습에 쓸 표본이 몇 개인가"를
묻고, 이쪽은 **"평가를 시작해도 되는가"**를 묻는다.

### 관문 (결과를 보기 전에 정했다)

| 항목 | 기준 |
|---|---|
| 유효 표본 | ≥ 30 |
| 음식 범주 | ≥ 3 |
| 고잔반 (≥ 0.8) | ≥ 6 |
| 잘못된 정답 (제외 후) | 0 |
| 중복 | 0 |

**통과하지 못하면 `PILOT_INSUFFICIENT`로 끝나고 모델 평가가 돌지 않는다.**
평가 스크립트에도 같은 관문이 걸려 있어서, 감사를 건너뛰고 바로 평가를
돌려도 멈춘다.

이 장치를 둔 이유는 단순하다. **표본 12개로 잰 MAE를 보고서에 적으면 그
숫자는 반드시 성능으로 인용된다.** 그럴 바에는 재지 않는 게 낫다.

### 거절 사유

조용히 버리지 않는다. W2에서 배운 것이다 — 라벨 관문이 154행을 걸렀는데 그
수를 세지 않았다면 데이터가 그만큼 있다고 착각했을 것이다.

```
WRONG_SOURCE  PARSE_ERROR  NON_POSITIVE_SERVED  NEGATIVE_LEFTOVER
LEFTOVER_EXCEEDS_SERVED  WASTE_RATE_MISMATCH  MISSING_IMAGE
DUPLICATE_SAMPLE_ID  DUPLICATE_IMAGE  ABSOLUTE_PATH
```

중복 사진 판정은 파일 내용의 SHA-256으로 한다. 다만 **반복 계량 행은 같은
사진을 의도적으로 공유**하므로 `repeat_index > 0`인 행은 중복에서 제외한다.

### 지금 상태

```
STATUS: BLOCKED_PENDING_CONTROLLED_DATA
매니페스트가 없습니다: research/waste_estimation/data/private/w6_pilot/manifest.csv
유효 표본 0
```

**0이라고 말한다.** 스키마가 있다는 것과 데이터가 있다는 것은 다른 말이다.

---

## 7. 동결된 모델

`research/waste_estimation/scripts/w6_evaluate.py`

W5가 고른 것 그대로다.

```
구조        S2_SOFT_HIGH_LOW_MIXTURE
표현        after-only · ResNet18 · IMAGENET1K_V1 · frozen
회귀        Ridge alpha=100 (저잔반 전문가 / 고잔반 전문가)
게이트      LogisticRegression alpha=100, class_weight=balanced, y >= 0.8
예측 정책    bounded — clip(ŷ, 0, 1)
동결 해시    cbcebd037f6677e127372576a8dbc106818e0aeb9b1244bc8afe360b9b04c6e0
```

### 가중치를 커밋하지 않았으므로 재학습한다

W5 산출물에 적합된 모델 가중치 자체를 넣지 않았다. 그래서 평가 시점에
**LeFood DEVELOPMENT 398건으로 W5의 동결 절차를 그대로 다시 밟아** S0와 S2를
만든다. 절차가 결정적이므로 같은 모델이 나온다.

### 학습에 들어가는 것과 들어가지 않는 것

| | |
|---|---|
| 학습에 쓴다 | LeFood DEVELOPMENT 398건 |
| **쓰지 않는다** | 통제 파일럿 라벨 |
| **쓰지 않는다** | 통제 파일럿 이미지 (임베딩만 뽑는다) |

전역 평균 baseline `B0`도 **LeFood DEVELOPMENT 평균**이다. 파일럿 라벨로
평균을 내면 그 순간 평가 누수다 — baseline이 정답 분포를 미리 본 것이 된다.

파일럿은 **평가 전용**이다. 모델 선택도, 보정도, 임계값 조정도, 게이트 조정도
하지 않는다. 그 코드 경로를 두지 않았다.

### 비교 대상

```
B0_GLOBAL_MEAN               LeFood DEVELOPMENT 평균
S0_STANDARD_RIDGE            W2/W4 baseline
S2_SOFT_HIGH_LOW_MIXTURE     W5 선택
```

### 재현 기록

평가가 돌면 `metadata.json`에 남는다.

```
w5PolicyHash              W5 freeze.json의 SHA-256
w5SelectedConfigHash      selected_config.yaml의 SHA-256
lefoodSplitManifestHash   LeFood 분할의 SHA-256
pilotManifestHash         파일럿 매니페스트의 SHA-256
```

---

## 8. 전체 결과

**없다.** 데이터가 0건이다.

데이터가 모이면 아래가 채워진다.

| 후보 | 전체 MAE | RMSE | Bias |
|---|---|---|---|
| B0_GLOBAL_MEAN | — | — | — |
| S0_STANDARD_RIDGE | — | — | — |
| S2_SOFT_HIGH_LOW_MIXTURE | — | — | — |

주 지표는 **bounded MAE**다(W5와 같은 정책). 측정 반복성(§5)을 반드시 같은
표에 병기한다.

### 사전 가설

| | 가설 |
|---|---|
| H-W6-A | frozen S2가 통제 파일럿에서도 전역 평균보다 낮은 MAE를 보인다 |
| H-W6-B | S2가 S0보다 고잔반 치명적 과소추정을 줄인다 |
| H-W6-C | LeFood보다 통제 파일럿 성능이 유의미하게 나빠질 수 있고, 그 차이가 제품 domain gap의 초기 추정치가 된다 |

**세 가설 모두 판정 보류.** 데이터가 없으므로 판정할 근거가 없다.

---

## 9. 고잔반 결과

**없다.** 데이터가 0건이다.

채워질 항목: 고잔반 MAE / Bias / 치명적 과소추정 건수·비율(pooled),
S2 게이트의 recall·precision·F1.

치명적 과소추정의 정의는 W4/W5 그대로다. 여기서 새로 만들지 않는다.

```
정답 >= 0.8  그리고  정답 − 예측 >= 0.30
```

### 게이트 진단이 갈래를 가른다

파일럿 결과가 나오면 §44에 해당하는 판단을 여기서 한다.

| 관찰 | 해석 |
|---|---|
| 게이트 recall은 유지되는데 고잔반 크기가 무너짐 | 표현/매핑 문제 |
| **게이트 recall 자체가 무너짐** | domain shift가 고잔반 인식부터 깨뜨린 것 |

두 경우의 다음 수가 완전히 다르다. 그래서 게이트 지표를 따로 낸다.

---

## 10. Domain Gap

**없다.** 데이터가 0건이다.

계산할 값은 서술용 차이 하나다.

```
ΔMAE_local = 통제 파일럿 MAE − LeFood ANALYSIS MAE
```

참고용 LeFood ANALYSIS 값 (bounded, 이미 결과를 본 집합에서의 사후 관찰):

| 후보 | 전체 MAE | 고잔반 MAE | 치명 |
|---|---:|---:|---:|
| S0_STANDARD_RIDGE | 0.1400 | 0.2916 | 16/34 = 47.1% |
| S2_SOFT_HIGH_LOW_MIXTURE | 0.1174 | 0.2015 | 8/34 = 23.5% |

> **이 차이를 통계적 동등 비교라고 말하지 않는다.** 두 집합은 같은 모집단이
> 아니다 — 음식도, 식기도, 카메라도, 표본 수도 다르고, 한쪽은 사람이 먹고
> 남긴 것이고 다른 쪽은 덜어 내서 만든 것이다. Δ는 **크기의 감을 잡기 위한
> 서술값**이지 검정 통계량이 아니다.

---

## 11. 실패 분석

**없다.** 데이터가 0건이다.

채워질 항목: 가장 큰 오차 표본 3건(sample_id·숫자·촬영 조건만, 사진 없음),
치명적 과소추정 표본의 게이트 점수, 음식 범주별 MAE·Bias.

범주별 표본이 적으면 **기술만 하고 비교하지 않는다.** 범주 하나에 6~8개가
들어올 텐데, 그 크기에서 범주 간 우열을 말하면 잡음을 읽는 것이다.

---

## 12. 데모 연결 준비

**판정 보류.** 데이터가 없으면 판정할 근거가 없다.

결과가 나왔을 때의 갈래는 미리 적어 둔다.

| | 관찰 | 다음 수 |
|---|---|---|
| CASE A | S2가 B0/S0보다 명확히 낫고 치명적 오류가 제한적 | 연구용 추정기의 데모 연결 검토 |
| CASE B | 전체는 괜찮은데 고잔반 치명적 오류가 큼 | 데모에서 **구간 피드백 또는 거절** 우선 |
| CASE C | LeFood 대비 크게 붕괴 | domain adaptation / 데이터 수집이 먼저 |
| CASE D | 관문 미달 | 평가 보류 ← **지금 여기** |

### 결과와 무관하게 지키는 최소 안전 조건

**고잔반 치명적 실패가 반복적으로 남으면, 숫자 잔반률을 사용자에게 확정값처럼
보여 주지 않는다.**

W5까지 와서도 정답 0.85를 0.03이라고 말하는 표본이 남아 있었다(W5 §13).
그런 오류가 화면에 "잔반률 3%"로 뜨면, 틀렸다는 사실보다 **확정적으로 보인다는
것**이 더 문제다. 구간 표시("많이 남음")나 판단 보류가 그보다 안전하다.

### 선택적 예측은 안전장치로 쓰지 않는다

W3·W4에서 novelty 기반 거절이 고잔반 내부 안전장치가 **아님**을 네 번 재서
확인했다. 이 파일럿에서 그것을 안전장치로 되살리지 않고, 보조 분석으로도
추가하지 않는다.

### production 경계

```
WASTE_MODEL_MODE=fixture      그대로
FastAPI wiring                없음
```

---

## 13. Threats to Validity

1. **연출된 상태다.** 사람이 먹고 남긴 모양과 다를 수 있다. 실제 잔반은
   한쪽으로 몰리거나 섞이거나 국물만 남는 등 더 불규칙하다. 이 파일럿의
   결과가 좋아도 자연 잔반에서 같다는 보장이 없다.
2. **사람의 식습관에 대해 아무것도 말하지 못한다.** 잔반 상태를 우리가
   정했으므로 분포 자체가 인위적이다.
3. **표본이 작다.** 30~60건이면 범주당 6~15건이다. 범주별 비교는 기술
   수준을 넘지 않는다.
4. **촬영 조건을 일부러 고정했다.** 조명·각도·거리를 묶었으므로 실제 사용자
   촬영의 변동성은 여기 없다. **성능이 실사용보다 낙관적으로 나올 것이다.**
5. **한 사람이 만든다.** 덜어 내는 방식에 개인 습관이 들어간다.
6. **LeFood와 같은 모집단이 아니다.** §10의 Δ는 서술값이지 검정이 아니다.
7. **측정 오차가 모델 오차와 섞인다.** §5의 반복 계량이 바닥을 알려 주지만
   완전히 분리해 주지는 않는다.
8. **frozen ImageNet 표현이다.** 한국 급식에 맞춰 학습된 적이 없다.
9. **식후 이미지만 쓴다.** production 요청과 맞췄지만, 식전 정보가 있으면
   달라질 수 있다는 W2의 물음은 여전히 열려 있다.
10. **파일럿이 관문을 넘어도 pristine external validation은 아니다.** 우리가
    만든 데이터로 우리 모델을 재는 것이다. 다만 **모델 개발에 쓰지 않았다는
    점**에서 LeFood ANALYSIS보다는 낫다.

---

## 14. Next Step

**지금 필요한 것은 코드가 아니라 사진 30장이다.**

```
1. docs/waste_controlled_pilot_runbook.md 를 따라 30~60건 수집
2. w6_pilot_audit.py 로 관문 확인 → PILOT_READY 가 나올 때까지
3. w6_evaluate.py 로 1회 평가
4. 이 문서의 §8~§12를 채우고 데모 연결 여부 판정
```

반나절이면 되는 규모로 잡았다. 음식 1종을 끝까지 해 보고 절차가 맞으면
나머지를 진행하는 것을 권한다.

**데이터가 모이기 전에 할 수 있는 다른 일도 있다.** W5 §16이 남긴 두 갈래 —
게이트 recall을 올리는 구조 쪽과 표현 쪽 — 중 구조 쪽은 LeFood에서 더 시험할
수 있다. 다만 §2의 이유로 그 순서를 뒤로 두었다. 도메인을 한 번도 재 보지
않은 채 여섯 번째 튜닝을 얹는 것보다, 사진 30장이 더 많은 것을 알려 준다.

---

## 재현

```
:: 감사 (데이터가 없으면 BLOCKED_PENDING_CONTROLLED_DATA)
.venv-waste\Scripts\python.exe research\waste_estimation\scripts\w6_pilot_audit.py ^
  --manifest research\waste_estimation\data\private\w6_pilot\manifest.csv ^
  --images research\waste_estimation\data\private\w6_pilot\images

:: 평가 (관문을 통과해야 돈다)
.venv-waste\Scripts\python.exe research\waste_estimation\scripts\w6_evaluate.py ^
  --manifest research\waste_estimation\data\private\w6_pilot\manifest.csv ^
  --images research\waste_estimation\data\private\w6_pilot\images
```

| 파일 | 내용 |
|---|---|
| `docs/waste_controlled_pilot_runbook.md` | 수집 절차 |
| `docs/templates/waste_controlled_pilot_manifest_template.csv` | 매니페스트 템플릿 |
| `research/waste_estimation/src/waste_estimation/pilot.py` | 매니페스트 검증·관문·반복성 |
| `research/waste_estimation/scripts/w6_pilot_audit.py` | 감사 CLI |
| `research/waste_estimation/scripts/w6_evaluate.py` | 동결 평가 CLI |

평가가 돌면 `research/waste_estimation/experiments/w6_pilot/`에 집계 수치만
남는다. 원본 사진과 개인 폴더 경로는 저장소에 들어가지 않는다.
