# A6 — Task-Specific Detection Policy and Trajectory Continuity

*밥플래너 AI-1. A5(`reports/cctv_a5_small_person.md`)에 이어지는 문서.
CCTV 연구 트랙의 마지막 단계다.*

작성 시점: 2026-08-28
상태: **fresh cross-camera test 1회 완료. 급식소 검증 아님. fine-tuning 없음**

> 이 문서의 데이터는 **EC Funded CAVIAR project / IST 2001 37540**
> (<http://homepages.inf.ed.ac.uk/rbf/CAVIAR/>)에서 왔다. 라이선스는
> Creative Commons BY-SA다. 라이선스가 요구하는 출처 표기다.

---

## 1. Motivation

A5는 하나의 문장으로 끝났다.

```
  E0이 놓친 통과 23건 중, 검출을 전부 회복시킨 detector로도
  살아난 통과는 10건 (43.5%)
```

검출이 **전부** 개선되었는데 통과 사건은 절반도 돌아오지 않았다. 남은 절반은
detector의 문제가 아니다. 사람을 더 잘 보게 되었는데도 "선을 넘었다"는 사건이
만들어지지 않았다는 뜻이고, 그것은 **같은 사람을 프레임 사이에서 같은 사람으로
잇지 못한다**는 뜻이다.

이 단계는 그 지점을 직접 잰다. 그리고 A5가 남긴 두 번째 관찰도 함께 다룬다 —
occupancy 기준으로 고른 설정이 통과 계수에서는 더 나빴다. **두 갈래가 서로
다른 정책을 원한다.**

## 2. A5 Findings — 무엇을 물려받았나

| A5의 결론 | A6에서의 취급 |
|---|---|
| 입력 해상도를 올려도 recall이 개선되지 않는다 (0.578 → 0.463) | 해상도 실험 종료. 다시 하지 않는다 |
| 큰 detector는 raw recall을 올리지만 occupancy MAE는 악화 (0.644 → 1.156) | detector 고정. 이번 축이 아니다 |
| 장소가 바뀌면 recall이 무너진다 (0.578 → 0.218) | 그대로 한계로 안고 간다 |
| 낮은 occupancy MAE가 오탐/누락 **상쇄**일 수 있다 | 선택 지표를 MAE 하나에서 다목적으로 바꾼다 (4절) |
| 통과 누락이 작은 사람 문제라는 H4는 기각 | 대신 궤적 연속성을 본다 |
| 검출을 고쳐도 통과 회복은 43% | **이 단계의 출발점** |

### 데이터 역할 재분류

**한 번 결과를 본 집합은 소비된 집합이다.** 역사적 사실(그때는 test였다)은
그대로 남기고, 지금 무엇에 쓸 수 있는지를 다시 적는다.

| 집합 | 원래 역할 | A6에서의 역할 | 쓸 수 있는 것 |
|---|---|---|---|
| 쇼핑센터 front 5편 | A4 DEV | **DEVELOPMENT** | 후보 비교, 선택 |
| 쇼핑센터 front 8편 | A4 TEST → A5 ANALYSIS | **ANALYSIS** | 원인 분석만 |
| INRIA 로비 17편 | **A5 FINAL (1회 채점)** | **ANALYSIS** | 원인 분석만 |
| 쇼핑센터 corridor 12편 | — | **FINAL TEST** | 한 번만 채점 |

**A5 FINAL에서 parameter를 고르지 않았다.** 그 집합은 이제 결과를 아는
집합이고, 아는 집합에서 고른 값은 일반화에 대해 아무것도 말하지 않는다.

## 3. Research Questions

**RQ-A6-1.** occupancy와 crossing이 서로 다른 confidence 정책을 쓰면, 같은
detector 추론을 공유하면서도 각 과제 성능을 개선할 수 있는가?
**RQ-A6-2.** crossing 누락은 detector miss 이후의 짧은 track 공백을
motion-aware tracking으로 이으면 회복되는가?
**RQ-A6-3.** 지금의 greedy tracker보다 궤적 연속성을 강화한 tracker가 통과
계수 오차를 줄이는가?
**RQ-A6-4.** 그 개선이 새로운 카메라에서도 유지되는가?

**사전에 등록한 가설** (`final_test_freeze.json`, FINAL 실행 전 동결):

| | |
|---|---|
| **H5** | 통과 누락은 전체 detector recall보다 **통과 순간 주변의 시간적 연속성**과 더 강하게 연결된다. |
| **H6** | entry 갈래에 낮은 신뢰도 상자를 허용하면 통과 recall이 오르고, motion association이 false crossing 증가를 억제한다. |

H5의 판정 조건은 두 개가 **모두** 필요하다.

```
  (a) 검출이 충분했던 사건만 봐도 (창 검출 coverage >= 0.5)
      놓친 사건의 창 연속성이 절대 0.10 이상 낮다
  (b) 연속성을 개선한 tracker가 통과 회복을 실제로 늘린다
```

(a)가 없으면 "검출 문제"를 이름만 바꿔 부른 것이 된다.

## 4. Metric Policy — A5의 병리를 고친다

### 낮은 occupancy MAE는 높은 detector 품질이 아니다

A5 FINAL에서 선택된 E0의 occupancy MAE(0.820)가 대조군 E3(1.252)보다 낮았다.
그런데 같은 표의 다른 줄은 이렇게 말한다.

```
              recall     오탐/frame    bias       MAE
  E0 (선택)    0.218       0.951       -0.415     0.820
  E3 (대조)    0.253       0.065       -1.252     1.252
```

E0은 **더 못 찾고 더 헛봤다.** 그런데 MAE가 낮다. 놓친 사람(음의 오차)과
계속 서 있는 오탐(양의 오차)이 프레임 안에서 서로를 지웠기 때문이다. 즉
**MAE 하나로 고르면 두 오류가 큰 설정이 두 오류가 작은 설정을 이길 수 있다.**

A6부터 occupancy 선택은 다목적이다.

```
  Primary     occupancy macro MAE
  Guardrail   검출 micro recall이 baseline 대비 절대 0.10 이상 낮으면 reject
  함께 본다   bias, 오탐/frame
```

### entry는 완전히 별개의 축이다

```
  Primary     평균 절대 통과 계수 오차   (|dIN| + |dOUT|) / 2 의 시퀀스 평균
  Guardrail   사건 precision >= 0.85, 그리고 baseline 대비 절대 0.10 이상 하락 금지
  Secondary   event recall / precision / F1
```

**precision guardrail이 왜 필요한가.** recall만 보면 없는 통과를 마구 만들어
내는 설정이 이긴다. 잘못 센 입장 인원은 그대로 수요 정답을 부풀리고, 그
정답으로 학습한 예측이 다시 발주를 부풀린다. 0.85는 그 관점의 절대 하한이고
**결과를 보기 전에** 정했다.

## 5. Error Taxonomy — 놓친 통과에 원인을 붙인다

"궤적이 끊긴다"는 말은 너무 넓다. 고칠 곳이 다른 다섯 가지를 구분한다.

| 원인 | 무슨 일이 일어났나 | 고칠 곳 |
|---|---|---|
| `DETECTION_GAP` | detector가 그 사람을 못 봤다 | detector / 임계값 |
| `ASSOCIATION_BREAK` | 상자는 있는데 track에 못 붙였다 | 연결 규칙 |
| `TRACK_EXPIRY` | 끊긴 시간이 max_age를 넘어 지워졌다 | 소멸 정책 |
| `ID_SWITCH` | 번호가 바뀌었다 | 운동 모델 / 할당 |
| `LINE_GEOMETRY` | 끊김 없이 이어졌는데도 사건이 없다 | 선 판정 규칙 |
| `OTHER` | 위 어디에도 해당하지 않는다 | — |

분류 규칙은 `attribution.py`에 있고 **위에서부터 처음 걸리는 것 하나**를
붙인다. 순서를 사후에 바꾸면 어떤 이야기든 만들 수 있으므로 결과를 보기 전에
고정했다.

### 자동 분류를 결론으로 쓰지 않았다

86건을 사람이 전부 볼 수는 없다. 그래서 자동 분류는 1차 분류로 두고, **대표
사건은 프레임 단위 기록을 직접 읽어** 확인했다(`a6_inspect.py`). 9절에 그
확인 결과가 있다.

## 6. Shared Detector, Task-Specific Policy

**추론은 프레임당 한 번이다. 갈라지는 것은 필터 정책뿐이다.**

```
  Video
    │
    └─ Shared Person Detector      (프레임당 정확히 한 번)
         │
         ├─ conf >= T_occ, ROI  ──▶  Occupancy Count      (추적 없음)
         │
         └─ conf >= T_low,  ROI ──▶  Motion Tracker ──▶ Line Crossing
                                       (내부에서 T_high로 새 track 생성)
```

두 과제가 싫어하는 오차가 다르다.

```
  occupancy   오탐 하나가 그 프레임 인원을 그대로 부풀린다.
              놓친 사람은 다음 프레임에 다시 셀 기회가 있다.
              -> 높은 임계값이 유리하다.

  entry       궤적이 한 번 끊기면 track이 쪼개지고, 쪼개진 track은
              자기가 선의 어느 쪽에 있었는지를 모른다. 사건 자체가 사라진다.
              되찾을 두 번째 기회가 없다.
              -> 낮은 신뢰도 상자라도 궤적을 잇는 데 쓰는 편이 유리하다.
```

`dualpath.py`가 이 구조를 코드로 못박는다. `DualPathCounter.detector_calls`가
프레임 수와 같아야 한다는 것이 테스트로 걸려 있고, 실험 자체도 캐시된 상자
하나를 다섯 정책이 나눠 쓴다. **detector를 두 번 돌리는 구현은 테스트를
통과하지 못한다.**

## 7. Tracker Candidates

### 후보는 하나만 추가했다

A5에서 배운 것이 있다. 축을 여러 개 동시에 움직이면 무엇이 무엇을 바꿨는지
알 수 없다. 그래서 **비교 후보는 하나**다.

| | 기존 `ByteTrackInspiredTracker` | 새 `MotionAwareTracker` |
|---|---|---|
| 소멸 | `max_age` 프레임 | **`max_age_seconds` 초** |
| 외삽 | 놓친 시간과 무관하게 1프레임분 | **놓친 시간에 비례** |
| 속도 | 직전 프레임 차이 | **지수이동평균** |
| 할당 | IoU 내림차순 greedy | **Hungarian 전역 할당** |
| 외형 특징 | 없음 | **없음** (그대로) |

### Kalman filter를 쓰지 않았다

§17의 두 방향(A: Kalman + 전역 할당 / B: 기존 운동 모델 개선 + 전역 할당)
중 **B를 골랐다.** 이유는 셋이다.

1. `.venv-cctv`에 scipy가 없다. 계수 로직 전체가 표준 라이브러리만으로
   돌아야 한다는 제약이 A2부터 있다. Kalman은 행렬 연산을, 전역 할당은
   `linear_sum_assignment`를 필요로 하는데, 후자는 100줄 남짓으로 직접
   구현할 수 있고 전자는 그렇지 않다.
2. **진단이 가리킨 결함이 측정 잡음이 아니었다.** 기존 코드의 결함은
   구체적이다 — `predicted_bbox()`가 놓친 시간과 무관하게 1프레임분만
   외삽한다. 10프레임을 놓쳐도 예측은 1프레임 앞에 머문다. Kalman이 아니라
   **시간 비례**가 그 결함의 답이다.
3. 결정적 단위 테스트가 쉽다.

**그래서 이것을 "full ByteTrack"이라고 부르지 않는다.** 이름은
`MotionAwareTracker`이고, 그 이름이 하는 일 그대로다.

### 전역 할당이 왜 필요한가

두 사람이 스쳐 지나갈 때 greedy가 지는 배치가 있다.

```
        track A ─ det 1   IoU 0.55        track A ─ det 2   IoU 0.50
        track B ─ det 1   IoU 0.52        track B ─ det 2   IoU 0.10

    greedy   A-1(0.55) 먼저 확정 -> B에게 det 2(0.10)만 남는다.
             IoU 임계값에 걸려 B가 끊긴다.
    전역     A-2(0.50) + B-1(0.52) -> 둘 다 산다.
```

통과 사건은 대개 사람이 카메라 앞을 가로지르며 서로를 가리는 구간에서
일어난다. 정확히 이 배치가 생기는 곳이다.

### re-identification은 하지 않는다

외형 embedding, 얼굴 특징, person re-id 네트워크를 넣지 않았다. 밥플래너의
목적은 익명 계수이고 필요한 것은 **짧은 시간의 운동 연속성**뿐이다. 이
제약은 문서가 아니라 테스트가 지킨다 — track 자료구조에 `embedding`,
`feature`, `appearance`, `face`, `reid` 같은 이름의 필드가 생기면 테스트가
깨진다.

## 8. Development Experiments

DEVELOPMENT = 쇼핑센터 front 5편. 정답 통과 사건 **9건**(IN 3 / OUT 6).

### occupancy 갈래 — 추적과 무관하다

| T_occ | macro MAE | bias | macro recall | micro recall | 오탐/frame |
|---|---:|---:|---:|---:|---:|
| 0.5 | 1.612 | +1.496 | 0.604 | 0.618 | 1.963 |
| **0.7** | **0.644** | **−0.082** | 0.567 | 0.578 | **0.593** |

주 지표에서 0.7이 단독 최소이고, recall guardrail(baseline 대비 −0.10)도
통과한다. **T_occ = 0.7.** A4/A5와 같은 값이다.

이 표는 A5 DEV의 E0 행(MAE 0.644, recall 0.578, 오탐 0.588)과 일치한다.
A6 코드가 A5 결과를 그대로 재현한다는 확인이다.

### entry 갈래 — 정책 행렬

**결과를 보기 전에 다섯 개로 고정했다.** Cartesian product를 돌리지 않는다.

| | T_occ | T_high | T_low | tracker | max_age |
|---|---|---|---|---|---|
| **P0** baseline | 0.7 | 0.7 | 0.7 | greedy | 30 frames |
| **P1** dual threshold | 0.7 | 0.5 | 0.3 | greedy | 30 frames |
| **P2** motion | 0.7 | 0.5 | 0.3 | **motion** | **1.2 s** |
| **P3** motion long age | 0.7 | 0.5 | 0.3 | motion | 2.0 s |
| **P4** motion low 0.1 | 0.7 | 0.5 | **0.1** | motion | 1.2 s |

| policy | count err | IN p/a | OUT p/a | precision | recall | F1 | frag/track | IDSW | gap rec | ms/frame |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 baseline | 0.30 | 1/3 | 5/6 | 1.000 | 0.583 | 0.737 | 13.44 | 65 | 0.793 | 0.085 |
| P1 dual threshold | 0.20 | 2/3 | 5/6 | 1.000 | 0.750 | 0.857 | 12.18 | 98 | 0.748 | 0.195 |
| **P2 motion** | **0.10** | **3/3** | 5/6 | **1.000** | **0.917** | **0.957** | 11.59 | 82 | 0.803 | 0.203 |
| P3 motion long age | 0.10 | 3/3 | 5/6 | 1.000 | 0.917 | 0.957 | 11.59 | 82 | 0.803 | 0.232 |
| P4 motion low 0.1 | 0.20 | 2/3 | 5/6 | 1.000 | 0.750 | 0.857 | 11.00 | 94 | 0.745 | 0.342 |

**세 가지를 읽을 수 있다.**

1. **임계값을 가르는 것만으로 절반이 개선된다** (P0 0.30 → P1 0.20). tracker는
   그대로다. RQ-A6-1의 DEV 답은 "그렇다"이다.
2. **연속성 tracker가 나머지를 가져간다** (P1 0.20 → P2 0.10).
3. **낮은 신뢰도를 더 열면 오히려 나빠진다** (P2 0.10 → P4 0.20). 약한 상자가
   많아지면 엉뚱한 상자에 track이 붙잡힌다. H6는 "낮출수록 좋다"가 아니다.

**DEV의 통과 사건은 9건뿐이다.** 이 표의 차이는 사건 한두 건이다. 그래서
선택은 여기서 하되, 판정은 FINAL에서 한다. 이 한계를 12절에 다시 적는다.

**IDSW를 정책 간에 직접 비교하면 안 된다.** P1/P2는 임계값이 낮아 훨씬 많은
사람을 추적한다(track 수 자체가 다르다). 분모가 다른 절대 개수다.


## 9. Error Attribution on ANALYSIS

ANALYSIS = 쇼핑센터 front 8편(정답 통과 41건) + INRIA 로비 17편(45건).
**합쳐 86건.** 여기서는 선택을 하지 않는다. 원인만 본다.

### baseline(P0)이 놓친 46건의 원인

| 원인 | 쇼핑센터 front | 로비 | 합계 |
|---|---:|---:|---:|
| `DETECTION_GAP` | 13 | 20 | **33** |
| `ID_SWITCH` | 10 | 1 | **11** |
| `ASSOCIATION_BREAK` | 0 | 0 | 0 |
| `TRACK_EXPIRY` | 0 | 0 | 0 |
| `LINE_GEOMETRY` | 0 | 0 | 0 |
| `OTHER` | 0 | 2 | 2 |
| **놓친 사건 합** | 23 | 23 | **46** |

세 가지를 읽을 수 있다.

1. **`LINE_GEOMETRY`가 0이다.** 끊김 없이 하나의 track이 선 양쪽을 덮었는데
   사건이 나오지 않은 경우가 한 건도 없다. band / hysteresis / cooldown 같은
   선 판정 규칙은 **지금 병목이 아니다.** A3에서 공들여 만든 삼중 방지가
   과도하지 않았다는 뜻이기도 하다.
2. **`ASSOCIATION_BREAK`도 0이다.** 이유가 구조적이다 — 임계값 0.7과
   `min_hits=1`에서는 살아남은 상자가 곧바로 track이 된다. 그래서 baseline의
   track coverage는 검출 coverage와 **소수점까지 같다.** baseline의 문제는
   "덮였는가"가 아니라 **"같은 번호인가"**다.
3. **`TRACK_EXPIRY`도 0이다.** `max_age` 30프레임(1.2초)은 짧지 않았다. 끊김은
   소멸 때문이 아니라 **연결 실패** 때문이다.

즉 tracker 쪽 원인은 사실상 `ID_SWITCH` 하나로 모인다. 쇼핑센터에서는 놓친
23건 중 **10건(43%)**이 여기다. 로비는 사람이 훨씬 작게 보여
`DETECTION_GAP`이 압도한다(20/23).

### H5 — 검출을 통제하고도 연속성이 갈리는가

전체 86건을 그냥 비교하면 이렇게 나온다.

| | n | 검출 coverage | track coverage | dominant track share |
|---|---:|---:|---:|---:|
| 맞힌 사건 | 40 | 0.621 | 0.621 | 0.566 |
| 놓친 사건 | 46 | 0.349 | 0.349 | 0.249 |

**이 표만으로는 H5를 지지할 수 없다.** 검출이 없으면 추적도 없으므로 두 값이
같이 떨어지는 것은 당연하다. 그래서 사전에 등록한 조건대로 **검출이 충분했던
사건만**(창 검출 coverage >= 0.5, 42건) 따로 본다.

| | n | 검출 coverage | dominant track share | 단일 track 비율 |
|---|---:|---:|---:|---:|
| 맞힌 사건 | 29 | 0.764 | **0.702** | **0.586** |
| 놓친 사건 | 13 | 0.677 | **0.430** | **0.154** |
| **차이** | | **0.087** | **0.272** | 0.432 |

**검출 coverage 차이는 0.087인데 창 연속성 차이는 0.272다.** 세 배가 넘는다.
쇼핑센터만 보면 더 뚜렷하다 — 놓친 10건 중 **단일 track으로 덮인 것은 0건**이고
맞힌 16건 중에는 43.8%가 그렇다.

**H5 판정 조건 (a): 충족.** 사전 기준 0.10을 크게 넘는다.
(b)는 12절에서 확인한다.

### 대표 사건을 직접 읽었다

자동 분류가 실제로 그 사건을 설명하는지, 프레임 단위 기록으로 확인했다.

**`ID_SWITCH` — `ThreePastShop1front` GT track 5, frame 1374 OUT**

```
  frame   x_norm  side    detConf  trackId
   1372   0.518    -       0.86      40      <- 아직 왼쪽 판정 유지 (선에서 0.018)
   1373   0.526    -        -         -
   1374   0.536  right      -         -      <- 정답 통과 시점
   ...                                        5프레임 동안 검출 없음
   1378   0.562  right    0.99      44      <- 새 번호. 여기서 처음 "오른쪽" 확정
```

track 40은 왼쪽을 확정한 채 살아 있었다(`max_age` 30프레임 안이다). 그런데
5프레임 공백 뒤 다시 나타난 사람이 **번호 44를 받았다.** 새 track의
`confirmed_side`는 None에서 시작하므로, 첫 확정("오른쪽")은 사건이 아니다.
**통과가 조용히 사라진다.** 소멸 때문이 아니라 연결 실패 때문이라는 자동
분류가 정확하다.

**`DETECTION_GAP` — `ShopAssistant1front` GT track 1, frame 232 IN**

```
  frame 207 ~ 231   x_norm 0.560 -> 0.471   검출 없음 (25프레임)
  frame 232         x_norm 0.469  left      첫 검출. track 9 생성
```

이 사람은 오른쪽에서 왼쪽으로 선을 넘는 동안 **한 번도 검출되지 않았다.**
처음 잡혔을 때는 이미 왼쪽이다. 어떤 tracker도 이 사건을 만들 수 없다.
자동 분류가 정확하다. 그리고 이것은 tracker가 고칠 수 있는 종류가 아니다 —
33건의 `DETECTION_GAP`이 tracking 개선의 상한을 정한다.

**회복 사례 — `LeftBox` GT track 0, frame 153 OUT**

```
  frame   x_norm   P0 trackId   P2 trackId   P2 detConf
   128    0.445        5            2          0.94
   145    0.500        -            2          0.41     <- 낮은 신뢰도 상자
   146    0.505        -            2          0.44
   148    0.510        -            2          0.50
   149    0.513        6            2          0.85
   152    0.529        7            2          0.86
   154    0.542        7            2          0.82     <- 오른쪽 확정
```

baseline은 25프레임 동안 이 한 사람에게 **번호 5, 6, 7 세 개**를 붙였다.
P2는 임계값 0.3 덕분에 0.41~0.50짜리 상자를 받고, 전역 할당과 시간 비례
외삽으로 **번호 2 하나**를 끝까지 유지했다. 두 장치가 같이 작동한 예다 —
낮은 신뢰도 상자가 재료를 공급하고, 운동 연속성이 번호를 지킨다.

이 시퀀스에서 baseline은 통과 3건 중 1건만 맞혔고 P2는 3건을 다 맞혔다.
`dominant track share`는 0.20 / 0.27 / 0.25에서 0.67 / 0.71 / 0.94로 올랐다.

## 10. Frozen Selection

**FINAL을 돌리기 전에 동결했다.**
`research/cctv_counting/experiments/a6_continuity/final_test_freeze.json`
sha256 `09116207068947f6…`

동결한 것:

```
  detector                E0_mobilenet_R0 (fasterrcnn_mobilenet_v3_large_fpn, R0)
  occupancy threshold     0.7
  tracker high threshold  0.5
  tracker low threshold   0.3
  tracker                 MotionAwareTracker
  max_age                 1.2 초
  min_hits                1
  line                    x = 0.5 세로선, band 0.01, min_confirm 0.03, cooldown 0.5초
  ROI                     (0.0, 0.20) ~ (1.0, 0.95)
  event tolerance         1.5 초
  crossing window         ±1.0 초
  FINAL 시퀀스            corridor 12편 (목록 고정)
  선택 규칙               4절 그대로
  판정 대상               P0(baseline) vs P2(selected)
```

**규칙의 형태는 DEV를 돌리기 전에 정했다** (주 지표가 무엇인지, guardrail이
어떤 종류인지, tie-break 순서). **수치 margin은 DEV 통계를 보고 정했고 FINAL
전에 동결했다.** 이 구분을 흐리지 않기 위해 동결 파일에도 그대로 적어 두었다.

### 왜 P2인가

P2와 P3가 DEV 계수 오차 0.10으로 공동 최소였고, F1(0.957)도 gap
recovery(0.803)도 같았다. tie-break 3(더 단순한 쪽)에 따라 `max_age`가 짧은
**P2**를 골랐다. 소멸 시간을 1.2초에서 2.0초로 늘려도 DEV에서 아무것도
바뀌지 않았다는 것 자체가 결과다 — **이 데이터에서 병목은 소멸 정책이 아니다.**
9절의 `TRACK_EXPIRY` 0건과 같은 이야기다.

P4(low 0.1)는 DEV 계수 오차가 0.20으로 P2보다 나빴다. **"낮출수록 좋다"가
아니라는 것**을 여기서 확인하고 선택에서 제외했다.

### FINAL에서 하지 않은 것

P1/P3/P4도 FINAL에서 함께 돌려 기록했지만 **선택을 바꾸는 데 쓰지 않았다.**
A5에서 대조군 E3를 함께 돌리되 선택에 쓰지 않은 것과 같은 취급이다.

## 11. Fresh Final Evaluation

**한 번만 돌렸다.** CAVIAR 쇼핑센터 **corridor** 시점 12편. front 13편(DEV +
ANALYSIS)과 **다른 카메라**이고, 계열의 **두 번째 촬영본**이라 사건도
처음이다. 공식 문서가 `1cor`와 `1front`를 "time synchronised"라고 명시하므로
`1cor`를 쓰면 이미 본 사건을 다시 채점하게 된다 — 그래서 `2cor`만 썼다.

```
  시퀀스        12편          채점 프레임    15,693 (검출 ∩ 정답)
  정답 상자     63,131        정답 통과      52건 (IN 21 / OUT 31)
  해상도        384x288       fps            25
```

**이것이 이 프로젝트의 첫 cross-camera 평가다.**

### occupancy

A6는 occupancy 정책을 바꾸지 않았다(T_occ = 0.7, 추적 없음). 따라서
baseline과 selected가 같은 값이다. 비교 대상은 동결에서 함께 기록한 0.5다.

| T_occ | macro MAE | bias | macro recall | micro recall | 오탐/frame |
|---|---:|---:|---:|---:|---:|
| 0.5 (비선택) | **1.268** | −1.117 | **0.876** | **0.878** | 0.243 |
| **0.7 (선택)** | 1.354 | −1.307 | 0.850 | 0.848 | **0.135** |

**동결된 선택이 FINAL에서 최적이 아니었다.** 0.5가 MAE 0.086만큼 낮다.
규칙대로 바꾸지 않는다. 이 크기(사람 0.086명)는 DEV에서 0.7이 0.5를 이긴
폭(0.968명)에 비하면 훨씬 작고, 방향이 뒤집힌 이유도 읽을 수 있다 —
corridor 시점은 사람이 크게 보여 검출 recall이 0.848까지 올라오고
오탐/frame이 0.135로 낮다. 오탐이 적으면 임계값을 낮춰서 잃는 것이 적다.

**그리고 A5의 경고가 여기서도 유효하다.** bias가 −1.307이다. 프레임당 평균
1.3명을 **덜 센다.** MAE 1.354의 거의 전부가 누락이고, 오차 상쇄가 아니다.
A5 로비(bias −0.415, 오탐 0.951)와 성격이 다른 실패다.

검출 recall 자체는 A5 FINAL(0.218)보다 훨씬 높다(0.848). 사람이 크게 보이기
때문이다. **그런데 occupancy MAE는 더 나쁘다**(0.820 → 1.354). 이 장면은
프레임당 사람이 4명꼴이라 같은 recall이라도 절대 오차가 커진다. **recall과
MAE는 장면 밀도를 통해 연결되고, 밀도가 다르면 둘의 관계도 달라진다.**

### crossing — 판정 대상

| policy | 계수 오차 | IN 예측/정답 | OUT 예측/정답 | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **P0 baseline** | **0.83** | 15/21 | 17/31 | 0.967 | 0.608 | 0.746 |
| **P2 selected** | **0.83** | 15/21 | 17/31 | **1.000** | **0.631** | **0.774** |
| *P1 (대조)* | *0.88* | *14/21* | *17/31* | *1.000* | *0.608* | *0.756* |
| *P3 (대조)* | *0.75* | *16/21* | *18/31* | *0.969* | *0.647* | *0.776* |
| *P4 (대조)* | *0.83* | *15/21* | *19/31* | *0.947* | *0.631* | *0.758* |

방향별 계수 오차(시퀀스 평균):

| | IN | OUT |
|---|---:|---:|
| P0 baseline | 0.500 | 1.167 |
| P2 selected | 0.500 | 1.167 |

### 판정: **채택하지 않는다**

동결 파일의 채택 조건은 세 개였고, 첫 번째에서 걸린다.

```
  [X]  FINAL 통과 계수 오차가 baseline보다 낮다      0.83 vs 0.83 — 낮지 않다
  [O]  precision guardrail 통과                      1.000 >= 0.85
  [O]  runtime이 서비스 시연을 막지 않음             +0.027 ms/frame
```

**주 지표가 개선되지 않았으므로 기존 tracker를 유지한다.** 이것이 동결
당시 정해 둔 결과다(`finalJudgement.otherwise`).

**P3가 0.75로 더 낮다는 것을 알고도 바꾸지 않는다.** P3는 DEV에서 P2와
완전히 동률이었고 tie-break로 탈락한 후보다. FINAL을 보고 순위를 바꾸면 그
0.75는 아무것도 재지 않은 숫자가 된다.

### 개선되지 않은 것은 계수 오차뿐이다

같은 표의 다른 칸은 전부 나아졌거나 같다.

```
  precision   0.967 -> 1.000      거짓 통과가 사라졌다
  recall      0.608 -> 0.631
  F1          0.746 -> 0.774
  frag/track  12.16 -> 11.29
  나빠진 지표  없음
```

**왜 계수 오차만 그대로인가.** 계수 오차는 |예측 − 정답|이라 **어떤 사건을
맞혔는지 보지 않는다.** P2는 baseline이 놓친 사건 2건을 살리고 1건을 잃었으며,
거짓 통과 하나를 참 통과로 바꿨다. 예측 **총수**는 IN 15, OUT 17로 같다.
그래서 사건의 질은 올라갔는데 개수 오차는 움직이지 않는다.

이 지표를 주 지표로 고른 것은 4절의 요구였고, 고른 이상 그대로 판정한다.
다만 **개수 오차가 사건 품질에 둔감하다는 것**은 이 단계에서 실제로 확인된
성질이고, 다음 단계의 지표 선택에 넘긴다.

## 12. Crossing Recovery Analysis

### 사건 단위 변화

| | ANALYSIS front (41) | ANALYSIS 로비 (45) | **FINAL corridor (52)** |
|---|---:|---:|---:|
| 둘 다 맞힘 | 15 | 16 | 30 |
| **baseline 실패 → selected 회복** | **14** | **13** | **2** |
| baseline 성공 → selected 실패 | 3 | 6 | 1 |
| 둘 다 실패 | 9 | 10 | 19 |
| 순증 | +11 | +7 | **+1** |

ANALYSIS에서 크게 나타난 이득이 FINAL에서 거의 사라졌다. **왜인지가 이 절의
핵심이다.**

### baseline 실패 원인별 회복률

| baseline 실패 원인 | ANALYSIS front | ANALYSIS 로비 | FINAL corridor |
|---|---:|---:|---:|
| `ID_SWITCH` | **8/10** | 1/1 | **2/2** |
| `DETECTION_GAP` | 6/13 | 11/20 | 0/3 |
| `OTHER` | — | 1/2 | **0/16** |

**개입은 겨냥한 것을 정확히 맞혔다.** `ID_SWITCH`로 분류된 실패는 세 집합
모두에서 **13건 중 11건이 회복**되었다(85%). A5에서 detector를 바꿔 검출을
전부 회복시켰을 때의 통과 회복률이 43%였던 것과 대조된다.

**그런데 FINAL에서는 `ID_SWITCH`가 21건 중 2건뿐이다.** 지배적인 것은
`OTHER` 16건이고, 개입은 그중 하나도 회복시키지 못했다.

### FINAL의 `OTHER` 16건은 무엇인가

기록을 직접 읽었다. 두 종류로 갈린다.

**(a) 계수 영역(ROI) 밖에서 일어난 통과 — 12건.**

```
  ShopAssistant2cor  gt 14  f 2377  OUT   windowFrames = 0
```

`windowFrames = 0`은 그 사람이 통과 시점 ±1초 동안 **ROI 안에 한 프레임도
없었다**는 뜻이다. ROI는 A4에서 front 시점의 발 위치 분포로 정한
(0.0, 0.20)~(1.0, 0.95)이고, corridor 시점에는 그 밖으로 나가는 궤적이 있다.

**정답 통과 사건은 ROI를 적용하지 않고 만들어진다**(A4부터 그랬다). 그래서
이 12건은 **어떤 tracker로도 셀 수 없는 사건**이다. 52건의 23%다.

이 12건을 빼고 보면 이렇게 된다.

| | 전체 52건 | ROI 안 40건 |
|---|---:|---:|
| P0 baseline 사건 recall | 0.596 | **0.775** |
| P2 selected 사건 recall | 0.615 | **0.800** |

**이 표는 사후 진단이지 판정이 아니다.** 동결된 판정은 위의 52건 표 그대로다.
다만 FINAL 성적의 상당 부분이 tracker가 아니라 **카메라가 바뀌었는데 ROI를
그대로 쓴 것**에서 온다는 사실은 기록해야 한다. ROI를 카메라마다 다시
그리는 것은 정당하지만, 그것은 **FINAL 결과를 보기 전에** 했어야 하는 일이다.

**(b) 창 안에서 선 양쪽이 확정되지 않은 통과 — 4건.**

```
  OneShopOneWait2cor  gt 7  f 474  OUT
      det coverage 1.00   track coverage 1.00   단일 track   spansBothSides = False
```

이 사람은 창 내내 **하나의 번호로 끊김 없이** 추적되었다. 그런데 ±1초 안에서
정답 발 위치가 선 양쪽으로 `min_confirm_distance`(0.03)만큼 벌어지지 않았다.
corridor 시점은 복도를 **따라** 보는 각도라 사람이 카메라 쪽으로 걸어오고,
가로 방향 이동이 느리다. **±1초 창이 front 시점 기준으로 정해졌고, 이
카메라에서는 좁다.**

창 크기도 동결 대상이므로 바꾸지 않는다. 이것 역시 카메라가 바뀔 때 함께
검토했어야 하는 값이다.

### H5 판정: **지지된다**

두 조건이 모두 충족된다.

```
  (a) 검출이 충분했던 사건만 봐도 (n=42)
      창 연속성 차이 0.272  vs  검출 coverage 차이 0.087        -> 충족
      (사전 기준 0.10)

  (b) 연속성을 개선한 tracker가 통과 회복을 늘렸다
      ID_SWITCH로 분류된 실패 13건 중 11건 회복                 -> 충족
      fragmentation은 네 집합 모두에서 감소
```

정확한 진술은 이렇다. **통과 누락은 detector recall만으로 설명되지 않는다.
검출이 충분한 사건 안에서도 맞힘과 놓침을 가르는 것은 창 안의 궤적 연속성이고,
그 연속성을 고치면 그 원인의 사건은 대부분 돌아온다.** 다만 **그 원인이
전체에서 차지하는 비중은 장면에 따라 크게 다르다** — front 43%, corridor 10%.

### H6 판정: **부분적으로만 지지된다**

```
  낮은 신뢰도 상자 허용 -> 통과 recall 증가
      DEV        0.583 -> 0.750    O
      ANALYSIS   0.438 -> 0.608    O
      FINAL      0.608 -> 0.608    X (변화 없음)

  precision이 무너지지 않는다
      P1(low 0.3)  DEV 1.000 / FINAL 1.000                O
      P4(low 0.1)  FINAL 0.947, 로비 0.770                X
```

**낮출수록 좋다는 것이 아니다.** low 0.3까지는 precision을 지키면서 recall이
오르지만, 0.1까지 내리면 로비에서 precision이 0.770까지 떨어지고 DEV 계수
오차도 나빠진다(0.10 → 0.20). 약한 상자가 많아지면 track이 엉뚱한 상자에
붙잡힌다.

그리고 FINAL에서는 낮은 상자를 열어도 recall이 오르지 않았다. corridor 시점은
사람이 크게 보여 애초에 0.7 이상으로 잡히기 때문이다 — 검출 recall이 0.848이다.
**낮은 신뢰도 구제는 사람이 작게 보이는 장면에서만 이득이 있다.**

### 궤적 연속성 지표 — tracker만 비교한다

P0과 P2를 직접 비교하면 임계값 차이가 섞인다(P2는 낮은 상자까지 받으므로
track 수 자체가 다르다). **임계값이 같은 P1과 P2를 비교해야 tracker의 효과만
남는다.**

| 집합 | 지표 | P1 (greedy) | P2 (motion) |
|---|---|---:|---:|
| DEV | fragments / GT track | 12.18 | **11.59** |
| | ID switch | 98 | **82** |
| | gap recovery | 0.748 | **0.803** |
| ANALYSIS front | fragments / GT track | 23.85 | **21.04** |
| | ID switch | 595 | **425** |
| | gap recovery | 0.681 | **0.733** |
| ANALYSIS 로비 | fragments / GT track | 21.76 | **20.46** |
| | ID switch | 264 | **192** |
| | gap recovery | 0.854 | **0.873** |
| **FINAL corridor** | fragments / GT track | 12.54 | **11.29** |
| | ID switch | 636 | **529** |
| | gap recovery | 0.708 | 0.711 |

**네 집합 모두에서 같은 방향이다.** fragmentation 감소, ID switch 감소
(FINAL에서 −17%), gap recovery 동률 이상. **tracker 자체는 의도한 대로
작동한다.** FINAL에서 통과 계수가 개선되지 않은 것은 tracker가 작동하지
않아서가 아니라, **그 장면의 실패가 주로 다른 원인이었기 때문이다.**

fragments/track이 11~24로 큰 것은 이 지표가 ROI 안에 잠깐씩 들어왔다 나가는
정답 track까지 모두 포함하기 때문이다. **절대값이 아니라 같은 집합 안에서의
차이로 읽어야 한다.**

## 13. Runtime

### 추가 tracker 비용

detector 추론은 두 갈래가 공유하므로 비교 대상이 아니다. A6가 바꾼 것은
tracker뿐이고, 그래서 잰 것도 `update()`뿐이다. FINAL 18,549 프레임, 3회 중
최소값, CPU 단일 스레드.

| | tracker | ms/frame | 상자/frame |
|---|---|---:|---:|
| P0 baseline | greedy | **0.037** | 2.07 |
| P1 | greedy | 0.050 | 2.50 |
| P2 selected | motion-aware | **0.064** | 2.50 |

**추가 비용은 프레임당 0.027 ms다.** 25 fps 예산(프레임당 40 ms)의 **0.07%**다.
Hungarian 전역 할당이 greedy보다 무겁지만, 프레임당 상자가 2~3개인 규모에서는
차이가 측정 잡음에 가깝다.

### detector 처리량 — A5와 나란히 놓지 않는다

같은 세션에서 잰 detector 순수 추론 속도는 이렇다(모델 적재와 warmup 제외).

```
  corridor  EnterExitCrossingPaths2cor   15.2 fps   65.7 ms/frame
  로비      Browse1                      11.9 fps   84.0 ms/frame
```

**로비는 A5가 45.1 fps로 잰 바로 그 집합이다.** 같은 모델, 같은 GPU인데 지금
11.9 fps가 나온다. 즉 이 차이는 데이터가 아니라 **측정 시점의 장비 상태**다.
실제로 측정 중 GPU에는 다른 프로세스들이 1.7 GB를 점유하고 있었다.

따라서 **A6의 detector 처리량 수치를 A5와 나란히 놓지 않는다.** 서비스 판단에
쓸 수 있는 것은 A5의 비경합 측정(45.1 fps > source 25 fps)이고, A6가 더한
것은 그 위의 0.027 ms/frame이다.

```
  source                     25 fps  (프레임당 40 ms)
  detector (A5, 비경합)      45.1 fps
  A6 tracker 추가 비용       +0.027 ms/frame   = 예산의 0.07%
```

**결론: A6의 변경이 실시간 시연을 막지 않는다.** 병목은 여전히 detector이고,
그것은 A6가 건드리지 않은 축이다.

## 14. Spring Integration — Demo Integration에 넘기는 것

A6가 CCTV 연구 트랙의 마지막 단계이므로, 다음 단계(Demo Integration)에서
필요한 출력을 여기서 정리한다. **이번 단계에서 실제 HTTP 전송은 하지 않았고,
백엔드 계약을 임의로 늘리지도 않았다.**

### 이미 있는 계약

백엔드에 `POST /internal/v1/congestion-observations`가 있다.

| AI-1 출력 | Spring 필드 | 비고 |
|---|---|---|
| `occupancy_count` | `peopleCount` | 순간값. **entry count가 아니다** |
| 관측 시각 | `observedAt` | 프레임 시각을 절대 시각으로 환산 |
| (고정) | `source` = `CCTV_AI` | 값의 진실은 백엔드가 가진다 |

occupancy 갈래의 출력이 그대로 이 계약에 맞는다. A6에서 occupancy 정책은
바뀌지 않았다(T_occ = 0.7, 추적 없음).

### entry count를 담을 자리가 없다 — Gap

`entries_total`은 **canonical storage contract가 없다.** 지금 백엔드에는
occupancy(`peopleCount`) 중심의 혼잡도 관측만 있고, "이번 끼니에 몇 명이
들어왔는가"를 누적해 보관하는 도메인이 없다.

이 값을 수요 예측의 **Demand Ground Truth**로 쓰려면 별도의 식수/이벤트
집계 도메인이 필요하다.

```
  필요한 것    끼니 단위 집계 (meal attendance)
               또는 이벤트 원본 저장 후 집계
  없는 것      그 도메인 자체, 그리고 그것과 발주/수요 모델의 연결
```

**이번 단계에서 계약을 늘리지 않았다.** 검증 없이 필드를 늘리면 보내는 쪽은
쓰이고 있다고 믿고 받는 쪽은 무시한다. 이 Gap을 문서에 남기는 것이 지금
할 수 있는 일이다.

### 정확도 관점에서의 전제

FINAL 통과 계수의 recall이 아직 0.7 수준이라는 것도 함께 넘긴다. 이 값을
그대로 식수로 쓰면 실제보다 적게 센다. **보정 없이 발주에 연결하면 안 된다.**

## 15. Threats to Validity

**1. 사건 대조가 시각과 방향만 본다.** `match_events`는 예측 통과와 정답
통과를 ±1.5초 안에서 방향만 맞으면 짝짓는다. **누가 넘었는지는 보지 않는다.**
두 사람이 비슷한 시각에 같은 방향으로 넘으면, 한 사람의 통과가 다른 사람의
정답에 짝지어질 수 있다. 실제로 `LeftBox`에서 baseline이 세 조각으로 쪼개진
track 중 하나가 낸 사건이 정답과 짝지어졌다(10절). **이 오염은 baseline과
selected에 똑같이 적용되므로 비교 방향은 유지되지만, 절대값은 낙관적일 수
있다.** A3부터 이어져 온 방식이라 A6에서 바꾸지 않았다. id를 보는 대조로
바꾸는 것은 다음 단계의 후보다.

**2. DEV의 통과 사건이 9건뿐이다.** 선택은 이 9건 위에서 이뤄졌다. 사건
한두 건이 순위를 뒤집을 수 있는 크기다. 그래서 FINAL(52건)에서 방향이
유지되는지가 이 단계의 실제 판정이고, ANALYSIS(86건)의 원인 분해가 그
메커니즘을 뒷받침한다. 그래도 **DEV 선택 자체의 통계적 근거는 약하다.**

**3. FINAL이 cross-camera이지 cross-site는 아니다.** corridor는 front와 다른
카메라·다른 각도지만 같은 쇼핑센터다. 각본 계열도 front와 겹친다(다른 촬영본).
**사건과 카메라는 겹치지 않지만 각본은 겹친다.** 완전한 독립 표본이 아니다.
cross-site 일반화에 대해서는 A5 로비 결과(recall 0.578 → 0.218)가 여전히
가장 정직한 참고값이다.

**4. 자동 원인 분류에 임의의 문턱이 있다.** `COVERAGE_FLOOR = 0.5`는 근거가
있는 값이 아니라 **결과를 보기 전에 정하고 바꾸지 않은** 값이다. 이 값을
움직이면 `DETECTION_GAP`과 `ASSOCIATION_BREAK`의 경계가 움직인다. 대표 사건을
직접 읽어 확인한 이유가 이것이다.

**5. occupancy 갈래는 A5에서 달라지지 않았다.** A6가 개선한 것은 entry
갈래다. occupancy는 여전히 검출 recall에 그대로 묶여 있고, 그 recall은
장소가 바뀌면 무너진다. **이 단계가 그 문제를 풀지 않았다.**

**6. 급식소가 아니다.** CAVIAR은 쇼핑센터와 사무실 로비다. 배식 줄의 밀집도,
쟁반을 든 자세, 조명, 카메라 높이가 모두 다르다. 이 숫자를 급식소 성능으로
옮겨 말하지 않는다.

**7. fine-tuning을 하지 않았다.** COCO pretrained 그대로다. 개선의 여지가
그만큼 남아 있다는 뜻이기도 하지만, CC BY-SA share-alike 때문에 이 데이터로
학습한 모델을 제품에 넣는 판단은 따로 해야 한다.

**8. tracker 비용 측정은 CPU 단일 스레드다.** 실제 서비스에서는 detector가
GPU를, tracker가 CPU를 쓰므로 겹칠 수 있다. 여기 숫자는 겹침을 고려하지 않은
보수적인 합이다.

## 16. Conclusion

### 무엇을 물었고 무엇을 알았나

**RQ-A6-1 — 과제별 임계값 분리가 같은 추론 위에서 각 과제를 개선하는가.**
**부분적으로.** DEV에서 임계값을 가르는 것만으로 통과 계수 오차가 0.30에서
0.20으로 줄었고, occupancy는 정책이 바뀌지 않아 그대로였다. FINAL에서는
이득이 나타나지 않았다 — corridor 시점은 사람이 크게 보여 낮은 신뢰도 상자를
열 이유가 적었기 때문이다. **구조는 옳지만 이득의 크기는 장면에 달렸다.**
추론을 두 번 돌리지 않고 이것을 시험할 수 있다는 것 자체는 확인되었다.

**RQ-A6-2 — 짧은 track 공백을 motion-aware tracking으로 이으면 통과가
회복되는가.** **그렇다. 그 원인의 사건에 한해서.** `ID_SWITCH`로 분류된
baseline 실패 13건 중 11건(85%)이 회복되었다. A5에서 detector를 통째로
바꿔 얻은 회복률이 43%였던 것과 비교된다.

**RQ-A6-3 — 연속성을 강화한 tracker가 통과 계수 오차를 줄이는가.**
**DEV/ANALYSIS에서는 그렇고, FINAL에서는 아니다.** DEV 0.30 → 0.10,
ANALYSIS front 1.44 → 0.75. FINAL corridor 0.83 → 0.83으로 **변화 없음.**
동결한 채택 조건을 통과하지 못했다.

**RQ-A6-4 — 개선이 새 카메라에서도 유지되는가.** **유지되지 않았다.**
그 이유는 세 층으로 나뉜다.

```
  1. tracker는 의도대로 작동했다
     fragmentation, ID switch가 네 집합 모두에서 감소했다 (FINAL IDSW −17%)

  2. 그런데 그 장면의 실패 원인이 달랐다
     front  놓친 통과의 43%가 ID_SWITCH
     corridor 놓친 통과의 10%만 ID_SWITCH

  3. corridor 실패의 다수는 파이프라인 밖의 문제였다
     52건 중 12건(23%)이 계수 ROI 밖에서 일어났다
```

### 채택 결정

**개선된 tracker를 기본값으로 채택하지 않는다.** 동결한 주 지표가 개선되지
않았고, 그 규칙을 결과를 본 뒤에 바꾸지 않는다.

`MotionAwareTracker`는 저장소에 남는다. 기본값이 아니라 선택지다. 근거는
이렇다 — FINAL에서 **나빠진 지표가 하나도 없고**(precision 0.967 → 1.000,
F1 0.746 → 0.774, fragmentation 12.16 → 11.29), 추가 비용이 프레임당
0.027 ms이며, 궤적 연속성 지표는 네 집합 모두에서 개선되었다. 다음 단계가
ROI와 창 크기를 카메라에 맞게 다시 정한 뒤 재평가할 수 있도록 코드와
동결 기록을 그대로 남긴다.

### 가설

| | 판정 |
|---|---|
| **H5** — 통과 누락은 detector recall보다 통과 순간의 시간적 연속성과 더 강하게 연결된다 | **지지된다** |
| **H6** — 낮은 신뢰도 상자를 허용하면 recall이 오르고 motion association이 false crossing을 억제한다 | **부분적으로 지지된다** |

H5의 근거는 두 개다. 검출이 충분했던 사건만 봐도(n=42) 맞힘과 놓침의 창
연속성 차이가 0.272인 반면 검출 coverage 차이는 0.087이었고, 연속성을 고치자
그 원인의 사건이 85% 돌아왔다.

H6는 low 0.3까지만 참이다. 0.1까지 내리면 precision이 0.770까지 떨어진다.
그리고 사람이 크게 보이는 장면에서는 애초에 이득이 없다.

### 이 단계의 값어치

숫자는 좋아지지 않았다. 그래도 이 단계는 셋을 남긴다.

**1. 오류의 출처를 이름 붙여 나눴다.** "통과를 놓친다"가
`DETECTION_GAP` / `ASSOCIATION_BREAK` / `TRACK_EXPIRY` / `ID_SWITCH` /
`LINE_GEOMETRY`로 갈렸고, 각각이 몇 건인지 세었다. 이 분해가 없었다면
"tracker를 좋게 만들면 통과 계수가 좋아진다"는 기대를 계속 붙들었을 것이다.
실제로는 **원인의 비중이 카메라마다 다르고**, 그것이 개입의 상한을 정한다.

**2. 개입이 겨냥한 원인을 정확히 맞혔다는 것을 보였다.** 겨냥한 원인의 85%를
회복시켰다. 개입이 실패한 것이 아니라 **그 원인이 FINAL 장면에서 드물었다.**
이 둘을 구분하지 못하면 "복잡한 tracker는 소용없다"는 틀린 일반화를 하게 된다.

**3. 파이프라인 밖의 병목을 찾았다.** FINAL 실패의 23%는 tracker와 무관하다 —
카메라가 바뀌었는데 ROI와 통과 창을 그대로 썼기 때문이다. **이것은 다음
단계에서 고칠 수 있고, tracker보다 효과가 크다.**

A5는 "해상도도 모델 크기도 아니다"를 배제했다. A6는 "궤적 연속성이 원인의
하나이지만 지배적인 원인은 장면에 따라 달라진다"를 확인했고, 다음에 고칠 곳을
**카메라별 기하 설정**으로 좁혔다.

### CCTV 연구 트랙을 여기서 닫는다

닫을 수 있다고 보는 이유는 이렇다.

```
  배제한 것    입력 해상도 (A5)
               detector 용량 (A5)
               작은 사람 가설 H4 (A5)
               선 판정 규칙 (A6 — LINE_GEOMETRY 0건)
               소멸 정책 (A6 — TRACK_EXPIRY 0건)

  확인한 것    공유 detector + 과제별 정책 구조가 성립한다
               궤적 연속성이 통과 누락의 실재하는 원인이다
               추적 비용은 무시할 수 있다 (프레임당 0.027 ms)

  남은 것      cross-site 검출 recall (A5, 미해결)
               카메라별 ROI / 통과 창 (A6에서 발견)
               entry count를 담을 백엔드 계약 (14절)
               급식소 데이터 자체
```

**남은 것들은 연구가 아니라 통합과 데이터의 문제다.** 카메라별 기하 설정은
현장 설치 때 정하는 값이고, entry count 저장은 백엔드 도메인 설계이며,
급식소 성능은 급식소 영상 없이는 알 수 없다. 그래서 다음 단계는 **Demo
Integration**이다.

다만 한 문장을 분명히 남긴다. **지금 수치로 발주를 움직이면 안 된다.**
통과 계수 recall이 0.6~0.8이고 occupancy bias가 −1.3명이다. 이 값을 보정
없이 수요 정답으로 쓰면 실제보다 적게 세고, 그 정답으로 학습한 모델이 다시
적게 발주한다.

## 재현

```bat
:: 1. 데이터 (재배포하지 않는다. 공식 출처에서 직접 받는다)
::    datasets/a6_dataset_gate.md 의 규칙과
::    datasets/a6_caviar_corridor_checksums.json 의 지문으로 확인한다

:: 2. 동결 (FINAL 실행 전)
.venv-cctv\Scripts\python.exe research\cctv_counting\scripts\a6_freeze.py

:: 3. 실험
.venv-cctv\Scripts\python.exe research\cctv_counting\scripts\a6_run.py ^
  --split DEV --out research\cctv_counting\experiments\a6_continuity\development.json
.venv-cctv\Scripts\python.exe research\cctv_counting\scripts\a6_run.py ^
  --split ANALYSIS --out research\cctv_counting\experiments\a6_continuity\analysis_shop.json
.venv-cctv\Scripts\python.exe research\cctv_counting\scripts\a6_run.py ^
  --split ANALYSIS_LOBBY --out research\cctv_counting\experiments\a6_continuity\analysis_lobby.json
.venv-cctv\Scripts\python.exe research\cctv_counting\scripts\a6_run.py ^
  --split FINAL --out research\cctv_counting\experiments\a6_continuity\final.json

:: 4. 집계와 분석
.venv-cctv\Scripts\python.exe research\cctv_counting\scripts\a6_aggregate.py ^
  research\cctv_counting\experiments\a6_continuity\final.json
.venv-cctv\Scripts\python.exe research\cctv_counting\scripts\a6_h5.py ^
  research\cctv_counting\experiments\a6_continuity\analysis_shop.json ^
  research\cctv_counting\experiments\a6_continuity\analysis_lobby.json
.venv-cctv\Scripts\python.exe research\cctv_counting\scripts\a6_recovery.py ^
  research\cctv_counting\experiments\a6_continuity\final.json
.venv-cctv\Scripts\python.exe research\cctv_counting\scripts\a6_latency.py --split FINAL

:: 5. 사건 하나를 프레임 단위로 읽기
.venv-cctv\Scripts\python.exe research\cctv_counting\scripts\a6_inspect.py ^
  --split ANALYSIS --sequence ThreePastShop1front --policy P0_baseline --track 5

:: 6. 테스트
.venv\Scripts\python.exe -m pytest
```

캐시된 검출 상자가 있으면 3번은 GPU 없이도 돈다. 상자가 없으면 그 시퀀스만
detector를 **한 번** 돌리고 캐시한다.
