# Stage A4 — 데이터셋 라이선스 게이트와 시퀀스 선택 규칙

**이 문서는 데이터를 내려받기 전에 작성했다.** 결과를 보고 규칙을 바꾸지
않는다. 바꾸면 그 뒤의 숫자는 아무것도 재지 않은 것이 된다.

## 1. MOT17 — 사용하지 않음

A3까지 1순위 후보였다. **공식 terms 원문을 끝내 확인하지 못해 제외한다.**

```
  motchallenge.net    DNS 정상 (131.159.19.34, 2a09:80c0:18::1034)
                      HTTP/HTTPS 모두 ECONNREFUSED
                      A2, A3, A4 세 번 연속 실패
```

2차 출처들은 CC BY-NC-SA 3.0이라고 일관되게 말하지만, **공식 문서를 읽지
못한 라이선스로 데이터를 받지 않는다.** GitHub 미러, Kaggle 재업로드, 블로그
설명을 법적 근거로 쓰지 않는다는 원칙 그대로다.

판정: **UNVERIFIED → 사용 금지.**

## 2. 대안 조사

| 후보 | 조건 확인 | 판정 |
|---|---|---|
| **CAVIAR** (Edinburgh / EC IST 2001 37540) | 공식 페이지 접속 성공(HTTP 200), 라이선스 원문 확인 | **VERIFIED → 채택** |
| WILDTRACK (EPFL CVLAB) | 부속 toolkit은 GPL-3로 확인되나 **데이터 자체의 조건**을 공식 페이지에서 확인하지 못함 | UNVERIFIED → 보류 |
| PETS2009 | A2부터 출처 접근성·조건 모두 확인 실패 | UNVERIFIED → 제외 |

> WILDTRACK은 A2에서 CrowdHuman으로 겪었던 함정과 같은 모양이다. **저장소
> 코드의 라이선스(GPL-3)를 데이터 라이선스로 읽으면 안 된다.** 데이터 조건을
> 따로 확인하지 못했으므로 쓰지 않는다.

## 3. 채택: CAVIAR

**공식 출처.** `https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/`

**라이선스 원문 (2026-08-28 직접 확인):**

> "If you publish results using the data, please acknowledge the data as
> coming from the EC Funded CAVIAR project/IST 2001 37540, found at URL:
> http://homepages.inf.ed.ac.uk/rbf/CAVIAR/"
>
> "Usability is Creative Commons BY-SA"

판정: **VERIFIED.**

### 이 라이선스로 할 수 있는 것과 없는 것

**할 수 있는 것.** 연구·포트폴리오 평가에 쓰고 결과를 공개하는 것. 출처
표기가 필수다.

**주의할 것.** CC BY-SA는 비상업 제한이 없지만 **share-alike** 조건이 있다.
이 데이터로 만든 파생물을 배포하면 같은 조건을 따라야 한다. 그래서
**이번 단계에서는 평가에만 쓰고 fine-tuning을 하지 않는다.** 이 데이터로
학습한 모델을 밥플래너 production에 넣어도 된다는 뜻으로 확대 해석하지
않는다. 그 판단이 필요해지면 그때 따로 검토한다.

**재배포하지 않는다.** 원본 영상과 annotation XML은 저장소에 커밋하지 않는다.

### 왜 CAVIAR가 이 과제에 맞는가

| | |
|---|---|
| 카메라 | **고정**. 우리 CCTV 설치 조건과 같다 |
| 장면 | **실내** — 사무실 로비, 쇼핑센터 복도 |
| annotation | 프레임별 사람 bounding box + **사람 id** (CVML XML) |
| 해상도 | 384 x 288, 25 fps |

A2/A3에서 쓴 `vtest.avi`는 실외 보행로였다. CAVIAR의 쇼핑센터 복도는
**실내이고 사람이 통로를 따라 오가는 장면**이라 급식소 통로에 더 가깝다.
그래도 급식소는 아니다.

## 4. 시퀀스 선택 규칙 (다운로드 전 고정)

결과가 좋아 보이는 시퀀스를 고르면 그 숫자는 의미가 없다. 규칙을 먼저 정한다.

```
  1. Lisbon 쇼핑센터 "front" 시점만 쓴다.
     - 실내 복도 정면 뷰. 급식소 통로에 가장 가깝다.
     - 같은 활동의 두 번째 시점(corridor)은 쓰지 않는다.
       같은 사건의 두 뷰는 독립 표본이 아니다.

  2. 시나리오 계열별로 하나씩만 쓴다.
     - 이름에서 끝의 숫자와 "front"를 뗀 것을 계열로 본다.
       예: OneLeaveShop1front, OneLeaveShop2front -> 계열 "OneLeaveShop"
     - 같은 계열의 다른 촬영본은 같은 장소에서 같은 각본을 다시 찍은
       것이라 상관이 높다. 계열당 알파벳 첫 번째만 쓴다.

  3. 자격 있는 계열 전부를 쓴다 (13개). 고르지 않는다.

  4. 분할: 계열 알파벳 순 앞 5개 = DEV, 나머지 8개 = TEST.
```

### 규칙 3의 수정 이력 (투명하게 적어 둔다)

처음에는 "계열 알파벳 순 앞 5개"로 정하고 그 5개를 받았다. 정답을 만들어
보니 **통과 사건이 시퀀스 전체에서 IN 3건, OUT 6건뿐**이었다. 그 크기로는
입장 계수에 대해 아무 말도 할 수 없다.

그래서 규칙을 **"자격 있는 계열 전부(13개)"** 로 넓혔다. 고른 것이 아니라
**남김없이 전부** 쓴 것이므로 cherry-pick이 아니다.

이 변경 시점을 분명히 해 둔다.

```
  보고 바꿨는가   정답(GT) 통계를 보고 바꿨다
  안 본 것        예측 결과는 이 시점에 하나도 없었다
  DEV 변화        없음. 앞 5개가 그대로 DEV로 남는다
  TEST 변화       8개가 새로 추가되었다 (그 전에는 3개)
```

즉 **넓힌 부분은 전부 TEST 쪽이고, 그 위에서 예측을 돌린 적이 없다.**
DEV는 원래 정한 그대로다.

### 최종 목록

| # | 시퀀스 | GT XML | 분할 |
|---|---|---|---|
| 1 | `EnterExitCrossingPaths1front` | `feecp1gt.xml` | **DEV** |
| 2 | `OneLeaveShop1front` | `fols1gt.xml` | **DEV** |
| 3 | `OneLeaveShopReenter1front` | `folsr1gt.xml` | **DEV** |
| 4 | `OneShopOneWait1front` | `fosow1gt.xml` | **DEV** |
| 5 | `OneStopEnter1front` | `fose1gt.xml` | **DEV** |
| 6 | `OneStopMoveEnter1front` | `fosme1gt.xml` | TEST |
| 7 | `OneStopMoveNoEnter1front` | `fosmne1gt.xml` | TEST |
| 8 | `OneStopNoEnter1front` | `fosne1gt.xml` | TEST |
| 9 | `ShopAssistant1front` | `fsa1gt.xml` | TEST |
| 10 | `ThreePastShop1front` | `f3ps1gt.xml` | TEST |
| 11 | `TwoEnterShop1front` | `f2es1gt.xml` | TEST |
| 12 | `TwoLeaveShop1front` | `f2ls1gt.xml` | TEST |
| 13 | `WalkByShop1front` | `fwbs1gt.xml` | TEST |

**DEV 5개 / TEST 8개.**

**영상 단위 분할이다.** 같은 영상의 앞뒤를 나눠 cross-video라고 부르지
않는다. A3의 holdout은 같은 영상의 뒤 구간이었고, 이번은 다른 영상이다.

## 5. 미리 적어 두는 한계

13개 시퀀스가 **전부 같은 쇼핑센터의 같은 카메라**다. 서로 다른 촬영이지만
장소도 시점도 조명도 같다. 따라서 이것은 **cross-sequence** 평가이지
**cross-camera / cross-site** 평가가 아니다. 보고서에도 그렇게 적는다.

그리고 급식소가 아니다. 이 결과를 급식소 성능으로 옮겨 말하지 않는다.
