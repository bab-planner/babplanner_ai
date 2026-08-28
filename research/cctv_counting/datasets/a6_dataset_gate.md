# Stage A6 — Fresh FINAL 데이터셋 게이트

**이 문서는 데이터를 내려받기 전에 작성했다.** A4의 `a4_dataset_gate.md`와
같은 순서다. 결과를 보고 규칙을 바꾸지 않는다.

## 1. 왜 새 FINAL이 필요한가

A6 시작 시점에서 기존 집합의 역할이 바뀐다.

| 집합 | A4/A5에서의 역할 | A6에서의 역할 | 이유 |
|---|---|---|---|
| 쇼핑센터 front 5편 | DEVELOPMENT | **DEVELOPMENT** | 그대로 |
| 쇼핑센터 front 8편 | A4 TEST → A5 ANALYSIS | **ANALYSIS** | 이미 여러 번 봤다 |
| INRIA 로비 17편 | A5 FINAL (1회) | **ANALYSIS** | A5에서 결과를 봤다. 더 이상 unseen이 아니다 |

**A5 FINAL에서 parameter selection을 하지 않는다.** 한 번 본 집합은
소비된 집합이다. A6의 판정에는 한 번도 추론한 적 없는 새 집합이 필요하다.

## 2. 라이선스 게이트

같은 공식 출처다. **다시 확인했다 (2026-08-28).**

출처: `https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/` (HTTP 200)

원문:

> "If you publish results using the data, please acknowledge the data as
> coming from the EC Funded CAVIAR project/IST 2001 37540, found at URL:
> http://homepages.inf.ed.ac.uk/rbf/CAVIAR/"
>
> "Usability is Creative Commons BY-SA"

A4에서 받은 쇼핑센터 front 영상과 **같은 페이지, 같은 문단, 같은 조건**이다.
동일 배포 단위이므로 새로 판정할 것이 없다. 판정: **VERIFIED.**

같은 페이지의 두 번째 데이터 집합 설명:

> "The second set of data also used a wide angle lens along and across the
> hallway in a shopping centre in Lisbon. For each sequence, there are two
> time synchronised videos, one with the view across and the other along the
> hallway. The resolution is half-resolution PAL standard (384 x 288 pixels,
> 25 frames per second)"

**재배포하지 않는다.** 영상과 annotation XML은 커밋하지 않는다
(`.gitignore`의 `research/cctv_counting/datasets/raw/`).

**fine-tuning 하지 않는다.** 평가에만 쓴다. CC BY-SA의 share-alike를
파생 모델까지 끌고 가지 않기 위해서다. A4/A5와 같다.

## 3. 시퀀스 선택 규칙 (다운로드 전 고정)

목표는 **처음으로 cross-camera 평가**를 하는 것이다.

```
  1. 쇼핑센터 "corridor"(along the hallway) 시점만 쓴다.
     - front(across)와 다른 카메라, 다른 시점이다.
     - front 13편은 DEVELOPMENT / ANALYSIS로 이미 소비되었다.

  2. 계열의 **두 번째 촬영본(2cor)만** 쓴다.
     - 공식 설명대로 1cor 와 1front 는 "time synchronised" 즉 같은 사건을
       두 카메라로 찍은 것이다. 1cor 를 FINAL에 넣으면 DEVELOPMENT/ANALYSIS
       에서 이미 본 사건을 다시 채점하게 된다. 카메라만 다르고 사건은 같다.
     - 2cor 는 **사건도 카메라도** 한 번도 쓰지 않았다.
       (2front 역시 A4 규칙 2에 의해 한 번도 쓰지 않았다.)

  3. 자격 있는 계열 전부를 쓴다. 고르지 않는다.

  4. 분할 없음. FINAL TEST 전체다. 여기서 parameter를 고르지 않는다.
```

이 규칙으로 자격을 얻는 시퀀스 = `*2cor` 가 존재하는 계열 전부.
공식 목록 기준 **12편**이다.

| # | 시퀀스 | GT XML |
|---|---|---|
| 1 | `EnterExitCrossingPaths2cor` | `ceecp2gt.xml` |
| 2 | `OneLeaveShop2cor` | `cols2gt.xml` |
| 3 | `OneLeaveShopReenter2cor` | `colsr2gt.xml` |
| 4 | `OneShopOneWait2cor` | `cosow2gt.xml` |
| 5 | `OneStopEnter2cor` | `cose2gt.xml` |
| 6 | `OneStopMoveEnter2cor` | `cosme2gt.xml` |
| 7 | `OneStopMoveNoEnter2cor` | `cosmne2gt.xml` |
| 8 | `OneStopNoEnter2cor` | `cosne2gt.xml` |
| 9 | `ShopAssistant2cor` | `csa2gt.xml` |
| 10 | `ThreePastShop2cor` | `c3ps2gt.xml` |
| 11 | `TwoEnterShop2cor` | `c2es2gt.xml` |
| 12 | `TwoLeaveShop2cor` | `c2ls2gt.xml` |

최소 3편, 권장 5편 이상이라는 기준을 넘는다.

`TwoEnterShop3cor`(세 번째 촬영본)는 **규칙 2에 걸려 쓰지 않는다.**
"두 번째 촬영본"이라는 규칙을 세 번째까지 늘리면 그것은 규칙이 아니라 편의다.

## 4. 평가 규약 — front/로비와 **똑같이** 쓴다

카메라가 바뀌었다고 선이나 ROI를 다시 맞추지 않는다. 맞추기 시작하면
그 숫자는 "이 카메라에 맞춘 뒤의 성능"이지 일반화 성능이 아니다.

```
  ROI                     NormalizedRect(0.0, 0.20, 1.0, 0.95)   A4와 동일
  LINE                    x = 0.5 세로선                          A4와 동일
  IN_DIRECTION            POSITIVE (오른쪽 -> 왼쪽)               A4와 동일
  BAND                    0.01                                    A4와 동일
  MIN_CONFIRM_DISTANCE    0.03                                    A4와 동일
  COOLDOWN_SECONDS        0.5                                     A4와 동일
  SAMPLE_EVERY            25 (1초)                                A4와 동일
  EVENT_TOLERANCE         1.5초                                   A3에서 고정
  프레임 크기 / fps        384x288 / 25                            공식 문서
```

**시퀀스마다 선을 옮기지 않는다.** 12편 전부 같은 선이다.

## 5. 미리 적어 두는 한계

1. corridor 시점은 복도를 **따라** 보는 각도다. front(가로지르는 각도)보다
   세로선을 넘는 사건이 적을 수 있다. 사건 수가 적으면 통과 지표의 신뢰구간이
   넓어진다. 그 경우 **규칙을 바꾸지 않고 표본 수를 그대로 보고한다.**
2. 같은 쇼핑센터다. cross-camera 이지만 cross-site는 아니다.
   (A5 FINAL 로비가 cross-site였고, 그것은 이제 ANALYSIS다.)
3. 급식소가 아니다. 이 숫자를 급식소 성능으로 옮겨 말하지 않는다.
4. 계열은 front DEVELOPMENT/ANALYSIS와 겹친다(같은 각본의 다른 촬영).
   **사건과 카메라는 겹치지 않지만 각본은 겹친다.** 완전한 독립 표본이
   아니라는 뜻이고, 그대로 적는다.
