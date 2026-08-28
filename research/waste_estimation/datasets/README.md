# 공개 데이터 후보

조사 시점: 2026-08-28
코드로 옮긴 것: `research/waste_estimation/src/waste_estimation/public_datasets.py`

> **이 디렉터리에 데이터를 받지 않았다.** W1에서 하는 일은 조건을 읽고
> 역할을 정하는 것까지다. 조건을 확인하기 전에 받으면, 쓸 수 없는 데이터를
> 이미 받아 둔 상태가 된다.

---

## 1. 판정 규칙 (CCTV 트랙에서 그대로 가져옴)

```
  공식 배포 페이지에서 조건 원문을 읽지 못한 데이터는 쓰지 않는다.
  GitHub 저장소 코드의 라이선스를 데이터셋 라이선스로 읽지 않는다.
  2차 출처(블로그, 재업로드, 요약 사이트)를 법적 근거로 쓰지 않는다.
```

| 상태 | 뜻 |
|---|---|
| `VERIFIED` | 공식 페이지에서 조건 원문을 직접 읽었다 |
| `PARTIALLY_VERIFIED` | 일부만. 예: 논문은 공개인데 데이터 조건은 못 읽음 |
| `UNVERIFIED` | 못 읽었다. **쓰지 않는다** |

코드가 이 규칙을 강제한다. `UNVERIFIED`인데 역할을 주면
`PublicDataset.__post_init__`이 거부한다.

## 2. 이 트랙에만 있는 위험

음식 분할 데이터셋에는 픽셀 마스크가 있다. 그것으로 "남은 면적 비율"을
계산하는 것은 쉽다. **그 값을 우리 정답으로 쓰면 안 된다.**

```
  남은 픽셀 비율   ≠   leftover_edible_mass_g / served_edible_mass_g
```

같은 면적이라도 두께가 다르고, 같은 부피라도 밀도가 다르다. 국물 한 국자와
밥 한 덩이는 화면에서 비슷해 보여도 무게가 몇 배 차이 난다. 두 값을 이어
붙이는 순간 모델이 무엇을 배웠는지 알 수 없게 되고, 그 모델의 오차를
"잔반률 오차"라고 부를 수 없게 된다.

그래서 등록부는 데이터마다 **질량 기반 정답이 있는지**를 따로 기록한다.

## 3. 후보

| 데이터셋 | 규모 | 질량 GT | before/after | 분할 마스크 | 라이선스 | 상태 | 역할 |
|---|---:|:---:|:---:|:---:|---|---|---|
| **Nutrition5k** | ~5,006 접시 | ✓ | ✗ | ✗ | Creative Commons 4.0 | **VERIFIED** | 표현 학습 |
| **LeFood-set** | 524 짝 | ✓ | ✓ | ✗ | CC BY 4.0 | **VERIFIED** | 외부 벤치마크 / 표현 학습 |
| FLIC | 401 짝 | ✓ | ✓ | ✓ | (못 읽음) | UNVERIFIED | **없음** |
| UNIMIB2016 | 1,027 식판 | ✗ | ✗ | ✓ | (못 읽음) | UNVERIFIED | **없음** |
| FoodSeg103 | 9,490 장 | ✗ | ✗ | ✓ | (표기 없음) | UNVERIFIED | **없음** |

### Nutrition5k — VERIFIED

<https://github.com/google-research-datasets/Nutrition5k>

> "We release all Nutrition5k data under the Creative Commons V4.0 license.
> You are free to share and adapt this data for any purpose, even commercially."

접시당 **재료별·전체 질량(g)** 과 열량·영양소가 있다. 위에서 찍은 RGB-D와
회전 영상이 있다.

**우리 정답과 다른 점.** 제공된 음식만 있고 **먹고 난 뒤가 없다.** 즉
`served_edible_mass_g`에 해당하는 것은 있어도 `leftover_edible_mass_g`가
없다. 잔반률을 만들 수 없다.

**그래서 무엇에 쓰는가.** 이 트랙의 가장 근본적인 질문 — "사진 한 장에서
음식의 **질량**을 어느 정도 맞힐 수 있는가" — 을 우리 데이터 없이 미리 재 볼
수 있는 유일한 후보다. 그 상한이 낮게 나오면, 잔반률 회귀의 기대치도 그만큼
낮춰 잡아야 한다.

### LeFood-set — VERIFIED

<https://data.mendeley.com/datasets/cchsk79jkt/1>
논문: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12087991/>

Mendeley Data의 라이선스 표기는 **CC BY 4.0**이다.

524개의 **식전/식후 짝**, 34개 음식 범주, **저울로 잰 식전·식후 그램**,
그리고 사람이 매긴 7단계 잔반 정도.

**우리 정의와 가장 가까운 공개 데이터다.** 식전·식후 그램이 둘 다 있으므로
우리와 같은 방식의 질량비를 계산할 수 있다.

**그래도 우리 서비스의 정답이 아니다.** 인도네시아 병원 급식이고 식판 구성,
메뉴, 촬영 조건이 한국 대학 학식과 다르다. 여기서 나온 성능을 급식소 성능으로
옮겨 말하지 않는다. **외부 벤치마크**로만 쓴다 — 같은 과제를 다른 도메인에서
풀었을 때 어느 정도가 나오는지의 참고치.

> Mendeley(Elsevier) 사이트 하단에는 "All rights are reserved, including
> those for text and data mining, AI training and similar technologies"라는
> 사이트 공통 문구가 있다. 데이터 레코드 자체의 라이선스 필드는 CC BY 4.0으로
> 명시되어 있어 그것을 따르되, 실제 사용 전에 한 번 더 확인한다.

### FLIC — UNVERIFIED

<https://doi.org/10.3390/app16115465> (Applied Sciences, 2026)

**과제가 거의 같다.** 대학 구내식당에서 22일간 모은 401개의 식전/식후 식판
짝, 픽셀 단위 분할 마스크, 그리고 **실제 계측된 질량**. 우리가 만들려는
데이터셋과 구조가 같다.

**그런데 배포 조건을 읽지 못했다.** MDPI 본문 페이지가 403을 돌려주어 Data
Availability Statement와 데이터 자체의 라이선스를 확인하지 못했다. 논문이
오픈 액세스(CC BY)라는 것은 **논문의 라이선스**이지 데이터셋의 라이선스가
아니다 — CCTV 트랙에서 WILDTRACK을 제외한 것과 같은 이유다.

**판정: 쓰지 않는다.** 다만 W2에서 접근이 되면 가장 먼저 다시 볼 후보다.

### UNIMIB2016 — UNVERIFIED

<http://www.ivl.disco.unimib.it/activities/food-recognition/>

이탈리아 셀프서비스 식당의 식판 1,027장, 73개 음식 범주, 손으로 그린 다각형
분할. 식판이라는 점은 가깝다.

공식 페이지 접속이 실패했다(`ECONNREFUSED`). 조건 원문을 읽지 못했고,
**질량 정답도 before/after도 없다.** 분할 마스크만으로는 우리 회귀 정답을
만들 수 없다(2절).

### FoodSeg103 — UNVERIFIED

<https://xiongweiwu.github.io/foodseg103.html>

9,490장, 재료 단위 픽셀 분할. 공식 페이지에 라이선스 표기가 없다. 식판도
잔반도 아닌 웹 음식 사진이다.

## 4. 허용하는 역할

```
  A. 표현 학습 / 사전학습 / 분할 타당성        Nutrition5k, LeFood-set
  B. 같은 과제의 외부 벤치마크                 LeFood-set
  C. 시연 파이프라인 입력                      (지금 없음)
```

**공개 데이터를 우리 서비스의 정답으로 직접 쓰지 않는다.** 질량 정답이 있는
Nutrition5k와 LeFood-set조차 우리 급식소의 `waste_rate`가 아니다. 그 값으로
학습한 모델을 급식소에 그대로 붙이는 판단은 별도 검증을 거친다.

## 5. 다운로드

**W1에서 아무것도 받지 않았다.** 필요해지는 시점은 W2에서 다음 둘 중 하나를
할 때다.

1. Nutrition5k로 "사진 한 장에서 질량 추정" 상한을 재 본다
2. LeFood-set으로 잔반률 회귀의 외부 참고치를 만든다

받기 전에 각 데이터의 배포 조건을 다시 한 번 확인하고, 이 문서의 표를
갱신한다.
