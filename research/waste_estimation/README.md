# AI-2. 식판 이미지 기반 잔반 분석

*Vision-based Food Waste Severity Estimation with Selective Prediction*

> **이 폴더에는 모델 코드가 없다.** 이번 단계에서 만들지 않았다. 여기 있는
> 것은 무엇을 만들 것인지에 대한 정의다.

## 목표

식판 사진 한 장을 받아, 얼마나 남겼는지와 **그 판단을 믿어도 되는지**를 낸다.

```
image
  → food / waste representation
  → severity (연속값)
  → confidence
  → APPROVE / REJECT
```

마지막 단계가 이 문제의 핵심이다. 흐릿하거나 식판이 제대로 안 보이는 사진에
대해 아무 숫자나 내면, 그 숫자로 포인트가 나간다. **틀린 값을 자신 있게
내보내는 것이 값을 안 내보내는 것보다 나쁘다.**

## 이미 정해진 것

정답의 정의는 이미 닫혀 있다. `docs/waste_ground_truth_protocol.md`.

    waste_rate = leftover_edible_mass_g / served_edible_mass_g

**질량비다.** 사진의 면적 비율도, 사람의 눈대중도 정답으로 쓰지 않는다. 둘 다
재기는 쉽지만 서로 다른 값이고, 섞이는 순간 모델이 무엇을 배웠는지 알 수 없게
된다.

서비스 계약과 데이터셋 도구도 이미 있다.

- `src/babplanner_ai/waste/` — API와 `WasteEstimator` 경계. 지금은 fixture
- `src/babplanner_ai/dataset/` — 매니페스트 검증, 요약, 분할
- `src/babplanner_ai/collection/` — 현장 수집 도구
- `docs/waste_dataset_spec.md` — 매니페스트 열, 개인정보 규칙
- `docs/waste_evaluation_protocol.md` — 지표와 Go/No-Go

**그래서 이 트랙에 없는 것은 모델 하나뿐이다.** 그리고 그것을 만들려면 데이터가
필요하다.

## 핵심 어려움

**먹기 전 사진이 없다.** 남은 양은 보이는데 원래 얼마였는지는 안 보인다. 같은
양이 남아 있어도 많이 받았으면 적게 남긴 것이다. 모델은 한 장의 사진에서
분모를 추정해야 한다.

**촬영 조건이 매번 다르다.** 각도, 거리, 조명, 그림자. 같은 잔반이 다르게 보인다.

**식판 구성이 현장마다 다르다.** 칸 수도 배치도 메뉴도 다르다. 한 학교에서
학습한 모델이 다른 학교에서 그대로 되지 않는다.

**정답 자체에 오차가 있다.** 저울로 재도 식판 무게, 국물, 계량 순서에서 오차가
생긴다. 모델 오차와 측정 오차를 구분해야 한다.

## 왜 선택적 예측인가

이 문제는 정답률 100%가 불가능하다. 그런데 이 값으로 포인트가 나간다.
그래서 "얼마나 맞히는가"보다 **"틀릴 때를 미리 아는가"** 가 중요하다.

지향하는 모습:

```
             ┌─ APPROVE → 잔반률을 낸다. 이 구간의 MAE가 보장된다
image → 모델 ┤
             └─ REJECT  → 숫자를 내지 않는다. 사람이 보거나 재촬영을 요청한다
```

평가도 두 축이다.

- **coverage** — 몇 %를 APPROVE 했는가
- **selective risk** — APPROVE 한 것들의 MAE

전체 MAE 하나로는 이 모델의 좋고 나쁨을 말할 수 없다. 90%를 거절하고 남은
10%에서 MAE 0.02를 내는 모델과, 전부 받아 MAE 0.12를 내는 모델 중 무엇이
나은지는 운영 정책이 정한다.

`AnalysisLabel.APPROVED / REJECTED`와 `RejectionReason`은 이미 스키마에 있다
(`src/babplanner_ai/dataset/schema.py`). 계약이 먼저 서 있는 상태다.

## 시작 조건

**파일럿 데이터가 필요하다.** 지금 저장소에 있는 것은 합성 fixture뿐이다.
`docs/waste_pilot_collection_runbook.md`가 현장 수집 절차이고, 그것이 돌기
전까지 이 트랙은 시작되지 않는다.

## 다음 단계 (착수 전)

1. 파일럿 수집 실행. 최소 몇 백 장 + 저울 정답
2. `dataset validate` / `summarize` / `make-splits`로 데이터 상태 확인
3. 단순 baseline부터. 전역 평균 예측, 색/면적 특징 + 선형 회귀
4. 그 위에서 CNN 회귀
5. confidence 추정과 APPROVE/REJECT 임계값
6. coverage-risk 곡선으로 평가

3번을 건너뛰지 않는다. 딥러닝 모델이 MAE 0.09를 냈을 때, 전역 평균이 0.11이면
그것은 거의 아무것도 배우지 못한 것이다. 그 사실은 비교 대상이 있어야 보인다.

## 관련

- [research/README.md](../README.md) — 세 트랙의 공통 목표
- [cctv_counting](../cctv_counting/) — 신뢰 신호를 먼저 다뤄 본 트랙
- `docs/waste_ground_truth_protocol.md`
- `docs/waste_evaluation_protocol.md`
