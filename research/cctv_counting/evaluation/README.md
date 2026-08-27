# 평가 절차

지표 구현은 `src/cctv_counting/metrics.py`에 있고, 실행 진입점은
`scripts/evaluate.py`다. 여기는 그것을 **어떤 순서로 어떤 조건에서** 돌릴
것인지를 정한다.

## 1. 두 baseline을 같은 조건에서 돌린다

```
A. detection only        configs/baseline_detection.yaml
B. detection + tracking  configs/baseline_tracking.yaml
```

두 설정의 detector 항목과 line/ROI 항목은 **같아야 한다.** 그래야 결과 차이를
추적 유무 탓으로 돌릴 수 있다. `tests/test_cctv_config.py`가 이것을 고정한다.

묻는 것:

- **Occupancy에서 추적이 도움이 되는가?** 꼭 그렇지 않을 수 있다. 추적은
  `min_hits` 때문에 새로 나타난 사람을 늦게 잡고, `max_age` 때문에 나간 사람을
  늦게 놓는다. 프레임별 상자를 그냥 세는 쪽이 순간값으로는 더 정확할 수도 있다.
  **재 보기 전에는 모른다.**
- **Entry counting에서 추적이 얼마나 필요한가?** 이 비교는 해석에 주의가
  필요하다. A는 **완전한 entry-count baseline이 아니다.** 추적 없이는 "넘었다"는
  사건 자체가 정의되지 않으므로, A가 못 세는 것은 성능이 나빠서가 아니라
  정의상 그렇다. A가 기록하는 `naiveLineBandDetections`(선 근처 상자 누적)는
  입장 인원이 아니라 순진한 대리 지표이고, 그 값이 실제 입장 인원보다 얼마나
  부풀려지는지를 보는 용도다.

  제대로 된 detection-only crossing heuristic(예: 선 양쪽의 상자 수 변화를
  보는 방식)을 나중에 추가할 수는 있다. 다만 그것은 벤치마크 단계에서 실제
  필요가 확인된 뒤의 일이고, **지금 알고리즘을 늘리지 않는다.**

## 2. Confidence threshold 실험

`--confidence`로 설정을 덮어써서 같은 영상을 여러 번 돌린다.

```
0.2  많이 잡는다. 오탐이 늘어 과대 계수 가능
0.4  기본값
0.6  적게 잡는다. 누락이 늘어 과소 계수 가능
```

**어떤 값이 최선인지 미리 가정하지 않는다.** 셋 다 돌리고 MAE와 bias를 같이
본다. bias의 부호가 임계값에 따라 뒤집히는 지점이 있다면 그것이 이 실험에서
가장 볼 만한 결과다.

각 실행은 `metadata.json`에 자기 설정을 남기므로, 나중에 어떤 숫자가 어떤
임계값에서 나온 것인지 알 수 있다.

## 3. 조건별로 나눠 본다

전체 MAE 하나로는 "언제 못 쓰는가"를 알 수 없다. 조건 태그가 붙은 표본에
대해 조건별 표를 만든다.

| Condition | Samples | Occupancy MAE | Entry Error |
|---|---:|---:|---:|

태그가 없으면 이 표를 만들지 않는다. 없는 조건을 지어내면 표는 채워지지만
그 표는 거짓이다.

## 4. Runtime을 같이 기록한다

`processing_fps`와 `metadata.json`의 장치/해상도를 함께 본다. 정확도가 조금
좋아졌는데 속도가 절반이 되었다면 서비스에서는 개선이 아닐 수 있다.

## 실행 순서

```
1. 데이터 확보     datasets/README.md 의 라이선스 확인 절차
2. 정답 작성       annotations/*.csv
3. A 실행          run_video.py --config configs/baseline_detection.yaml
4. B 실행          run_video.py --config configs/baseline_tracking.yaml
5. 임계값 스윕     --confidence 0.2 / 0.4 / 0.6
6. 채점            evaluate.py
7. 실패 사례 관찰  annotated.mp4 를 보면서 기록
8. 보고서 갱신     reports/cctv_baseline.md
```

7번을 건너뛰지 않는다. 숫자만으로는 왜 틀렸는지 모른다. 그리고 **실제로 본
실패만 보고서에 적는다.** 예상되는 실패는 예상이라고 적는다.
