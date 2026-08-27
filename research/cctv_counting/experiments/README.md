# 실행 결과

여기에 쌓이는 것은 커밋하지 않는다(`.gitignore`). 이 README만 tracked다.

실행 하나가 폴더 하나가 된다.

```
experiments/
  2026-08-28_baseline_tracking_conf040/
    annotated.mp4
    observations.csv
    summary.json
    metadata.json
```

`metadata.json`이 재현을 담당한다. 여기에 설정 전체, detector 이름과 weight,
장치, 패키지 버전, 실행 시각, seed가 들어간다. 이 파일 없이 남은 숫자는
어디서 나온 것인지 알 수 없으므로 결과가 아니다.

폴더 이름에 날짜와 설정 이름을 넣어 두면 나중에 찾기 쉽다. 강제하지는 않는다.

## a2_local_demo

Stage A2의 첫 실제 추론 결과. 작은 기록 파일만 커밋되어 있다.

```
a2_local_demo/
  config_detection.yaml    Baseline A 설정 (재현용 사본)
  config_tracking.yaml     Baseline B 설정 (재현용 사본)
  metrics_detection.json   A 채점 결과
  metrics_tracking.json    B 채점 결과
  metrics_conf03.json      임계값 0.3
  metrics_conf07.json      임계값 0.7
  metrics_fpsfix.json      max_age/min_hits 진단
```

`annotated.mp4`, `observations.csv`, `summary.json`, `metadata.json`은
`outputs/`에 남고 커밋하지 않는다. 재현하려면:

```
.venv-cctv\Scripts\python.exe research\cctv_counting\scripts\run_video.py ^
  --input <vtest.avi 경로> ^
  --config research\cctv_counting\configs\a2_demo_tracking.yaml ^
  --output-dir outputs\a2_tracking

.venv-cctv\Scripts\python.exe research\cctv_counting\scripts\a2_evaluate.py ^
  --observations outputs\a2_tracking\observations.csv ^
  --occupancy-gt research\cctv_counting\evaluation\a2_occupancy_ground_truth.csv ^
  --crossing-gt research\cctv_counting\evaluation\a2_crossing_ground_truth.csv ^
  --video-id a2-demo-vtest --end-seconds 29
```

영상은 `opencv/opencv` 저장소의 `samples/data/vtest.avi`다(Apache-2.0).
`datasets/raw/`에 두면 되고 그 경로는 `.gitignore`에 있다.
