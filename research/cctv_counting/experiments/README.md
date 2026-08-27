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
