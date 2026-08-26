# 데이터

**이 디렉터리에 실제 이미지나 매니페스트를 커밋하지 않는다.**

급식 사진에는 사람이 찍혀 있을 수 있고, 한번 공개 저장소에 올라간 것은
지웠다고 사라지지 않는다. `.gitignore`가 막고 있지만 강제로 넣을 수 있으니
먼저 이 규칙을 안다.

## 배치

```
data/
├─ README.md          # 이 파일. 저장소에 있는 유일한 것
├─ private/           # 현장 수집. 커밋 금지
│  └─ sessions/<cs_...>/
│     ├─ session.json          # 회차 정보
│     ├─ tare_profiles.jsonl   # 식기 무게
│     ├─ journal.jsonl         # 원시 기록(덧붙이기만)
│     ├─ raw_images/           # 원본 사진. 절대 커밋 금지
│     ├─ sanitized_images/     # EXIF 제거 사본
│     └─ exports/              # 학습용 매니페스트
├─ raw/               # 외부에서 받은 원본. 커밋 금지
└─ processed/         # 정리한 데이터셋. 커밋 금지
```

현장 수집은 도구가 이 구조를 만들어 준다. 손으로 만들지 않는다. 절차는
[현장 운영 안내](../docs/waste_pilot_collection_runbook.md).

각 데이터셋은 이런 모양이다.

```
data/processed/pilot-2026-01/
├─ metadata.json
├─ manifest.csv
└─ images/
   └─ <sample_id>.png
```

## 매니페스트

열과 뜻은 [데이터셋 명세](../docs/waste_dataset_spec.md)에 있다. 정답을
어떻게 재는지는 [측정 절차](../docs/waste_ground_truth_protocol.md)에 있다.

한 줄 예시:

```csv
sample_id,image_path,menu_id,capture_session_id,served_edible_mass_g,leftover_edible_mass_g,waste_rate,served_mass_source,analysis_label,label_version
0fc2...,images/0fc2....png,3,S01,500.0,100.0,0.2,MEASURED,APPROVED,waste-mass-ratio-v1
```

## 도구

### 현장 수집

```bash
python -m babplanner_ai.collection init-session --root data/private ...
python -m babplanner_ai.collection add-sample   --root data/private ...
python -m babplanner_ai.collection qc           --root data/private ...
python -m babplanner_ai.collection export       --root data/private ...
```

### 데이터셋 점검

```bash
# 점검 — 문제가 있으면 0이 아닌 값으로 끝난다
python -m babplanner_ai.dataset validate \
    --manifest data/processed/pilot-2026-01/manifest.csv \
    --root data/processed/pilot-2026-01

# 요약
python -m babplanner_ai.dataset summarize \
    --manifest data/processed/pilot-2026-01/manifest.csv

# 분할
python -m babplanner_ai.dataset make-splits \
    --manifest data/processed/pilot-2026-01/manifest.csv \
    --out data/processed/pilot-2026-01/manifest.split.csv \
    --seed 42
```

## 개인정보

- 파일명은 `sample_id`. 이름·학번·전화번호를 넣지 않는다.
- 매니페스트에 개인정보 열을 두지 않는다. 검증기가 이름으로 거른다.
- 학습용 사본에서 EXIF의 위치 정보를 지운다.
- 얼굴·학생증·명찰·문서가 찍힌 사진은 쓰지 않는다.

## 저장소에 있는 것

`tests/fixtures/dataset/`의 합성 데이터뿐이다. Pillow로 그린 단색 이미지
16장이며 도구가 도는지 확인하는 용도다. **학습 데이터가 아니다.**
