# 잔반 파일럿 현장 운영 안내

현장에서 이 문서만 보고 하루를 돌릴 수 있게 썼다. 잔반률을 왜 그렇게
재는지는 [측정 절차](waste_ground_truth_protocol.md)에, 파일이 어떤 모양인지는
[데이터셋 명세](waste_dataset_spec.md)에 있다.

한 줄 요약: **받은 음식 무게와 남긴 음식 무게를 재고, 그 사이에 사진을 한 장
찍는다.**

---

## 0. 하루 전에

- [ ] 노트북에 도구가 설치돼 있는지 확인 (아래 "설치")
- [ ] 전자저울 준비. 눈금이 몇 g인지 확인 (0.1g인지 1g인지)
- [ ] 저울 영점 확인
- [ ] 사진 찍을 기기 준비. 배터리와 저장 공간
- [ ] 참가자 동의 받기
- [ ] 촬영 자리 정하기 — 조명이 일정하고 다른 식판이 화면에 안 들어오는 곳

### 설치

```bash
cd babplanner_ai
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

아래 명령은 모두 `babplanner_ai` 폴더에서 실행한다.

---

## 1. 수집 회차 만들기 (파일럿 시작 때 한 번)

```bash
python -m babplanner_ai.collection init-session ^
    --root data/private ^
    --dataset-version pilot-2026-09 ^
    --scale-resolution-g 1.0 ^
    --site-code SITE-A
```

`--scale-resolution-g`에는 **저울 눈금**을 적는다. 0.1g 단위 저울이면 `0.1`.
이 값이 나중에 "모델이 이만큼 틀린다"를 판단할 때의 바닥이 된다.

출력된 `cs_...` 를 적어 둔다. 이후 모든 명령에 필요하다.

> 화면에 나오는 `cs_...`, `p_...`, `cap_...`, `tp_...` 는 전부 임의로 만든
> 값이다. 이름이나 학번을 대신 쓰면 안 된다.

---

## 2. 식기 무게 재기 (식기 조합마다 한 번)

빈 식판과 그릇만 저울에 올려 잰다. **세 번 재서 세 번 다 적는다.**

```bash
python -m babplanner_ai.collection tare-add ^
    --root data/private --session cs_... ^
    --description "4칸 스테인리스 식판 + 국그릇" ^
    --weight 400.0 --weight 400.5 --weight 399.5
```

출력된 `tp_...` 를 적어 둔다.

편차가 크게 나오면(예: 5g 이상) 저울이나 식기를 다시 본다. 이 값의 오차는
모든 표본에 그대로 옮겨 간다.

식기 구성이 바뀌면 **새 프로파일을 만든다.** 같은 것을 계속 쓰지 않는다.

---

## 3. 참가자 번호 주기

참가자마다 한 번:

```bash
python -m babplanner_ai.collection new-participant
```

출력된 `p_...` 를 그 사람의 종이에 적는다.

**이 번호와 사람을 잇는 표는 이 프로그램 밖에서, 잠글 수 있는 곳에 둔다.**
파일럿이 끝나면 그 표를 어떻게 할지는 기관 개인정보 정책에 따른다.

---

## 4. 끼니 번호 주기 (한 끼마다 한 번)

```bash
python -m babplanner_ai.collection new-capture-session
```

출력된 `cap_...` 를 그 끼니 동안 쓴다. 점심 한 번이면 하나면 된다.

이 번호로 같은 자리·같은 시간의 사진들을 묶는다. 나중에 학습과 평가를 나눌
때 이 묶음이 갈라지지 않게 하려는 것이다.

---

## 5. 배식 직후 — 무게 재기

1. 배식받은 식판을 통째로 저울에 올린다 (**식기 포함**)
2. 숫자를 적는다 → `배식 후 중량`

식기를 빼고 계산하는 것은 프로그램이 한다. 사람이 빼서 적지 않는다.

---

## 6. 식사 후 — 사진 찍기

- 식판 전체가 화면에 들어오게
- 가능하면 위에서 내려다보게
- 같은 거리에서
- 그림자가 지지 않게
- 다른 식판이 겹치지 않게

**화면 안에 다음이 들어오면 다시 찍는다.**

- 얼굴
- 학생증, 명찰
- 문서
- 다른 사람

프로그램이 사진의 위치·기기 정보는 지우지만, **화면에 보이는 것은 지우지
못한다.** 여기가 1차 방어다.

---

## 7. 식사 후 — 무게 재기

1. 남은 음식이 있는 식판을 통째로 저울에 올린다
2. 숫자를 적는다 → `식사 후 중량`

---

## 8. 표본 등록

```bash
python -m babplanner_ai.collection add-sample ^
    --root data/private --session cs_... ^
    --image "C:\photos\IMG_1234.jpg" ^
    --capture-session cap_... ^
    --participant p_... ^
    --menu-id 3 ^
    --tare-profile tp_... ^
    --pre-weight 900.0 ^
    --post-weight 500.0 ^
    --mass-source MEASURED ^
    --meal-type LUNCH ^
    --service-date 2026-09-15
```

바로 잔반률이 나온다.

```
표본을 기록했습니다: 396ee669...
  제공량   500 g
  잔반량   100 g
  잔반률   0.2000
```

**이 자리에서 숫자를 확인한다.** 눈으로 본 식판과 비슷한지. 다르면 지금
확인하는 편이 낫다. 나중에 알게 되면 그 사람은 이미 가고 없다.

### 오류가 나면

| 메시지 | 뜻 |
|---|---|
| `식기 무게보다 크지 않습니다` | tare 프로파일이 다르거나 저울을 잘못 읽음 |
| `식기 무게보다 작습니다` | 다른 식기를 썼거나 저울을 잘못 읽음 |
| `잔반량이 제공량보다 많습니다` | 두 중량이 뒤바뀌었을 수 있음 |
| `같은 이미지가 이미 있습니다` | 같은 사진을 두 번 등록하려 함 |

숫자를 고쳐서 억지로 통과시키지 않는다. 무엇이 잘못됐는지 찾는다.

### 사진을 쓸 수 없을 때

흔들렸거나, 식판이 안 보이거나, 엉뚱한 것을 찍었으면:

```bash
python -m babplanner_ai.collection add-sample ^
    --root data/private --session cs_... ^
    --image "C:\photos\IMG_1235.jpg" ^
    --capture-session cap_... --participant p_... --menu-id 3 ^
    --label REJECTED --rejection-reason SEVERE_BLUR
```

사유는 `NO_RELEVANT_FOOD` / `SEVERE_BLUR` / `SEVERE_OCCLUSION` /
`INVALID_VIEW` / `OTHER` 중 하나.

---

## 9. 잘못 적었을 때

지우지 않는다. 고친 기록을 새로 붙인다.

```bash
python -m babplanner_ai.collection correct-sample ^
    --root data/private --session cs_... ^
    --sample-id 396ee669... ^
    --post-weight 450.0 ^
    --note "저울 숫자를 잘못 읽음"
```

원래 값은 기록에 그대로 남는다. 나중에 "그때 무슨 일이 있었나"를 알 수 있어야
하기 때문이다.

---

## 10. 하루를 마치며

```bash
python -m babplanner_ai.collection qc --root data/private --session cs_...
```

보는 것:

- **측정 오류** — 0이어야 한다. 있으면 그 표본을 고친다
- **중복 이미지** — 같은 사진이 두 번 들어갔다는 뜻
- **없는 파일** — 사본이 지워졌다
- **잔반률 분포** — 0 근처에만 몰려 있으면 배식량이나 참가자 구성을 다시 본다
- **tare 편차** — 저울이 흔들리고 있다는 신호
- **참가자 그룹 / 촬영 회차** — 나중에 학습·평가를 나누려면 셋 이상 필요하다

숫자만 나온다. 합격 여부를 프로그램이 말하지 않는다. 무엇이 정상인지는 실제
데이터를 보고 사람이 정한다.

---

## 11. 파일럿을 마치며

```bash
# 학습용 매니페스트 만들기
python -m babplanner_ai.collection export --root data/private --session cs_...

# 검증
python -m babplanner_ai.dataset validate ^
    --manifest data/private/sessions/cs_.../exports/manifest.csv ^
    --root data/private/sessions/cs_...

# 회차 닫기
python -m babplanner_ai.collection close-session --root data/private --session cs_...
```

닫은 회차에는 표본을 더 넣을 수 없다.

### 백업

`data/private/sessions/cs_...` 폴더 전체를 백업한다. 안에 든 것:

| | 무엇 | 저장소에 올리나 |
|---|---|---|
| `session.json` | 회차 정보 | ✗ |
| `tare_profiles.jsonl` | 식기 무게 | ✗ |
| `journal.jsonl` | 원시 기록 전부 | ✗ |
| `raw_images/` | 원본 사진 | ✗ **절대** |
| `sanitized_images/` | 비식별 사본 | ✗ |
| `exports/` | 학습용 매니페스트 | ✗ |

`data/` 전체가 Git에서 제외돼 있다. 그래도 실수로 넣지 않도록 이 표를 안다.

원본 사진을 얼마나 보관할지는 기관 개인정보 정책에 따라 따로 정한다. 이
도구가 정하지 않는다.

---

## 12. 파일럿을 마친 뒤

[Pilot-0 보고서 양식](pilot0_waste_collection_report.md)을 채운다. 집계값만
적고 참가자 번호 목록이나 실제 사진은 넣지 않는다.

마지막에 Data Gate를 판단한다. 여기서 "모델 단계로 넘어가도 되는가"가
정해진다.

---

## 자주 나오는 질문

**Q. 식기를 빼고 음식만 저울에 올리면 안 되나요?**
그래도 되지만 매번 옮겨 담아야 하고 그 과정에서 음식이 흘린다. 식판째 올리고
식기 무게를 빼는 편이 실수가 적다.

**Q. 표준 배식량을 쓰면 안 되나요?**
쓸 수 있다(`--mass-source STANDARD_PORTION --standard-served 500`). 다만 그
표본은 주 평가 집합에 들어가지 않는다. 사람마다 실제 배식량이 다르기
때문이다.

**Q. 사진만 보고 잔반률을 적으면 안 되나요?**
안 된다. 이 도구는 그 값을 받지 않는다. 사람의 눈대중은 정답이 아니라 나중에
"모델이 사람보다 나은가"를 비교할 때 쓸 별도 자료다.

**Q. 잔반률이 0.05 정도로 나왔는데 반올림해서 0으로 적어도 되나요?**
안 된다. 잰 값을 그대로 둔다. 프로그램이 계산한다.
