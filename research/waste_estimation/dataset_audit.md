# 잔반 데이터 상태 점검

점검 시점: 2026-08-28
점검 방식: `research/waste_estimation/scripts/audit_dataset.py` (읽기 전용)

> **결론부터. 학습에 쓸 수 있는 실제 표본은 0건이다.**
>
> 문서·스키마·수집 도구·평가 규약은 전부 있다. **없는 것은 데이터다.**
>
> **정확히 말하면 막힌 것은 in-domain 학습이다.** 밥플래너 급식소 이미지로
> 모델을 만드는 일은 시작할 수 없다. 조건을 확인한 공개 데이터로 하는 외부
> 벤치마크는 가능하고, W2에서 LeFood-Set으로 수행했다
> (`reports/waste_w2_external_benchmark.md`).

---

## 1. 저장소가 가리키는 위치

`data/README.md`가 정한 배치는 이렇다.

```
data/
├─ README.md          ← 저장소에 있는 유일한 것
├─ private/           현장 수집. 커밋 금지
├─ raw/               외부에서 받은 원본. 커밋 금지
└─ processed/         정리한 데이터셋. 커밋 금지
```

**실제로 존재하는 것은 `data/README.md` 하나뿐이다.** `private/`, `raw/`,
`processed/` 는 만들어진 적이 없다.

사용자가 지정하지 않은 개인 폴더를 뒤지지 않았다. 저장소가 명시한 경로와
기존 도구가 가리키는 위치만 확인했다.

```
$ python research/waste_estimation/scripts/audit_dataset.py \
    --manifest data/processed/pilot-2026-01/manifest.csv \
    --images   data/processed/pilot-2026-01/images

매니페스트가 없습니다: data/processed/pilot-2026-01/manifest.csv
표본 0건. 학습을 시작할 수 없습니다.

이미지  data/processed/pilot-2026-01/images
  디렉터리가 없습니다. 이미지 0장.
```

## 2. 데이터 상태 표

| 항목 | 파일럿 수집 데이터 | 저장소 합성 fixture |
|---|---:|---:|
| Dataset Source | `data/processed/` (없음) | `tests/fixtures/dataset/` |
| Number of Images | **0** | 16 |
| Number of Valid Labels | **0** | 13 |
| Ground Truth Type | — | 합성값 (측정 아님) |
| Before Image | **없음** | 없음 |
| After Image | **없음** | 있음(합성) |
| Served Mass | **0건** | 13건 (합성) |
| Leftover Mass | **0건** | 13건 (합성) |
| Menu Metadata | — | `menu_id` 있음 |
| License/Consent Status | 수집 전 | 저장소 자체 생성 |
| **Usable for Training?** | **아니오** | **아니오** |

합성 fixture는 Pillow로 그린 단색 이미지다. `data/README.md`가 "**학습
데이터가 아니다**"라고 못박아 두었고, 이 점검도 같은 판정이다. 도구가 도는지
확인하는 용도이고, 여기서 나온 어떤 숫자도 모델 성능이 아니다.

## 3. 파일럿은 수행되지 않았다

`docs/pilot0_waste_collection_report.md` 첫 줄이 그 상태를 그대로 적고 있다.

> **상태: 아직 수행하지 않음.**
> 이 문서는 파일럿을 마친 뒤 채우는 양식이다. 아래 표에 숫자가 들어 있지
> 않다면 수집이 이루어지지 않은 것이다.

표가 전부 비어 있다. 수행 조건(장소 허가, 참여자 동의, 기관 절차, 원본 보관
장소, 보유·삭제 정책, 참가자 번호 대응표) 여섯 칸도 비어 있다.

**하나라도 끝나지 않으면 실제 참가자 데이터를 수집하지 않는다**는 것이 그
문서의 규칙이다. 즉 지금은 수집 이전 단계다.

## 4. before/after 짝

**없다. 그리고 설계상 앞으로도 없을 가능성이 높다.**

`docs/waste_ground_truth_protocol.md`의 측정 절차는 사진을 한 장만 찍는다.

```
3. 배식 후 가식부 중량을 잰다 → served_edible_mass_g   (사진 없음)
5. 식사 직후 사진을 찍는다                              (사진 여기 한 장)
6. 남은 가식부 중량을 잰다   → leftover_edible_mass_g
```

매니페스트에도 `image_path` 열이 하나뿐이다. 즉 **분모(제공량)는 저울로
재지만 사진으로는 보이지 않는다.** 이것이 이 과제의 정보 한계이고,
`reports/waste_w1_problem_definition.md` 5절에서 다룬다.

식전 사진을 함께 찍도록 절차를 바꾸는 것은 가능하지만, 그것은 수집 설계
변경이고 이 단계에서 결정할 일이 아니다. **현장에서 한 장 더 찍는 부담과
모델이 얻는 정보를 저울질해야 한다.**

## 5. 그룹 정보

분할 단위로 쓸 수 있는 열이 이미 스키마에 있다.

| 열 | 필수 | 분할 단위로서의 강도 |
|---|---|---|
| `participant_group_id` | 선택 | 가장 강함 (같은 사람의 습관·식판이 묶인다) |
| `capture_session_id` | **필수** | 차선 (같은 회차의 조명·배경이 묶인다) |
| `service_date` | 선택 | 날짜 단위 |
| `menu_id` | 필수 | 메뉴 단위(일반화 축으로는 별개) |

서비스 코드가 이미 같은 규칙을 쓴다 —
`src/babplanner_ai/dataset/validation.py::group_key_of` 는 참가자 키가 있으면
그것을, 없으면 촬영 회차를 쓴다. 연구 쪽도 같은 규칙을 따른다
(`waste_estimation/splits.py`).

**없는 그룹 정보를 발명하지 않는다.** `participant_group_id`가 비어 있으면
회차로 묶고, 그 사실을 결과에 같이 적는다.

## 6. 중복 점검

`audit_images`가 SHA-256 정확 중복을 센다. 저장소 fixture 16장에는 중복이
없다.

지각적 중복(비슷하지만 다른 파일)은 다루지 않는다. 실제 데이터를 보기 전에
임계값을 정할 수 없고, 정확 중복만으로도 같은 사진이 학습과 평가에 함께
들어가는 가장 흔한 경우는 잡힌다.

## 7. 다음 점검 시점

파일럿 수집이 한 번 돌면 이 문서를 다시 채운다. 그때 채워야 하는 칸:

```
  실제 이미지 수
  MEASURED / STANDARD_PORTION / VISUAL_ESTIMATE 별 표본 수
  APPROVED / REJECTED 비율과 거절 사유 분포
  잔반률 분포 (min / median / mean / max / 히스토그램)
  그룹 수와 가장 큰 그룹의 크기
  반복 계량의 편차 (측정 자체의 오차 하한)
```

마지막 항목이 특히 중요하다. **같은 끼니를 두 번 재도 값이 다르다면, 그
차이보다 작은 MAE는 의미가 없다.** 모델의 목표선을 정하려면 그 값을 먼저
알아야 한다(`docs/waste_evaluation_protocol.md` 7절).
