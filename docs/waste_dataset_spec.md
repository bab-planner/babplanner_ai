# 잔반 데이터셋 명세

정답의 뜻은 [측정 절차 문서](waste_ground_truth_protocol.md)에 있다. 이
문서는 그 값을 파일로 어떻게 적고 어떻게 나누는지를 정한다.

---

## 1. 구성

```
<dataset-root>/
├─ metadata.json      # 데이터셋 전체에 한 번
├─ manifest.csv       # 표본 한 건당 한 줄
└─ images/
   └─ <sample_id>.png
```

CSV를 쓴다. 사람이 열어 볼 수 있고, 변경 이력이 줄 단위로 남고, 새
의존성이 필요 없다.

---

## 2. 매니페스트 열

| 열 | 필수 | 설명 |
|---|---|---|
| `sample_id` | ✓ | 비식별 식별자. 이미지 파일명도 이 값을 쓴다 |
| `image_path` | ✓ | 이미지 루트로부터의 상대 경로 |
| `menu_id` | ✓ | 서비스의 메뉴 id |
| `site_id` | | 급식소 |
| `service_date` | | 배식일 |
| `meal_type` | | `BREAKFAST` / `LUNCH` / `DINNER` |
| `captured_at` | | 촬영 시각 |
| `participant_group_id` | | 연구용 익명 참가자 키 |
| `capture_session_id` | ✓ | 촬영 회차 |
| `served_edible_mass_g` | 승인 시 ✓ | 제공 가식부 중량(g) |
| `leftover_edible_mass_g` | 승인 시 ✓ | 잔반 가식부 중량(g) |
| `waste_rate` | 승인 시 ✓ | 위 두 값의 비 |
| `served_mass_source` | 승인 시 ✓ | `MEASURED` / `STANDARD_PORTION` / `VISUAL_ESTIMATE` |
| `gt_exclusion_reason` | | `BONE_IN` / `EVAPORATIVE` / `OTHER`. 비어 있는 것이 정상 |
| `gt_exclusion_note` | `OTHER` 시 ✓ | 왜 뺐는지 |
| `analysis_label` | ✓ | `APPROVED` / `REJECTED` |
| `rejection_reason` | 거절 시 ✓ | 아래 참고 |
| `image_sha256` | | 파일 해시. 검증기가 대조한다 |
| `split` | | `train` / `val` / `test` |
| `label_version` | ✓ | 정답 정의의 버전 |

정의되지 않은 열은 거부한다. 오타 난 열 이름이 조용히 무시되면 그 데이터가
어디로 갔는지 알 수 없다.

### 거절 사유

`NO_RELEVANT_FOOD` / `SEVERE_BLUR` / `SEVERE_OCCLUSION` / `INVALID_VIEW` /
`OTHER`

실제 데이터를 보기 전이라 굵게만 나눴다. 어디에 몰리는지 보고 나서 쪼갠다.

`REJECTED`인데 사유가 없으면 **오류**다. 왜 버렸는지 되짚을 수 없는 표본은
데이터셋을 고치는 데 쓸 수 없다.

### 정답 제외 (`gt_exclusion_reason`)

거절과 다르다. 거절은 사진에서 잔반을 판단할 수 없다는 뜻이고, 정답 제외는
**사진도 계량도 멀쩡하지만 질량비가 그 식판의 잔반률을 뜻하지 않는다**는
뜻이다. 뜻과 현장 규칙은 `waste_ground_truth_protocol.md` §2-2에 있다.

값이 있는 행은 매니페스트에 그대로 남지만 다음에서 빠진다.

- `split`이 붙지 않는다. 붙어 있으면 검증기가 **오류**로 잡는다
- `is_primary_benchmark` / `is_supervised_usable`이 거짓이 된다
- 표본 수·범주 수·고잔반 수 등 **관문 계산에 들어가지 않는다**

`OTHER`인데 `gt_exclusion_note`가 비어 있으면 **오류**다. 사유만 남고 이유를
모르는 표본이 쌓이면 되살릴 수도 규칙을 고칠 수도 없다. 반대로 설명만 있고
사유가 없어도 오류다.

도구는 **전체 표본 수와 정답 사용 가능 표본 수를 항상 따로** 보고한다. 뼈
메뉴가 많아 실제로 학습할 데이터가 모자란 상황이 총계 뒤에 숨지 않게 하려는
것이다.

옛 매니페스트에는 이 두 열이 없다. 없으면 비어 있는 것으로 읽는다.

---

## 3. 세 가지 버전

혼동하기 쉬워 이름을 분리한다.

| 이름 | 무엇의 버전 | 예 |
|---|---|---|
| `label_version` | 정답 정의 | `waste-mass-ratio-v1` |
| `dataset_version` | 수집한 데이터 묶음 | `pilot-2026-01` |
| `model_version` | 학습된 추론 모델 | 아직 없음 |

정답 정의가 바뀌면 `label_version`이 바뀐다. 한 데이터셋 안에 여러
`label_version`이 섞이면 검증기가 오류로 잡는다. 정의가 다른 정답을 한데
모아 계산한 평가 숫자는 의미가 없다.

현재 서비스가 쓰는 `fixture-waste-v1` 계열 값은 개발용 fixture 식별자이지
학습된 모델의 버전이 아니다.

### `protocol_version`은 또 다른 축이다

메타데이터에만 있고 행에는 없다. **수집 절차**의 버전이지 정답 정의의
버전이 아니다.

| | `label_version` | `protocol_version` |
|---|---|---|
| 무엇이 바뀌면 올리나 | 정답의 **산술식** | 현장 **수집 규칙** |
| 섞이면 | 검증기가 오류로 잡는다 | 섞여도 된다. 기록만 남는다 |

`pilot-v2`에서 바뀐 것: 비가식물·식기 구성 규칙(§2-1)과 메뉴 단위 정답
제외(§2-2)가 들어왔다. **`waste_rate = leftover / served`는 그대로이므로
`label_version`은 `waste-mass-ratio-v1`을 유지한다.** 산술이 그대로인데
정답 정의 버전을 올리면, 옛 데이터와 새 데이터를 같이 쓸 수 없게 되어 아무
이득 없이 데이터가 갈라진다.

---

## 4. 데이터셋 메타데이터

`metadata.json`:

```json
{
  "dataset_version": "pilot-2026-01",
  "label_version": "waste-mass-ratio-v1",
  "created_at": "2026-01-15T00:00:00Z",
  "collection_site": null,
  "scale_resolution_g": 1.0,
  "protocol_version": "pilot-v2",
  "notes": null
}
```

행마다 반복할 필요 없는 수집 조건이 여기 온다. 저울 눈금은 정답의 정밀도
한계를 정하는 값이라 반드시 남긴다. 개인정보는 넣지 않는다.

---

## 5. 개인정보

### 열 이름 차단

다음 이름의 열이 있으면 매니페스트를 읽지 않는다 (대소문자 무시).

`name`, `phone`, `phone_number`, `student_id`, `email`, `user_id`,
`access_token`, `refresh_token`, `verification_token`, `password`

값까지 완전히 탐지하려 하지 않는다. 개인정보가 새는 가장 흔한 경로는 "일단
넣어 둔 열"이라, 이름 단계에서 막는 편이 확실하다.

### 식별자

`sample_id`는 UUID 등 임의 값이다. 이름·학번·전화번호·앱 사용자 id를 넣지
않는다. 이미지 파일명도 같은 값을 쓴다. 촬영 기기가 붙인 원본 파일명을 그대로
두면 그 안에 무엇이 들어 있는지 통제할 수 없다.

`participant_group_id`는 분할 누수를 막기 위한 연구용 익명 키다. 전화번호
해시를 그대로 쓰지 않는다. 앱의 사용자 id와 같을 필요도 없다.

### confidence

매니페스트에 `confidence`나 `confidence_target` 열을 두지 않는다. 검증기가
거부한다.

`confidence`는 사람이 붙이는 정답이 아니다. 모델이 자기 추론의 불확실성에서
내놓아야 하는 값이다. 사람이 0.9, 0.7을 임의로 적어 학습 target으로 삼으면
모델은 그 사람의 버릇을 배운다.

---

## 6. 중복

같은 이미지가 여러 `sample_id`로 들어오면 오류다. SHA-256으로 잡는다.
중복이 있으면 분할이 새고 평가 숫자가 부풀려진다.

지각적 해시(perceptual hash)는 이번 단계에 넣지 않는다. 거의 같은 사진을
잡으려면 새 의존성이 필요한데, 실제 데이터에서 그런 중복이 얼마나 나오는지
아직 모른다.

---

## 7. 분할

행을 무작위로 나누지 않는다.

같은 사람이 같은 회차에 찍은 사진들은 배경도 식판도 습관도 닮아 있다.
그것들이 학습과 평가에 나뉘어 들어가면 모델이 사람을 외운 것을 일반화로
착각하게 된다.

### 묶는 기준

1. `participant_group_id`가 있으면 참가자 단위
2. 없으면 `capture_session_id` 단위

한 그룹은 정확히 한 분할에만 들어간다. 검증기가 확인한다.

### 두 가지 평가 관점

| | 무엇을 보는가 | 그룹 기준 |
|---|---|---|
| A. 일반 | 새로운 사람·상황에서의 성능 | `auto` (참가자 → 회차) |
| B. 처음 보는 메뉴 | 학습에 없던 메뉴에서의 성능 | `menu` |

B는 도구로 지원하되, 표본이 적을 때 억지로 만들지 않는다. 평가 집합이 한두
메뉴로 쏠리면 그 숫자는 메뉴 하나에 대한 평가일 뿐이다.

### 결정성

`seed`가 같고 매니페스트가 같으면 언제나 같은 분할이 나온다. 입력 줄 순서를
바꿔도 결과는 같다. 그러지 않으면 어제 숫자와 오늘 숫자를 비교할 수 없다.

비율은 인자로 받는다. 기본값 `0.8 / 0.1 / 0.1`은 출발점이지 정책이 아니다.

그룹이 셋보다 적으면 **실패한다.** 행을 쪼개서 억지로 세 갈래를 만드는 것이
곧 누수다.

---

## 8. 도구

```bash
python -m babplanner_ai.dataset validate --manifest manifest.csv --root .
python -m babplanner_ai.dataset summarize --manifest manifest.csv
python -m babplanner_ai.dataset make-splits \
    --manifest manifest.csv --out split.csv --seed 42
```

`validate`가 보는 것: 스키마 / 값 범위 / 질량비 일치 / 파일 존재 / 이미지
디코딩 / SHA-256 대조 / 중복 / 라벨 불변식 / 개인정보 열 / 정답 정의 버전
일관성 / 분할 누수.

오류가 있으면 0이 아닌 값으로 끝난다.

`summarize`는 개수만 센다. 실제 식별값은 출력하지 않는다.

---

## 9. 아직 정하지 않은 것

- 실제 데이터셋 규모
- 수집 기관과 기간
- 촬영 기기·조건의 범위
- 지각적 중복의 처리 기준
- 반복 측정의 허용 오차
- 반찬별 라벨의 도입 여부
