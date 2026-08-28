# AI-1. CCTV 기반 비식별 식당 혼잡도·이용 인원 분석

*CCTV-based Anonymous Occupancy and Entry Counting*

---

## 1. 문제 정의

급식소 운영에 두 가지 숫자가 필요하다.

**지금 붐비는가.** 앱에 혼잡도를 띄우려면 현재 그 공간에 몇 명이 있는지
알아야 한다.

**오늘 몇 명이 먹었는가.** 식수 예측(AI-3)을 학습하려면 정답이 필요한데,
그 정답이 지금 시스템 어디에도 쌓이지 않는다. 입구를 통과한 사람 수는 그
정답의 대리 지표가 될 수 있다.

두 숫자는 서로 다르고, **하나로 합치면 둘 다 틀린다.**

### Occupancy vs Entry Count

| | Occupancy | Entry Count |
|---|---|---|
| 세는 것 | 지금 그 공간에 있는 사람 | 입구 선을 IN 방향으로 넘은 사람 |
| 시간 축 | 순간값 | 누적값 |
| 쓰는 곳 | 실시간 혼잡도 | 한 끼 이용 인원의 대리 지표 |
| 값의 성질 | 오르내린다 | 단조 증가 |
| 필요한 것 | 프레임 하나 | 프레임 사이의 연결(추적) |

점심 시간에 300명이 다녀가도 한 순간의 occupancy는 40명일 수 있다. 40이
혼잡도이고 300이 이용 인원이다. 하나의 "인원 수" 열에 이 둘을 담으려는
순간부터 그 열은 아무것도 뜻하지 않게 된다.

이 저장소는 두 값을 별도 필드로 내보내고, 추적이 없을 때 entry는 0이 아니라
**빈 값**으로 낸다. 0은 "아무도 안 들어왔다"이고 빈 값은 "재지 않았다"이다.

### `naiveLineBandDetections`는 입장 인원이 아니다

detection-only baseline에서 같이 기록하는 값이 하나 있다. 이름 그대로
**naive line-band detection proxy** — 선 근처 불감대 안에 상자가 몇 번
보였는지의 누적이다.

이것을 "detection-only 입장 인원"이라고 부르면 안 된다.

- 같은 사람이 선을 지나는 동안 매 프레임 다시 세인다.
- 그래서 실제 입장 인원보다 훨씬 크게 나온다.
- 그 배율은 fps와 걷는 속도에 따라 달라진다. 정규화된 값이 아니다.

**쓰라고 두는 것이 아니라 대조하려고 둔다.** 이 값이 실제 입장 인원보다
얼마나 부풀려지는지가 곧 추적이 하는 일의 크기다. 그것을 숫자로 보여 주는
것이 이 필드의 유일한 목적이다.

## 2. 왜 식별이 아니라 계수인가

기술적으로는 사람을 알아보는 쪽이 계수에 유리하다. 같은 사람을 다시 알아볼
수 있으면 재입장을 정확히 처리하고, 가려졌다 나타난 사람도 놓치지 않는다.

그런데 이것은 학생이 밥 먹는 장면을 찍는 카메라다.

- 필요한 것은 "몇 명"이지 "누가"가 아니다. 신원은 목적에 기여하지 않는다.
- 얼굴 특징을 한 번 뽑아 두면, 그것을 다른 데 쓰지 않겠다는 약속만 남는다.
  약속은 코드 리뷰로 강제되지 않는다.
- 회원 정보와 연결되는 순간 이것은 출입 감시 시스템이 된다. 그것은 학교가
  동의한 것이 아니다.

그래서 **애초에 신원 정보를 만들지 않는다.** 만들지 않은 것은 유출되지 않고,
잘못 쓰이지 않고, 요구받아도 내줄 것이 없다.

`tests/test_cctv_privacy.py`가 얼굴 인식 라이브러리와 신원 필드가 코드에
들어오는지 매 실행마다 확인한다. 문서에만 적힌 약속은 조용히 깨진다.

## 3. 구조

```mermaid
flowchart TD
    V[Video<br/>파일 또는 스트림] --> D[Shared Person Detector<br/>torchvision COCO pretrained<br/>프레임당 한 번]
    D --> FO[high-confidence filtering<br/>conf >= T_occ, ROI]
    D --> FE[high + low detections<br/>conf >= T_low, ROI]
    FO --> O1[Occupancy Count<br/>필터 통과 상자 수]
    FE --> T[Multi Object Tracker<br/>기본 ByteTrack-inspired greedy<br/>선택 MotionAwareTracker]
    T --> C[Line Crossing<br/>상태 기계]
    C --> E[Entry / Exit Count]
    O1 --> R[Reliability Signal<br/>mean/min confidence, flag]
    E --> R
    R --> OUT[observations.csv<br/>summary.json<br/>annotated.mp4]
    OUT -.threshold 정책 필요.-> S[Spring<br/>/internal/v1/congestion-observations]
```

**두 갈래다(dual-path).** A4에서 서로 다른 영상 13편으로 검증한 구조다.

```
  occupancy   detector + 임계값 + ROI       추적을 거치지 않는다
  entry       추적 + 선 통과                 추적이 있어야 정의된다
```

**A6에서 두 갈래의 임계값을 갈랐다.** 추론은 여전히 프레임당 한 번이고,
갈라지는 것은 필터 정책뿐이다. 두 과제가 서로 다른 오차를 싫어하기 때문이다.

```
  occupancy   오탐 하나가 그 프레임 인원을 그대로 부풀린다.
              놓친 사람은 다음 프레임에 다시 셀 기회가 있다.   -> 높은 임계값

  entry       궤적이 한 번 끊기면 track이 쪼개지고, 쪼개진 track은
              선을 넘은 것을 모른다. 되찾을 기회가 없다.        -> 낮은 임계값까지 받는다
```

`dualpath.py`가 이 규칙을 코드로 못박는다 — `detector_calls`가 프레임 수와
같아야 한다는 것이 테스트로 걸려 있다.

**임계값을 가르는 이득의 크기는 장면에 달렸다.** DEV(쇼핑센터 front)에서는
통과 계수 오차가 0.30에서 0.20으로 줄었지만, FINAL(corridor)에서는 변화가
없었다. 사람이 크게 보이는 장면에서는 낮은 신뢰도 상자를 열 이유가 적다.
근거는 [A6 보고서](../../reports/cctv_a6_trajectory_continuity.md).

**현재 occupancy 읽기 방식에서는 추적이 기여할 자리가 없다.** `update()`가
"이번 프레임에 검출과 매칭된 track"만 돌려주므로, `min_hits=1`이면
detection-only와 계수가 **구조적으로 같아진다**. 실제 비교 대상인
`min_hits=3`도 TEST 8편 macro MAE 차이가 0.008명으로 사전 마진 0.25명 안이었다.

매칭되지 않은 track을 유지해 검출 누락을 메우는 방식은 아직 시험하지 않았다.
자세한 것은 [A4 보고서](../../reports/cctv_a4_cross_video.md)와
[A5 보고서](../../reports/cctv_a5_small_person.md).

**detector는 한 번만 돌린다.** 비용의 99.9%가 detector이고 추적은 그
0.1% 수준이라, 두 경로가 같은 검출 결과를 나눠 쓰면 된다.

점선은 **아직 연결되지 않은 구간**이다. 왜 연결하지 않았는지는 12절.

모듈은 이렇게 나뉜다.

| 파일 | 하는 일 | 무거운 의존성 |
|---|---|---|
| `geometry.py` | 정규화/픽셀 좌표, 상자, 선, ROI | 없음 |
| `detection.py` | Detection, detector 경계, 필터 | 없음 |
| `tracking.py` | ByteTrack-inspired 추적 (A2 baseline) | 없음 |
| `motion_tracking.py` | 시간 기반 소멸 + 시간 비례 외삽 추적 (A6) | 없음 |
| `assignment.py` | Hungarian 전역 할당 | 없음 |
| `dualpath.py` | 공유 detector + 과제별 정책 | 없음 |
| `continuity.py` | fragmentation / gap recovery / 통과 창 | 없음 |
| `attribution.py` | 놓친 통과 사건의 원인 분류 | 없음 |
| `crossing.py` | 선 통과 상태 기계 | 없음 |
| `reliability.py` | 신뢰 신호 | 없음 |
| `observations.py` | CSV 스키마 | 없음 |
| `windows.py` | 시간 창 통계 | 없음 |
| `summary.py` | summary.json | 없음 |
| `metrics.py` | 평가 지표 | 없음 |
| `groundtruth.py` | 정답 파일 스키마 | 없음 |
| `config.py` | 실험 설정 | pydantic |
| `pipeline.py` | 위를 엮는 흐름 | 없음 |
| `video.py` | 영상 읽기/쓰기 | opencv |
| `annotate.py` | 겹쳐 그리기 | opencv |
| `detectors/torchvision_detector.py` | 실제 추론 | torch, torchvision |

**계수 로직 전체가 torch 없이 돈다.** 단위 테스트가 가짜 detector로 pipeline을
끝까지 돌리는 이유다. 모델 설치 여부와 무관하게 "선을 넘었는가"의 정확성이
검증되어야 한다.

## 4. Detector

### 선택: torchvision `fasterrcnn_mobilenet_v3_large_fpn` (COCO pretrained)

detector를 처음부터 학습하지 않는다. 급식소 라벨 데이터가 하나도 없고,
사람 검출은 공개 pretrained 모델이 이미 잘 하는 일이다.

### 라이선스는 세 층이다

이 절을 읽기 전에 구분해 둘 것이 있다. 흔히 한 덩어리로 말하지만 서로 다른
문서에서 나오는 서로 다른 조건이다.

```
  A. 라이브러리 코드 라이선스     소프트웨어를 쓰는 조건
  B. pretrained weight 조건       그 가중치를 쓰는 조건
  C. 학습 데이터셋 조건           그 가중치가 학습된 데이터의 조건
```

**A가 허용적이라고 B와 C가 따라오지 않는다.** torchvision 저장소가 직접 그렇게
말한다.

> "The pre-trained models provided in this library may have their own licenses
> or terms and conditions derived from the dataset used for training. It is
> your responsibility to determine whether you have permission to use the
> models for your use case."
>
> — pytorch/vision README (2026-08-28 확인)

같은 저장소에서 SWAG 계열 가중치는 CC-BY-NC 4.0이라고 따로 명시한다. 즉
torchvision 안에서도 가중치마다 조건이 다르다.

같은 함정이 데이터셋 쪽에도 있다. CrowdHuman의 GitHub 저장소 LICENSE는
MIT이지만, 공식 다운로드 페이지의 데이터 이용 조건은 비상업 연구·교육 전용에
재배포 금지다. **저장소 라이선스를 데이터 라이선스로 읽으면 정반대 결론이
나온다.**

### 후보 비교

| 후보 | A. 코드 | B. weight 조건 | Python 3.12 | 판단 |
|---|---|---|---|---|
| **torchvision detection** | BSD-3-Clause (공식 LICENSE 확인) | 저장소가 "가중치는 별도 조건일 수 있고 확인은 사용자 책임"이라고 명시 | 지원 | **채택** |
| Ultralytics YOLOv8 / YOLO11 | AGPL-3.0 | **가중치도 AGPL-3.0.** 공식 라이선스 페이지가 "All Ultralytics YOLO trained models fall under the AGPL-3.0 License by default"라고 명시. 상용은 Enterprise | 지원 | 제외 |
| YOLOX (Megvii) | Apache-2.0 (미확인) | 미확인 | 확인 필요 | 향후 후보 |
| RT-DETR (lyuwenyu) | Apache-2.0 (미확인) | 미확인 | 확인 필요 | 향후 후보 |
| RTMDet / MMDetection | Apache-2.0 (미확인) | 미확인 | mmcv 빌드 문제 | 보류 |

Ultralytics에 대한 기술은 **Ultralytics의 구현과 배포본**에 대한 것이다.
"YOLO 계열 전부가 AGPL"이라는 뜻이 아니다. YOLO라는 이름은 여러 조직이
서로 다른 조건으로 배포하고 있고, 여기서 확인한 것은 Ultralytics 한 곳이다.
아래 두 줄(YOLOX, RT-DETR)의 라이선스는 이번에 공식 원문을 확인하지 않았다.
쓰기로 결정하는 시점에 확인한다.

### 왜 torchvision인가

**Ultralytics를 제외한 이유가 분명했다.** 공식 라이선스 페이지가 코드뿐 아니라
학습된 모델까지 기본값으로 AGPL-3.0이라고 밝히고, 프로젝트 전체를 공개하지
않으려면 Enterprise 라이선스를 사야 한다고 명시한다. 밥플래너는 서비스로
나갈 제품이다. 연구 단계에서 그 모델로 숫자를 만들어 두면 그 숫자를 제품의
근거로 쓸 수 없다. **나중에 버리게 될 결과는 처음부터 만들지 않는다.**

**torchvision을 고른 것은 "제약이 없어서"가 아니다.** 코드가 BSD-3-Clause라
소프트웨어 층은 깨끗하고, 가중치 층에 대해 저장소가 무엇을 보장하고 무엇을
보장하지 않는지 **명시적으로 밝혀 두었기 때문**이다. 확인해야 할 것이
무엇인지 알 수 있는 쪽이, 아무 말이 없는 쪽보다 낫다.

> **배포 가능 여부는 여기서 결론 내지 않는다.** 소프트웨어 라이선스와
> 가중치·데이터 이용 조건을 각각 검토했고, 실제 배포 자격은 상위 모델과
> 데이터의 조건을 따라야 한다. 이 문서는 법률 자문이 아니며, 실제 상용
> 배포 전에 해당 조건을 다시 확인하고 필요하면 법률 검토를 받는다.

부수적인 이유도 맞는다. torch/torchvision은 AI-2(잔반 이미지)에서 어차피
필요하다. 두 트랙이 같은 스택을 쓰면 환경이 하나로 유지된다.

### COCO pretrained라는 것의 의미

채택한 가중치는 COCO로 학습되었다. COCO는 그 자체로 층이 갈린다. 널리
알려진 바로는 annotation이 CC BY 4.0이고 이미지는 원저작자(Flickr)의 조건을
따르는데, 이번 조사에서 공식 Terms of Use 페이지 원문을 직접 읽지 못했다
(페이지가 스크립트로 렌더링되어 조회에 실패). 참고로 `cocodataset/cocoapi`
저장소의 license.txt는 BSD-2-Clause인데, 그것은 **API 코드**의 라이선스이지
데이터셋 조건이 아니다.

따라서 COCO 관련 표현은 **부분 확인** 상태로 둔다. 실제 배포 판단이 필요한
시점에 공식 페이지를 직접 확인한다.

**성능은 양보했다.** torchvision이 보고하는 COCO box mAP는
`fasterrcnn_mobilenet_v3_large_fpn` 32.8, `ssdlite320_mobilenet_v3_large` 21.3,
`fasterrcnn_resnet50_fpn_v2` 46.7이다. 최신 YOLO 계열보다 낮다. 이것은 baseline을
낮게 잡은 대가이고, 개선의 여지가 그만큼 남아 있다는 뜻이기도 하다. detector
경계(`PersonDetector`)를 따로 둔 이유가 이것이다. Apache-2.0인 YOLOX나 RT-DETR로
갈아끼울 때 바꿀 파일은 `detectors/` 안의 하나다.

### 사람 클래스 번호를 코드에 박지 않는다

COCO에서 person은 보통 1이지만, 모델마다 클래스 목록의 배치가 다를 수 있다.
어댑터는 weight 메타데이터의 `categories`에서 `"person"`의 위치를 찾는다.
모델을 바꿨을 때 조용히 다른 클래스를 세는 것이 가장 찾기 어려운 종류의
버그다.

## 5. Tracker

### 선택: ByteTrack에서 영감을 받은 2단계 IoU 연결 (저장소 내 구현)

> **이것은 ByteTrack 구현이 아니다.** 클래스 이름은
> `ByteTrackInspiredTracker`이고, 그 이름이 사실이다. 원 저자의 코드를
> 옮기거나 포팅한 것이 아니며, 원 알고리즘을 재현한 것도 아니다.

원 논문: Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every
Detection Box*, ECCV 2022. 공식 구현(ifzhang/ByteTrack)은 MIT이고, 여기서
코드를 가져오지 않았다.

### 빌려온 것과 다른 것

**빌려온 것은 하나다.** 신뢰도가 높은 상자로 먼저 잇고, 남은 track을 신뢰도가
낮은 상자로 한 번 더 구제한다는 2단계 연결 전략.

**나머지는 다르다.**

| | ByteTrack 원본 | 여기 |
|---|---|---|
| 운동 모델 | Kalman filter | 직전 프레임 이동량 등속 외삽 |
| 할당 | Hungarian 최적 할당 | IoU 내림차순 greedy |

**왜 단순화했는가.** 이 baseline의 목적에서 나온 결정이다. 익명 계수가
목적이고, 가벼운 시연이 필요하고, 단위 테스트가 결정적이어야 하고,
re-identification은 하지 않기로 했다. numpy/scipy 없이 표준 라이브러리만으로
돌면 계수 로직 전체를 무거운 의존성 없이 그대로 검증할 수 있다.

**단순화가 더 빠르다거나 원본과 같은 성능이라고 주장하지 않는다.** 이
tracker에서 나온 수치를 ByteTrack의 성능으로 읽으면 안 된다.

### A6에서 비교 후보를 하나 만들었다 — `MotionAwareTracker`

A5에서 "검출을 전부 회복시켜도 통과 사건은 43%만 살아난다"가 나왔다. 남은
절반은 궤적이 끊기는 문제였고, A6가 그것을 겨냥해 tracker 하나를 더 만들었다.

| | `ByteTrackInspiredTracker` (기본) | `MotionAwareTracker` (선택) |
|---|---|---|
| 소멸 | `max_age` 프레임 | `max_age_seconds` 초 |
| 외삽 | 놓친 시간과 무관하게 1프레임분 | 놓친 시간에 비례 |
| 속도 | 직전 프레임 차이 | 지수이동평균 |
| 할당 | IoU 내림차순 greedy | Hungarian 전역 할당 (`assignment.py`) |
| 외형 특징 | 없음 | 없음 |

**Kalman filter는 쓰지 않는다.** 진단이 가리킨 결함이 측정 잡음이 아니라
"놓친 시간과 무관한 1프레임 외삽"이었기 때문이다. 그래서 이름도
`MotionAwareTracker`이고, 이것을 ByteTrack이라고 부르지 않는다.

**기본값은 여전히 greedy 쪽이다.** A6 FINAL(처음 보는 corridor 카메라 12편)에서
동결한 주 지표인 통과 계수 오차가 개선되지 않았다(0.83 → 0.83). 사전에 정한
채택 조건을 통과하지 못했으므로 바꾸지 않았다.

그래도 저장소에 남긴 이유가 있다. 같은 평가에서 **나빠진 지표가 하나도 없고**
(precision 0.967 → 1.000, F1 0.746 → 0.774, fragmentation 12.16 → 11.29),
`ID_SWITCH`로 분류된 통과 실패는 13건 중 11건이 회복되었으며, 추가 비용이
프레임당 0.027 ms다. 카메라별 ROI와 통과 창을 다시 정한 뒤 재평가할 값어치가
있다. 자세한 것은 [A6 보고서](../../reports/cctv_a6_trajectory_continuity.md).

### 왜 이 계열인가

1. **외형 특징(re-id embedding)을 쓰지 않는다.** 상자의 위치만으로 잇는다.
   사람을 식별하지 않겠다는 제약과 정확히 맞는다. DeepSORT 계열은 re-id
   embedding을 쓰는데, 그것은 이 프로젝트가 하지 않기로 한 바로 그것이다.
2. **낮은 신뢰도 상자를 버리지 않는다.** 배식 줄에서 사람이 서로를 가리면
   신뢰도가 잠깐 떨어진다. 그때 track이 끊기면 같은 사람이 새 번호를 받고
   입장 인원이 부풀려진다. 2단계 매칭이 정확히 이 상황을 위한 것이다.
3. **detector 출력만 있으면 되고 별도 학습이 필요 없다.** baseline에 맞다.

### 추적 파라미터가 하는 일

| 값 | 짧으면 | 길면 |
|---|---|---|
| `max_age` | 잠깐 가려진 사람이 새 번호 → 입장 인원 과대 | 나간 사람이 남음 → occupancy 과대 |
| `min_hits` | 1회성 오탐이 바로 인원에 포함 | 실제 입장이 늦게 잡힘 |

기본값(`max_age=30`, `min_hits=3`)은 30fps 기준 약 1초와 0.1초에 해당한다.
**정답 데이터로 검증한 값이 아니라 출발점이다.**

이 둘은 아직 프레임 수 단위다. cooldown과 달리 fps 독립으로 만들지 않은
이유는, 이 값들이 "몇 번 관측했는가"에 대한 것이라 시간으로 옮기면 의미가
달라지기 때문이다. 다만 fps가 크게 다른 영상을 비교할 때는 이 둘도 함께
조정해야 한다. 한계로 기록해 둔다.

## 6. ROI

창밖을 지나가는 사람, 복도, 옆 매장이 프레임에 들어오면 식당 혼잡도가 아닌
것이 혼잡도에 섞인다. ROI는 그것을 자른다.

```yaml
roi:
  enabled: true
  x1: 0.0
  y1: 0.35
  x2: 1.0
  y2: 1.0
```

**0~1 정규화 좌표다.** 픽셀로 적으면 카메라를 1080p에서 720p로 바꾸는 순간
ROI가 엉뚱한 곳을 가리킨다.

**판정은 상자 전체가 아니라 한 점으로 한다.** 상자 전체가 ROI 안에 있기를
요구하면 경계에 반쯤 걸친 사람이 전부 빠지고, 조금이라도 겹치면 통과시키면
ROI가 사실상 넓어진다. 기본은 상자의 아래쪽 가운데(`BOTTOM_CENTER`), 즉 발이
닿는 지점이다. "그 공간에 있는가"에 가장 가깝다.

## 7. Line Crossing

입구에 선을 하나 긋고, IN 방향으로 넘은 사람을 센다.

```yaml
line:
  start: [0.10, 0.60]
  end:   [0.90, 0.60]
  in_direction: NEGATIVE
```

### 방향은 코드가 아니라 설정이 정한다

카메라를 반대편에 달면 IN과 OUT이 뒤집힌다. 그래서 방향을 하드코딩하지 않는다.

선을 `start → end`로 그으면, 그 진행 방향 기준으로 부호가 정해진다.

```
   화면 좌표계 (y가 아래로 증가)

            NEGATIVE  (y가 작은 쪽 = 화면 위쪽)
                 ↑
   start ●───────────────────● end      ← line: [0.1,0.6] → [0.9,0.6]
                 ↓
            POSITIVE  (y가 큰 쪽 = 화면 아래쪽)

   in_direction: NEGATIVE
     → 화면 아래에서 위로 넘어가면 ENTRY
     → 위에서 아래로 넘어가면 EXIT
```

**처음 자기 영상에 적용할 때 반드시 확인한다.** 사람이 들어오는데 `EXITED`가
오르면 방향이 뒤집힌 것이다. `in_direction`을 `POSITIVE`로 바꾸거나 `start`와
`end`를 맞바꾼다. `annotated.mp4`에 선과 방향이 그려져 있으니 눈으로 확인할
수 있다.

### 중복 계수를 막는 세 겹

이것이 이 모듈에서 가장 신경 쓴 부분이다. 사람이 선 근처에서 조금만 떨려도
`IN, IN, IN`이 연달아 찍힌다. 실제로는 한 번 들어간 것이다. 반대로 진짜
나갔다 다시 들어온 사람은 두 번 세야 한다. **이 둘을 가르는 것이 핵심이다.**

```
  1. 불감대 (band = 0.01)
     선에서 이보다 가까우면 ON_LINE. 확정된 쪽을 바꾸지 않는다.
     └ 선 바로 위의 미세 떨림을 통째로 무시한다.

  2. 이력 (min_confirm_distance = 0.03)
     band를 벗어나도 이 거리 이상 떨어져야 그쪽에 있다고 확정한다.
     └ 넘어갔다 얕게 되돌아오는 진동을 걸러낸다.

  3. 재계수 금지 구간 (cooldown_seconds = 0.5)
     한 번 센 track은 이 동안 다시 세지 않는다.
     └ 이 시간이 지난 IN→OUT→IN은 실제 재방문으로 보고 두 번 센다.
```

track별 상태는 다음과 같다.

```
  확정된 쪽 없음  ──(min_confirm_distance 넘어 확정)──▶  POSITIVE 또는 NEGATIVE
                                                            │
                              반대쪽으로 확정 + cooldown 지남 │
                                                            ▼
                                                        사건 발생
```

**영상 시작 시점에 이미 안쪽에 있던 사람은 입장으로 세지 않는다.** 처음
확정되는 쪽은 사건이 아니다. 그러지 않으면 영상을 켜는 순간 식당 안 사람이
전부 "방금 들어온" 것이 된다.

track이 사라지면 그 상태도 지운다. 상태가 영상 밖으로 나가지 않는다.

### cooldown은 초 단위다. 프레임 수가 아니다

이것이 중요한 이유는, 프레임 수로 두면 **같은 설정 파일이 영상마다 다른
정책이 되기** 때문이다.

```
   cooldown = 15 frames  라고 적으면

     10 fps 영상에서  → 1.5초
     30 fps 영상에서  → 0.5초
     60 fps 영상에서  → 0.25초
```

카메라를 바꾸거나 `frame_stride`를 조정하는 것만으로 입장 인원이 달라진다.
그 변화는 사람이 다르게 다녀서가 아니라 계수 규칙이 조용히 바뀌어서 생긴
것이고, 그런 두 실험은 비교할 수 없다.

그래서 판정을 관측 시각(`timestamp_seconds`)으로 한다. `LineCrossingCounter`는
프레임 번호를 받기는 하지만 그것은 사건이 난 장면을 나중에 찾아보기 위한
것이고, cooldown 판정에는 쓰지 않는다.

`tests/test_cctv_fps_invariance.py`가 이것을 지킨다. 동일한 물리적 궤적을
10 / 30 / 60 fps로 샘플링해 같은 계수가 나오는지 확인한다. 프레임 기반
정책으로 되돌리면 그 파일이 즉시 빨개진다.

**`max_age`와 `min_hits`는 아직 프레임 수 단위다.** 이 둘은 "몇 번
관측했는가"에 대한 값이라 시간으로 옮기면 뜻이 달라진다. 다만 fps가 크게
다른 영상을 비교할 때는 함께 조정해야 한다. 15절에 한계로 적어 둔다.

## 8. Reliability

숫자 하나만 내보내면 받는 쪽은 그것을 항상 같은 무게로 받는다. 역광이라
사람이 반쯤만 잡히는 프레임과 잘 보이는 프레임이 같은 값으로 나가면 안 된다.

관측마다 다음을 같이 낸다.

| 필드 | |
|---|---|
| `mean_detection_confidence` | 이 프레임 상자들의 평균 신뢰도. 상자가 없으면 **빈 값** |
| `min_detection_confidence` | 최소 신뢰도 |
| `active_track_count` | 세고 있는 사람 수 |
| `reliability_flag` | `OK` / `LOW_CONFIDENCE` / `NO_DETECTION` |

### 신뢰도와 혼잡도는 다른 축이다

이 flag에 **`CROWDED`는 없다.** 한때 넣었다가 뺐다. 이유가 셋이다.

1. **혼잡도와 신뢰도는 다른 개념이다.** 사람이 많다고 관측이 자동으로 못 믿을
   것이 되지는 않는다. 밀집이 계수 오차를 키울 것이라고 예상은 하지만, 그것은
   측정으로 확인할 일이지 정의로 못박을 일이 아니다.
2. **백엔드에 이미 `CROWDED`라는 업무 등급이 있다.**
   (`COMFORTABLE / NORMAL / CROWDED / VERY_CROWDED`) 같은 이름을 다른 뜻으로
   쓰면 반드시 섞인다.
3. **몇 명부터 붐비는 것인지 정해진 적이 없다.** 좌석 수에 달렸고, 그 정책은
   아직 없다(12절). 정해지지 않은 정책을 모델 출력에 박을 수 없다.

밀집·저조도·가림 같은 것은 **scene condition**이고, 평가 metadata 쪽
(`groundtruth.py`의 `condition` 열)에서 다룬다. 데이터셋이 제공했거나 사람이
직접 붙인 경우에만 쓴다. 런타임 신뢰 신호와 섞지 않는다.

### `NO_DETECTION`이 뜻하는 것

정확히 이것뿐이다.

> **detector가 이 프레임에서 유효한 사람 상자를 하나도 내지 않았다.**

이것은 "식당이 비었다"가 **아니다.** "모델이 실패했다"도 **아니다.** 실제로
아무도 없었을 수도 있고, 있었는데 전부 놓쳤을 수도 있는데, 런타임에서 그
둘을 구분할 방법이 없다. 구분할 수 없다는 사실 자체를 값으로 남긴 것이다.

`is_low_reliability`가 이 값을 참으로 보는 이유도 같다. 구분할 수 없으면
자동으로 믿지 않는다.

상자가 없을 때 평균 신뢰도를 0.0으로 채우지 않는다. 0.0은 "신뢰도가 0이었다"로
읽히고, 그것은 "잴 것이 없었다"와 다르다. 나중에 평균을 내면 그 차이가 그대로
지표를 왜곡한다.

### confidence를 정확도로 읽지 마라

> **detector confidence 0.8은 "이 상자가 사람일 가능성에 대한 모델의 점수"이지
> "사람 수를 80% 맞혔다"가 아니다.**

실제로 이런 일이 일어난다.

- 두 사람이 겹쳐 한 상자로 잡혔다. 그 상자의 confidence는 높다. **모델은
  확신하고 개수는 틀렸다.**
- 사람이 아닌 것을 사람으로 잡았는데 confidence가 0.9다.
- 사람이 아예 안 잡혔다. **못 잡은 것에는 confidence 자체가 없다. 누락은 이
  신호로 절대 알 수 없다.**

그러니 `reliability_flag`는 "이 값이 틀렸다"가 아니라 "이 프레임은 조건이
나쁘니 자동으로 믿지 말고 사람이 확인할 후보로 두자"는 뜻이다.

### 이것은 uncertainty estimator가 아니다

용어를 정확히 쓴다. 이 신호는 **heuristic signal**이다.

- 보정된(calibrated) 불확실성 추정이 아니다. `LOW_CONFIDENCE`가 붙은 프레임이
  틀릴 확률이 몇 %인지 이 값은 말하지 않는다.
- 임계값 하나로 갈리는 이분법이다. 평균 신뢰도 0.499와 0.501은 실질적으로
  같은 프레임이지만 flag는 다르게 붙는다.
- **누락은 원리적으로 잡지 못한다.** 실제로 10명이 있는데 5명만 잡혔고 그
  5개가 전부 0.95라면, 평균 신뢰도는 높고 flag는 `OK`이며 계수는 절반이다.
  못 잡은 것에는 confidence 자체가 없기 때문이다. 이것은 임계값을 조정해서
  해결되는 문제가 아니라 이 신호가 볼 수 없는 영역이다.

**이 flag와 실제 계수 오차 사이의 관계는 아직 측정하지 않았다.** 그것이
RQ3이고, 상관이 없다는 결과가 나올 수도 있다. 그러면 다른 신호를 찾아야 한다.

### 확장 지점

향후 "낮은 신뢰 구간은 백엔드 전송에서 보류한다"로 이어질 자리를 만들어 두었다.
지금은 flag를 계산해서 CSV와 요약에 담는 데까지만 하고, 실제 전송 차단은
구현하지 않았다. 무엇을 기준으로 보류할지가 아직 측정되지 않았기 때문이다.

## 9. 시간 창

프레임 하나의 occupancy는 생각보다 흔들린다. `windows.py`가 1/3/5초 창의
평균·중앙값·최댓값을 낸다.

**그렇지만 창 통계를 기본 결과로 두지 않는다.** 평활한 곡선은 언제나 더
그럴듯해 보인다. 원본을 감추고 평활값만 보여 주면 실제로 얼마나 흔들리는지가
보고서에서 사라진다. 기본 출력은 프레임별 원본이고, 창 통계는 `windows.csv`로
따로 나간다.

## 10. 평가 지표

### Occupancy

| | |
|---|---|
| 주 지표 | **MAE** — 단위가 사람 수 그대로다. "평균 1.8명 틀린다" |
| 보조 | RMSE — 큰 오차에 민감 |
| 보조 | bias = mean(예측 − 실제) — 부호가 있다 |

MAE가 같아도 계속 적게 세는 것과 많이 세는 것은 운영에서 완전히 다른 문제라
bias를 같이 본다.

**MAPE는 주 지표로 쓰지 않는다.** 식당은 비는 시간이 있고, 실제 인원이 0인
구간에서 MAPE는 정의되지 않거나 무한대로 튄다. 0을 빼고 계산하면 가장 쉬운
구간만 남는다. 어느 쪽이든 숫자가 실제보다 좋아 보인다.

### Entry Count

영상 하나에 대해 `|예측 − 실제|`. 영상이 여럿이면 그 평균(MAE across videos).

사건 단위 annotation이 충분해지면 precision / recall / F1으로 넘어갈 수 있다.
그 전에 그 숫자를 쓰면 정답이 없는데 있는 것처럼 된다.

### Tracking

ID switch, IDF1, HOTA는 이번 단계의 필수가 아니다. 추적은 여기서 목적이 아니라
entry counting의 수단이고, 이 지표들을 제대로 재려면 track 단위 annotation이
있는 데이터셋과 `TrackEval` 같은 의존성이 필요하다. **계수가 주 과제라는 점을
유지한다.** 추적 품질은 entry count 오차로 간접 평가한다.

### Runtime

서비스용이므로 `processing_fps`를 기록한다. `metadata.json`에 장치, 해상도,
패키지 버전이 같이 남는다. **이 숫자를 다른 환경의 일반 성능으로 옮겨 말하지
않는다.**

## 11. Domain Shift

포트폴리오에서 가장 중요한 부분이다. "잘 되는 영상에서 잘 된다"는 결과가
아니다.

평가 표본에 조건 태그를 붙여 조건별로 나눠 본다.

```
daylight / low-light / sparse / crowded / occlusion / far-distance / oblique-camera
```

`metrics.score_by_condition()`과 `render_condition_table()`이 다음 표를 만든다.

| Condition | Samples | Occupancy MAE | Entry Error |
|---|---:|---:|---:|
| (조건 annotation 없음) | 0 | N/A | N/A |

**지금은 비어 있고, 그것이 맞다.** 조건 태그는 데이터셋이 제공한 것이거나
사람이 직접 붙인 것이어야 한다. 조건을 모르는 표본에 "주간"이라고 적으면 표는
채워지지만 그 표는 거짓이다. 빈 태그를 넣으면 코드가 거부한다.

## 12. Spring 연동 계약

백엔드에 `POST /internal/v1/congestion-observations`가 있다. AI-1의 출력이
언젠가 여기로 간다.

| AI-1 출력 | Spring 필드 | 비고 |
|---|---|---|
| `occupancy_count` | `peopleCount` | 순간값. **entry count가 아니다** |
| 관측 시각 | `observedAt` | 프레임 시각을 절대 시각으로 환산 |
| (고정) | `source` = `CCTV_AI` | 값의 진실은 백엔드가 가진다 |
| `reliability_flag` | (없음) | 아래 참고 |
| `entries_total` | (없음) | 아래 참고 |

### 지금 연결하지 않는 이유 — congestion level

백엔드는 `COMFORTABLE / NORMAL / CROWDED / VERY_CROWDED` 같은 등급을 필요로
한다. 그런데 **`peopleCount`를 등급으로 바꾸는 임계값이 정해진 적이 없다.**

40명이 붐비는 것인지는 그 식당의 좌석 수에 달렸다. 100석짜리에서 40명은
한산하고 50석짜리에서는 붐빈다. 모델이 임계값을 정할 근거가 없다.

**연구 출력은 `people_count`까지다.** 등급 변환은 좌석 수 또는 운영 정책에
기반한 threshold policy가 정해진 뒤, 서비스 계층에서 한다. 모델 안에 임의
임계값을 넣으면 정해지지도 않은 정책이 모델 버전에 박혀 버린다.

### 아직 계약에 없는 것

`reliability_flag`와 `entries_total`을 받을 자리가 백엔드 계약에 없다. 둘 다
지금 늘리지 않는다. 신뢰 신호는 무엇을 기준으로 보류할지가 측정되지 않았고,
입장 인원은 그것이 실제 식수와 얼마나 맞는지 검증되지 않았다. **검증 전에
계약을 늘리면, 보내는 쪽은 쓰이고 있다고 믿고 받는 쪽은 무시한다.**

이 두 가지는 [Gap](#15-한계)에 기록해 둔다.

## 13. 시연

### 설치 — 서비스 환경과 분리한다

핵심 로직은 추가 설치 없이 돌지만, 실제 영상을 처리하려면 무거운 의존성이
필요하다.

**`.venv`는 FastAPI 서비스 개발 환경이다.** 여기에 torch/CUDA를 넣지 않는
것을 권장한다. 수 GB짜리 설치가 서비스 의존성 해석에 끼어들면, 서비스가
깨졌을 때 원인이 연구 쪽인지 서비스 쪽인지 가리기 어려워진다.

연구용 환경을 따로 만든다.

    py -3.12 -m venv .venv-cctv
    .venv-cctv\Scripts\python.exe -m pip install --upgrade pip
    .venv-cctv\Scripts\python.exe -m pip install -e ".[cctv]"

`pyyaml`, `opencv-python`, `numpy`, `torch`, `torchvision`을 받는다. CUDA 빌드는
용량이 크다. GPU를 쓰지 않을 것이면 CPU 빌드로 충분하다.

`.venv-cctv/`는 `.gitignore`에 있다.

**아직 만들지 않았다.** 실제 설치와 영상 실행은 다음 단계의 일이다. 위
명령은 그때 현재 `pyproject.toml`과 환경을 확인한 뒤 실행한다.

### 실행

    .venv-cctv\Scripts\python.exe ^
      research\cctv_counting\scripts\run_video.py ^
      --input C:\path\to\your\video.mp4 ^
      --config research\cctv_counting\configs\demo.yaml ^
      --output-dir outputs\cctv_demo

**영상 경로는 반드시 직접 준다.** 이 도구는 저장소 어딘가에 있는 영상을
찾아서 알아서 쓰지 않는다. CCTV 영상에는 사람이 찍혀 있고, 어느 영상을
처리할지는 사람이 정해야 한다.

유용한 옵션:

    --device cpu          GPU 없이 돌린다
    --confidence 0.6      설정 파일의 임계값을 덮어쓴다
    --max-frames 300      앞부분만 빠르게 확인한다

### 만들어지는 것

```
outputs/cctv_demo/
  annotated.mp4       상자, 번호, 선, ROI, CURRENT/ENTERED/EXITED, 낮은 신뢰 표시
  observations.csv    프레임별 관측
  summary.json        영상 전체 요약
  metadata.json       설정, 모델, weight 이름, 장치, 패키지 버전, 실행 시각
  windows.csv         시간 창 통계 (설정에서 켰을 때)
```

**원본은 건드리지 않는다.** 출력 경로가 입력과 같으면 시작 전에 거부한다.

`metadata.json`이 재현을 담당한다. 이 파일 하나로 어떤 설정과 어떤 weight로
언제 돌린 결과인지 알 수 있다. 지금 pipeline에 난수는 없지만 `seed`도 같이
기록해 둔다.

### 평가

정답 파일을 채운 뒤:

    .venv-cctv\Scripts\python.exe ^
      research\cctv_counting\scripts\evaluate.py ^
      --video-id demo-01 ^
      --observations outputs\cctv_demo\observations.csv ^
      --occupancy-gt research\cctv_counting\datasets\annotations\occupancy_ground_truth.csv ^
      --crossing-gt research\cctv_counting\datasets\annotations\crossing_ground_truth.csv

정답이 비어 있으면 지표를 내지 않고 그렇다고 말한 뒤 0이 아닌 코드로 끝난다.
0건으로 계산한 MAE 0.0은 완벽한 성능처럼 보이지만 아무것도 재지 않은 것이다.

### Demo와 Benchmark를 섞지 않는다

| | Demo | Benchmark |
|---|---|---|
| 무엇 | 우리 영상에서 실제로 도는 모습 | 정답이 있는 데이터에서 정량 평가 |
| 결과 | "돌아간다" | MAE, 오차, FPS |
| 지금 | 실행 대기 (영상 미제공) | 미측정 |

시연 영상에서 나온 숫자는 성능이 아니다. 정답이 없기 때문이다. 보고서에서도
절을 나눠 둔다.

## 14. 개인정보

### 하지 않는 것

- 얼굴 검출, 얼굴 특징 추출, 얼굴 대조
- 같은 사람을 다른 영상 / 다른 날 / 다른 카메라에서 다시 알아보기
- 이름, 학번, 전화번호, 회원 번호와의 연결
- 성별, 나이 등 속성 추정

### 하는 것

- 익명 사람 상자
- 영상 한 세션 안에서만 유효한 임시 track 번호

### track_id의 경계

`track_id`는 프레임 사이를 잇는 포인터일 뿐이다. 다른 영상, 다른 날의 같은
번호는 같은 사람이 아니다.

- **파일로 나가지 않는다.** `observations.csv`에 `track_id` 열이 없다. 한 줄은
  한 프레임의 집계값이지 한 사람의 기록이 아니다. 번호를 시각과 함께 파일로
  남기는 순간 그것은 사람별 동선 기록이 된다.
- **영상 경계를 넘지 않는다.** `reset()`이 번호를 1부터 다시 시작한다.
- **끝난 번호의 상태는 지운다.** `LineCrossingCounter.forget()`.

### 검사

`tests/test_cctv_privacy.py`가 매 실행마다 확인한다.

- 얼굴 인식 / re-identification 라이브러리가 코드에 없는가
- 신원 필드 이름이 코드에 없는가
- CSV, 정답 파일, 요약 스키마에 식별 가능한 열이 없는가
- `summary.json`의 `video`가 전체 경로가 아니라 파일 이름인가 (경로에는 현장
  이름이나 사람 이름이 섞이곤 한다)

### 저장

원본 CCTV 영상과 `annotated.mp4`는 **저장소에 커밋하지 않는다.** `.gitignore`가
`*.mp4`, `outputs/`, `datasets/raw/`를 막는다. 원본의 보관 기간과 접근 권한은
이 저장소가 아니라 운영 정책에서 정한다.

## 15. 한계

**정량 성능이 없다.** 정답 데이터가 아직 없어서 MAE도 오차도 측정하지 않았다.
지금 있는 것은 "끝까지 도는 pipeline"이지 "얼마나 맞는지 아는 pipeline"이
아니다.

**실제 급식소 데이터가 하나도 없다.** 카메라 각도도 조명도 밀집도도 모른다.
공개 데이터로 잰 숫자가 현장에서 얼마나 유지될지는 domain shift 실험 전에는
알 수 없다.

**추적 단순화의 대가를 재지 않았다.** Kalman 대신 등속 외삽, Hungarian 대신
greedy를 썼는데, 이것이 얼마나 손해인지 모른다. 이 tracker는 ByteTrack이
아니며, 여기서 나올 숫자를 ByteTrack의 성능으로 옮겨 말할 수 없다.

**`max_age`와 `min_hits`가 아직 프레임 수 단위다.** cooldown은 초 기반으로
바꿨지만 이 둘은 남아 있다. fps가 크게 다른 영상을 비교할 때는 함께 조정해야
하고, 그 조정이 필요하다는 사실 자체가 문서에만 있다.

**reliability는 heuristic이지 보정된 불확실성이 아니다.** 특히 누락은
원리적으로 잡지 못한다. 8절.

**라이선스 검토는 코드/가중치/데이터 세 층으로 나눠서 했지만, 전부 확인하지는
못했다.** COCO의 공식 Terms 페이지와 MOT 계열은 조회에 실패했다. 실제 배포
판단이 필요한 시점에 다시 확인한다. 이 저장소의 문서는 법률 자문이 아니다.

**밀집 상황이 가장 약할 것으로 보이나 확인되지 않았다.** 사람이 겹치면 상자가
합쳐져 적게 세일 것이라고 예상하지만, 관찰한 것이 아니라 예상이다. 실제로
돌려 보기 전까지는 실패 사례 목록에 적지 않는다.

**congestion level 임계값이 없다.** 12절.

**reliability flag와 실제 오차의 관계가 미측정이다.** RQ3.

## 16. 향후 Spring 연동

12절의 표대로 연결하되, 그 전에 다음이 필요하다.

1. 정답 데이터로 occupancy MAE와 entry count 오차를 측정한다.
2. 급식소별 좌석 수 또는 운영 기준으로 congestion level 임계값 정책을 정한다.
3. reliability flag가 실제 오차와 상관이 있는지 확인한다. 있으면 보류 기준을,
   없으면 다른 신호를 찾는다.
4. 위가 끝난 뒤 백엔드 계약에 `reliability` 필드를 추가할지 논의한다.

---

## 연구 질문

**RQ1.** 공개 pretrained detector와 tracker만으로, 식당/보행 영상에서
occupancy estimation과 entry counting을 어느 정도까지 할 수 있는가?
→ *pipeline 구현 완료. 정량 측정 대기.*

**RQ2.** 조명, 가림, 밀집도, 카메라 각도 같은 domain condition에 따라 계수
오차가 어떻게 변하는가?
→ *조건별 평가 코드 준비 완료. 조건 annotation이 붙은 데이터 필요.*

**RQ3.** detector/tracker의 confidence로 신뢰하기 어려운 관측을 식별할 수
있는가?
→ *신호 계산까지 구현. flag와 실제 오차의 상관은 미측정.*

RQ3는 이번 단계에서 논문 수준의 uncertainty calibration까지 가지 않는다.
최소한의 신뢰 신호를 만들어 두고, 그것이 쓸모 있는지를 다음 단계에서 잰다.

## 참고

- [데이터셋 후보 조사](datasets/README.md)
- [평가 절차](evaluation/README.md)
- [baseline 보고서](../../reports/cctv_baseline.md)
