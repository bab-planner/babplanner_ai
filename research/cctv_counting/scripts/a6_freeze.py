"""FINAL 실행 전에 설정과 판정 규칙을 동결한다.

**왜 파일로 남기는가.** 결과를 본 뒤에 임계값이나 규칙을 조금 고치면 그
숫자는 아무것도 재지 않은 것이 된다. 사람의 기억으로는 그것을 막을 수 없다.
동결 내용을 JSON으로 쓰고 SHA-256을 같이 남긴다. 나중에 누가 봐도 FINAL을
돌리기 전의 규칙이 무엇이었는지 확인할 수 있다.

A5의 `final_test_freeze.json`과 같은 방식이다.

    .venv-cctv\\Scripts\\python.exe research\\cctv_counting\\scripts\\a6_freeze.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from a4_run import (  # noqa: E402
    BAND,
    COOLDOWN_SECONDS,
    EVENT_TOLERANCE_SECONDS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    LINE_X,
    MIN_CONFIRM_DISTANCE,
    SAMPLE_EVERY,
    SOURCE_FPS,
)
from a6_run import (  # noqa: E402
    CORRIDOR_SEQUENCES,
    CROSSING_WINDOW_SECONDS,
    DETECTOR_MODEL,
    DETECTOR_TAG,
    OCCUPANCY_THRESHOLDS,
    POLICIES,
)

OUT = Path("research/cctv_counting/experiments/a6_continuity/final_test_freeze.json")

FREEZE = {
    "stage": "A6",
    "frozenAt": "2026-08-28",
    "purpose": (
        "FINAL(corridor 12편) 실행 전에 detector / 임계값 / tracker / 선 규칙 / "
        "판정 규칙을 고정한다. 이 파일이 쓰인 뒤에는 바꾸지 않는다."
    ),
    "detector": {
        "tag": DETECTOR_TAG,
        "model": DETECTOR_MODEL,
        "note": (
            "A5에서 선택된 조건 그대로. A6의 축은 tracking mechanism이므로 "
            "detector는 고정한다. A5에서 crossing이 더 좋아 보였던 resnet50을 "
            "사후 이유로 채택하지 않는다."
        ),
        "scoreFloor": 0.05,
        "fineTuned": False,
    },
    "dataRoles": {
        "DEVELOPMENT": "CAVIAR 쇼핑센터 front 5편 (A4 DEV). 후보 선택에 쓴다.",
        "ANALYSIS_SHOP": "CAVIAR 쇼핑센터 front 8편 (A4 TEST -> A5 ANALYSIS). 원인 분석 전용.",
        "ANALYSIS_LOBBY": "CAVIAR INRIA 로비 17편 (A5 FINAL). 결과를 이미 봤다. 원인 분석 전용.",
        "FINAL": "CAVIAR 쇼핑센터 corridor 12편. 한 번도 추론한 적 없다. 한 번만 채점한다.",
    },
    "finalSequences": [s for s, _ in CORRIDOR_SEQUENCES],
    "geometry": {
        "frameWidth": FRAME_WIDTH,
        "frameHeight": FRAME_HEIGHT,
        "sourceFps": SOURCE_FPS,
        "lineXNormalized": LINE_X,
        "band": BAND,
        "minConfirmDistance": MIN_CONFIRM_DISTANCE,
        "cooldownSeconds": COOLDOWN_SECONDS,
        "roi": [0.0, 0.20, 1.0, 0.95],
        "anchor": "BOTTOM_CENTER",
        "note": "A4와 완전히 같다. 카메라가 바뀌었다고 선을 다시 맞추지 않는다.",
    },
    "scoring": {
        "sampleEvery": SAMPLE_EVERY,
        "eventToleranceSeconds": EVENT_TOLERANCE_SECONDS,
        "crossingWindowSeconds": CROSSING_WINDOW_SECONDS,
        "detectionMatchIou": 0.5,
        "scoredFrames": "검출 프레임 ∩ 정답 프레임",
        "occludedGtInDetectionRecall": False,
    },
    "occupancyCandidates": OCCUPANCY_THRESHOLDS,
    "policies": {name: policy.as_dict() for name, policy in POLICIES.items()},
    "baseline": "P0_baseline",
    "selected": "P2_motion",
    "selectionRule": {
        "note": (
            "규칙의 형태(주 지표, guardrail 종류, tie-break 순서)는 DEV를 "
            "돌리기 전에 정했다. 수치 margin은 DEV 통계를 보고 정했고, "
            "FINAL을 돌리기 전에 이 파일로 동결한다."
        ),
        "occupancy": {
            "primary": "DEV occupancy macro MAE 최소",
            "guardrail": (
                "검출 micro recall이 baseline(T=0.7) 대비 절대 0.10 이상 "
                "낮으면 reject. A5에서 낮은 MAE가 오탐/누락 상쇄일 수 있다는 "
                "것을 확인했으므로 MAE 하나로 고르지 않는다."
            ),
            "alsoReported": ["bias", "falsePositivesPerFrame"],
            "devResult": {
                "T=0.5": {"macroMae": 1.612, "microRecall": 0.618, "fpPerFrame": 1.963},
                "T=0.7": {"macroMae": 0.644, "microRecall": 0.578, "fpPerFrame": 0.593},
            },
            "decision": "T_occ = 0.7. 주 지표에서 단독 최소이고 guardrail을 통과한다.",
        },
        "entry": {
            "primary": "DEV 평균 절대 통과 계수 오차 ((|dIN| + |dOUT|) / 2의 시퀀스 평균) 최소",
            "guardrail": (
                "사건 precision(IN/OUT 평균)이 0.85 미만이면 reject, 그리고 "
                "baseline precision 대비 절대 0.10 이상 낮으면 reject. "
                "recall을 올리려고 없는 통과를 만들어 내는 것을 막는다. "
                "0.85는 제품 관점의 절대 하한이다 — 잘못 센 입장은 수요 "
                "정답을 그대로 부풀린다."
            ),
            "tieBreak": [
                "1) 계수 오차 차이가 0.05건 이내이면 사건 F1이 높은 쪽",
                "2) 그래도 같으면 gap recovery rate가 높은 쪽",
                "3) 그래도 같으면 더 단순한 쪽 (max_age가 짧고 low threshold가 높은 쪽)",
            ],
            "devResult": {
                "P0_baseline": {"countError": 0.30, "precision": 1.0, "f1": 0.737, "gapRecovery": 0.793},
                "P1_dual_threshold": {"countError": 0.20, "precision": 1.0, "f1": 0.857, "gapRecovery": 0.748},
                "P2_motion": {"countError": 0.10, "precision": 1.0, "f1": 0.957, "gapRecovery": 0.803},
                "P3_motion_long_age": {"countError": 0.10, "precision": 1.0, "f1": 0.957, "gapRecovery": 0.803},
                "P4_motion_low01": {"countError": 0.20, "precision": 1.0, "f1": 0.857, "gapRecovery": 0.745},
            },
            "decision": (
                "P2와 P3가 0.10으로 공동 최소이고 F1(0.957)과 gap "
                "recovery(0.803)까지 같다. tie-break 3에 따라 max_age가 짧은 "
                "P2를 선택한다. 두 후보 모두 guardrail을 통과한다."
            ),
        },
    },
    "finalJudgement": {
        "note": (
            "FINAL에서 P0과 P2만이 판정 대상이다. P1/P3/P4도 함께 돌려 "
            "기록하지만 **선택을 바꾸는 데 쓰지 않는다.** A5에서 대조군 E3를 "
            "함께 돌리되 선택에 쓰지 않은 것과 같다."
        ),
        "adoptIf": [
            "FINAL 통과 계수 오차가 baseline보다 낮다",
            "FINAL 사건 precision guardrail을 통과한다",
            "추가 tracker 비용이 서비스 시연을 막지 않는다",
        ],
        "otherwise": (
            "개선하지 못하면 기존 tracker를 유지하고 '더 복잡한 tracker가 "
            "항상 좋지는 않다'를 결과로 기록한다."
        ),
    },
    "hypotheses": {
        "H5": (
            "통과 누락은 전체 detector recall보다 통과 순간 주변의 시간적 "
            "연속성과 더 강하게 연결된다. 판정 근거: (a) 검출이 충분했던 "
            "사건(창 검출 coverage >= 0.5)만 봐도 놓친 사건의 창 연속성이 "
            "절대 0.10 이상 낮다, (b) 연속성을 개선한 tracker가 통과 회복을 "
            "실제로 늘린다. 둘 다 필요하다."
        ),
        "H6": (
            "entry 갈래에 낮은 신뢰도 상자를 허용하면 통과 recall이 오르고, "
            "motion association이 false crossing 증가를 억제한다. 판정 근거: "
            "recall 상승 + precision guardrail 통과."
        ),
    },
    "forbidden": [
        "FINAL 결과를 본 뒤 임계값/tracker/선 규칙 변경",
        "시퀀스별 선 위치 튜닝",
        "detector 재탐색, 해상도 실험, fine-tuning",
        "re-identification / appearance embedding",
        "cross-session identity",
    ],
}


def main() -> int:
    # A5와 같은 규약. 지문을 **기록 안에** 넣는다. 옆 파일로 두면
    # .gitignore가 experiments/ 안의 .json만 남기므로 지문이 커밋되지 않고,
    # 커밋되지 않은 지문은 나중에 아무것도 증명하지 못한다.
    payload = json.dumps(FREEZE, ensure_ascii=False, indent=2, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    record = dict(FREEZE)
    record["sha256OfRecordWithoutHash"] = digest
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{OUT}\nsha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
