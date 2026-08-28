"""AI-2. 식판 이미지 기반 잔반 분석 — 연구 도구.

**모델이 없다.** 이 패키지에는 학습 코드가 없고, W1에서 만들지 않았다.
여기 있는 것은 그 앞에 있어야 하는 것들이다.

    schemas.py          정답의 정의를 코드로 못박는다
    metrics.py          bias와 순서형 지표. MAE/RMSE는 서비스 것을 쓴다
    selective.py        coverage / selective risk
    splits.py           그룹 단위 분할. 누수를 막는다
    audit.py            지금 쓸 수 있는 표본이 몇 개인가
    public_datasets.py  공개 데이터 후보와 확인된 라이선스

**정답은 질량비다** (`docs/waste_ground_truth_protocol.md`).

    waste_rate = leftover_edible_mass_g / served_edible_mass_g

사진의 면적 비율도, 사람의 눈대중도 이 값이 아니다.

무거운 의존성이 없다. 표준 라이브러리와 서비스 패키지만 쓰므로 `.venv`의
`pytest` 한 번으로 전부 검증된다.
"""

from waste_estimation.audit import audit_images, audit_samples, summarize_rates
from waste_estimation.metrics import bias, class_index_mae, to_class_index
from waste_estimation.public_datasets import (
    CANDIDATES,
    DatasetRole,
    LicenseStatus,
    PublicDataset,
)
from waste_estimation.schemas import (
    LABEL_VERSION,
    SampleSource,
    WasteSample,
    WasteSchemaError,
    validate_waste_rate,
    waste_rate_from_mass,
)
from waste_estimation.selective import (
    SelectivePoint,
    coverage_at,
    coverage_risk_curve,
    risk_at_coverage,
    selective_risk,
)
from waste_estimation.splits import (
    group_holdout_split,
    group_kfold,
    leaked_groups,
)

__all__ = [
    "CANDIDATES",
    "DatasetRole",
    "LABEL_VERSION",
    "LicenseStatus",
    "PublicDataset",
    "SampleSource",
    "SelectivePoint",
    "WasteSample",
    "WasteSchemaError",
    "audit_images",
    "audit_samples",
    "bias",
    "class_index_mae",
    "coverage_at",
    "coverage_risk_curve",
    "group_holdout_split",
    "group_kfold",
    "leaked_groups",
    "risk_at_coverage",
    "selective_risk",
    "summarize_rates",
    "to_class_index",
    "validate_waste_rate",
    "waste_rate_from_mass",
]
