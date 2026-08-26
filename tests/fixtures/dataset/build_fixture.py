"""테스트용 합성 데이터셋을 만든다.

실제 급식 사진이 아니다. Pillow로 그린 단색 이미지이며, 도구가 도는지
확인하는 용도다. 이 파일들을 학습 데이터라고 부르면 안 된다.

    python tests/fixtures/dataset/build_fixture.py
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
IMAGES = HERE / "images"

#: 정답 정의 버전. 스키마 쪽과 같은 값을 쓴다.
LABEL_VERSION = "waste-mass-ratio-v1"

#: (sample_id, participant, session, menu, served_g, leftover_g, source, label, reason)
ROWS = [
    ("a1000000000000000000000000000001", "P01", "S01", 3, 500.0, 100.0, "MEASURED", "APPROVED", None),
    ("a1000000000000000000000000000002", "P01", "S01", 3, 480.0, 0.0, "MEASURED", "APPROVED", None),
    ("a1000000000000000000000000000003", "P02", "S01", 3, 520.0, 520.0, "MEASURED", "APPROVED", None),
    ("a1000000000000000000000000000004", "P02", "S02", 4, 450.0, 90.0, "MEASURED", "APPROVED", None),
    ("a1000000000000000000000000000005", "P03", "S02", 4, 610.0, 244.0, "MEASURED", "APPROVED", None),
    ("a1000000000000000000000000000006", "P03", "S02", 5, 500.0, 25.0, "MEASURED", "APPROVED", None),
    ("a1000000000000000000000000000007", "P04", "S03", 5, 500.0, 350.0, "STANDARD_PORTION", "APPROVED", None),
    ("a1000000000000000000000000000008", "P04", "S03", 5, 500.0, 175.0, "STANDARD_PORTION", "APPROVED", None),
    ("a1000000000000000000000000000009", "P05", "S03", 3, 470.0, 47.0, "MEASURED", "APPROVED", None),
    ("a1000000000000000000000000000010", "P05", "S04", 3, 530.0, 106.0, "MEASURED", "APPROVED", None),
    ("a1000000000000000000000000000011", "P06", "S04", 4, 490.0, 392.0, "MEASURED", "APPROVED", None),
    ("a1000000000000000000000000000012", "P06", "S04", 4, 505.0, 50.5, "MEASURED", "APPROVED", None),
    ("a1000000000000000000000000000013", "P07", "S05", 5, None, None, None, "REJECTED", "SEVERE_BLUR"),
    ("a1000000000000000000000000000014", "P07", "S05", 5, None, None, None, "REJECTED", "NO_RELEVANT_FOOD"),
    ("a1000000000000000000000000000015", "P08", "S05", 3, None, None, None, "REJECTED", "SEVERE_OCCLUSION"),
    ("a1000000000000000000000000000016", "P08", "S06", 3, 500.0, 150.0, "MEASURED", "APPROVED", None),
]

FIELDNAMES = [
    "sample_id",
    "image_path",
    "menu_id",
    "site_id",
    "service_date",
    "meal_type",
    "captured_at",
    "participant_group_id",
    "capture_session_id",
    "served_edible_mass_g",
    "leftover_edible_mass_g",
    "waste_rate",
    "served_mass_source",
    "analysis_label",
    "rejection_reason",
    "image_sha256",
    "split",
    "label_version",
]


def _color(index: int) -> tuple[int, int, int]:
    """표본마다 다른 색. 같은 파일이 중복으로 잡히지 않게 한다."""
    return (60 + index * 11 % 180, 90 + index * 7 % 150, 120 + index * 13 % 120)


def build() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, row in enumerate(ROWS):
        (
            sample_id,
            participant,
            session,
            menu_id,
            served,
            leftover,
            source,
            label,
            reason,
        ) = row

        image_path = IMAGES / f"{sample_id}.png"
        Image.new("RGB", (24, 24), _color(index)).save(image_path, format="PNG")
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()

        rate = None if served is None or leftover is None else leftover / served
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": f"images/{sample_id}.png",
                "menu_id": menu_id,
                "site_id": 1,
                "service_date": "2026-01-15",
                "meal_type": "LUNCH",
                "captured_at": "2026-01-15T04:10:00Z",
                "participant_group_id": participant,
                "capture_session_id": session,
                "served_edible_mass_g": "" if served is None else served,
                "leftover_edible_mass_g": "" if leftover is None else leftover,
                "waste_rate": "" if rate is None else f"{rate:.10f}",
                "served_mass_source": source or "",
                "analysis_label": label,
                "rejection_reason": reason or "",
                "image_sha256": digest,
                "split": "",
                "label_version": LABEL_VERSION,
            }
        )

    with (HERE / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "dataset_version": "synthetic-fixture-v1",
        "label_version": LABEL_VERSION,
        "created_at": "2026-01-15T00:00:00Z",
        "collection_site": None,
        "scale_resolution_g": 1.0,
        "protocol_version": "pilot-v1",
        "notes": "테스트용 합성 데이터. 실제 급식 사진이 아니다.",
    }
    (HERE / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    build()
