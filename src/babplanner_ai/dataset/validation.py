"""데이터셋 점검.

스키마가 막는 것은 행 하나의 앞뒤가 맞는지까지다. 여기서는 파일이 실제로
있는지, 이미지가 열리는지, 같은 사진이 두 번 들어왔는지, 분할이 새지
않았는지처럼 행 하나만 봐서는 알 수 없는 것을 본다.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from babplanner_ai.dataset.schema import (
    AnalysisLabel,
    DatasetSample,
    ServedMassSource,
    Split,
)


@dataclass
class Finding:
    """점검 결과 한 줄."""

    level: str
    sample_id: str | None
    message: str

    def render(self) -> str:
        where = f"[{self.sample_id}] " if self.sample_id else ""
        return f"{self.level}: {where}{self.message}"


@dataclass
class ValidationReport:
    errors: list[Finding] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)

    def error(self, message: str, sample_id: str | None = None) -> None:
        self.errors.append(Finding("ERROR", sample_id, message))

    def warn(self, message: str, sample_id: str | None = None) -> None:
        self.warnings.append(Finding("WARN", sample_id, message))

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [finding.render() for finding in (*self.errors, *self.warnings)]
        return "\n".join(lines)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_unique_sample_ids(
    samples: list[DatasetSample], report: ValidationReport
) -> None:
    counts = Counter(sample.sample_id for sample in samples)
    for sample_id, count in sorted(counts.items()):
        if count > 1:
            report.error(f"같은 sample_id가 {count}번 나옵니다.", sample_id)


def _check_label_version(
    samples: list[DatasetSample], report: ValidationReport
) -> None:
    versions = {sample.label_version for sample in samples}
    if len(versions) > 1:
        # 정의가 다른 정답을 한 데 섞으면 평가 숫자의 의미가 사라진다.
        report.error(
            "정답 정의 버전이 섞여 있습니다: " + ", ".join(sorted(versions))
        )


def _check_files(
    samples: list[DatasetSample], root: Path, report: ValidationReport
) -> dict[str, str]:
    """파일을 열어 보고 해시를 구한다. 반환값은 sample_id별 해시."""
    hashes: dict[str, str] = {}
    for sample in samples:
        path = root / sample.image_path
        if not path.is_file():
            report.error(f"이미지가 없습니다: {sample.image_path}", sample.sample_id)
            continue

        try:
            with Image.open(path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            report.error(
                f"이미지를 열 수 없습니다: {sample.image_path}", sample.sample_id
            )
            continue

        digest = sha256_of(path)
        hashes[sample.sample_id] = digest
        if sample.image_sha256 and sample.image_sha256 != digest:
            report.error("기록된 해시가 실제 파일과 다릅니다.", sample.sample_id)

        # 파일 이름은 sample_id를 쓰기로 했다. 원본 파일명을 그대로 두면
        # 촬영 기기가 붙인 이름이나 사람 이름이 섞여 들어온다.
        if path.stem != sample.sample_id:
            report.warn(
                f"파일 이름이 sample_id와 다릅니다: {path.name}", sample.sample_id
            )
    return hashes


def _check_duplicates(hashes: dict[str, str], report: ValidationReport) -> None:
    by_digest: dict[str, list[str]] = defaultdict(list)
    for sample_id, digest in hashes.items():
        by_digest[digest].append(sample_id)
    for digest, sample_ids in sorted(by_digest.items()):
        if len(sample_ids) > 1:
            # 같은 사진이 여러 표본으로 들어오면 분할이 새고 평가가 부풀려진다.
            report.error(
                "같은 이미지가 여러 표본에 있습니다: " + ", ".join(sorted(sample_ids))
                + f" (sha256 {digest[:12]}…)"
            )


def group_key_of(sample: DatasetSample) -> str:
    """이 표본이 속한 분할 단위.

    같은 사람이 찍은 사진은 배경도 식판도 습관도 닮아 있다. 그것들이 학습과
    평가에 나뉘어 들어가면 모델이 사람을 외운 것을 일반화로 착각하게 된다.
    참가자 키가 없으면 촬영 회차로 묶는다.
    """
    participant, session = sample.group_key_candidates
    return f"participant:{participant}" if participant else f"session:{session}"


def _check_split_leakage(
    samples: list[DatasetSample], report: ValidationReport
) -> None:
    assigned = [sample for sample in samples if sample.split is not None]
    if not assigned:
        return

    splits_by_group: dict[str, set[Split]] = defaultdict(set)
    for sample in assigned:
        splits_by_group[group_key_of(sample)].add(sample.split)  # type: ignore[arg-type]

    for group, splits in sorted(splits_by_group.items()):
        if len(splits) > 1:
            report.error(
                f"같은 그룹이 여러 분할에 있습니다: {group} → "
                + ", ".join(sorted(split.value for split in splits))
            )

    if len(assigned) != len(samples):
        report.warn(
            f"분할이 없는 표본이 {len(samples) - len(assigned)}건 있습니다."
        )


def _check_benchmark_eligibility(
    samples: list[DatasetSample], report: ValidationReport
) -> None:
    visual = [
        sample.sample_id
        for sample in samples
        if sample.analysis_label is AnalysisLabel.APPROVED
        and sample.served_mass_source is ServedMassSource.VISUAL_ESTIMATE
    ]
    if visual:
        # 눈대중은 정답이 아니다. 지우라는 뜻이 아니라, 주 평가 집합에서
        # 빠진다는 사실을 분명히 해 둔다.
        report.warn(
            f"제공량을 눈으로 추정한 표본 {len(visual)}건은 주 평가 집합에서 제외됩니다."
        )

    primary = [sample for sample in samples if sample.is_primary_benchmark]
    if not primary:
        report.warn("주 평가 집합에 넣을 수 있는 표본이 없습니다.")


def validate_dataset(samples: list[DatasetSample], root: Path) -> ValidationReport:
    """전부 본 뒤 한 번에 알려 준다.

    첫 오류에서 멈추지 않는다. 데이터를 고치는 사람은 한 번에 다 보고 싶다.
    """
    report = ValidationReport()
    _check_unique_sample_ids(samples, report)
    _check_label_version(samples, report)
    hashes = _check_files(samples, root, report)
    _check_duplicates(hashes, report)
    _check_split_leakage(samples, report)
    _check_benchmark_eligibility(samples, report)
    return report
