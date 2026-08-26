"""수집 상태 점검.

숫자만 보여준다. "0이 절반을 넘으면 실패" 같은 판단을 여기에 박지 않는다.
실제 분포를 처음 보는 사람이 그것을 보고 정해야 한다. 지금 정하면 근거 없는
숫자가 기준으로 굳는다.

참가자 목록처럼 하나하나 식별되는 값은 출력하지 않는다. 몇 명인지만 센다.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from babplanner_ai.collection.journal import CollectionStore
from babplanner_ai.collection.measurement import compute_mass, repeat_stats
from babplanner_ai.collection.models import MeasurementError
from babplanner_ai.dataset.schema import AnalysisLabel, ServedMassSource

#: 0이나 1에 얼마나 붙어 있는지 셀 때 쓰는 폭. 판정 기준이 아니라 분포를
#: 보여주기 위한 눈금이다.
EDGE_WIDTH = 0.05


@dataclass
class Distribution:
    count: int
    minimum: float
    maximum: float
    mean: float
    median: float

    @classmethod
    def of(cls, values: list[float]) -> "Distribution | None":
        if not values:
            return None
        return cls(
            count=len(values),
            minimum=min(values),
            maximum=max(values),
            mean=statistics.fmean(values),
            median=statistics.median(values),
        )

    def render(self, indent: str = "  ") -> str:
        return (
            f"{indent}건수   {self.count}\n"
            f"{indent}최소   {self.minimum:.4g}\n"
            f"{indent}최대   {self.maximum:.4g}\n"
            f"{indent}평균   {self.mean:.4g}\n"
            f"{indent}중앙값 {self.median:.4g}"
        )


@dataclass
class QcReport:
    session_id: str
    scale_resolution_g: float
    total_raw_records: int = 0
    effective_samples: int = 0
    corrected_samples: int = 0
    labels: Counter = field(default_factory=Counter)
    primary_candidates: int = 0
    invalid_measurements: list[tuple[str, str]] = field(default_factory=list)
    duplicate_images: list[list[str]] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    served: Distribution | None = None
    leftover: Distribution | None = None
    rate: Distribution | None = None
    near_zero: int = 0
    near_one: int = 0
    menu_count: int = 0
    participant_count: int = 0
    capture_session_count: int = 0
    tare_profiles: list[tuple[str, int, float, float]] = field(default_factory=list)
    repeat_spreads: list[tuple[str, float, float]] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"수집 회차        {self.session_id}",
            f"저울 눈금        {self.scale_resolution_g} g",
            "",
            f"원시 기록        {self.total_raw_records}",
            f"유효 표본        {self.effective_samples}",
            f"수정된 표본      {self.corrected_samples}",
        ]
        for label in sorted(self.labels):
            lines.append(f"  {label:<14} {self.labels[label]}")
        lines.append(f"주 평가 후보     {self.primary_candidates}")

        lines.append("")
        lines.append(f"측정 오류        {len(self.invalid_measurements)}")
        for sample_id, message in self.invalid_measurements:
            lines.append(f"  {sample_id}  {message}")

        lines.append(f"중복 이미지      {len(self.duplicate_images)}")
        for group in self.duplicate_images:
            lines.append("  " + ", ".join(group))

        lines.append(f"없는 파일        {len(self.missing_files)}")
        for sample_id in self.missing_files:
            lines.append(f"  {sample_id}")

        for title, distribution in (
            ("제공량(g)", self.served),
            ("잔반량(g)", self.leftover),
            ("잔반률", self.rate),
        ):
            lines.append("")
            lines.append(title)
            lines.append(distribution.render() if distribution else "  없음")

        if self.rate:
            lines.append("")
            lines.append(f"0 근처(≤{EDGE_WIDTH})  {self.near_zero}")
            lines.append(f"1 근처(≥{1 - EDGE_WIDTH})  {self.near_one}")

        lines.extend(
            [
                "",
                f"메뉴             {self.menu_count}",
                f"참가자 그룹      {self.participant_count}",
                f"촬영 회차        {self.capture_session_count}",
            ]
        )

        lines.append("")
        lines.append(f"tare 프로파일    {len(self.tare_profiles)}")
        for profile_id, count, representative, spread in self.tare_profiles:
            lines.append(
                f"  {profile_id}  {count}회  대표 {representative:.4g} g  "
                f"편차 {spread:.4g} g"
            )

        if self.repeat_spreads:
            lines.append("")
            lines.append("반복 계량 편차(g)")
            for sample_id, pre_spread, post_spread in self.repeat_spreads:
                lines.append(f"  {sample_id}  배식 {pre_spread:.4g}  잔반 {post_spread:.4g}")

        return "\n".join(lines)


def build_report(store: CollectionStore) -> QcReport:
    session = store.read_session()
    tare_profiles = store.read_tare_profiles()
    all_records = store.read_records()
    latest = store.latest_records()

    report = QcReport(
        session_id=session.collection_session_id,
        scale_resolution_g=session.scale_resolution_g,
        total_raw_records=len(all_records),
        effective_samples=len(latest),
        corrected_samples=sum(1 for record in latest if record.revision > 1),
    )

    served_values: list[float] = []
    leftover_values: list[float] = []
    rate_values: list[float] = []
    by_hash: dict[str, list[str]] = defaultdict(list)

    for record in latest:
        report.labels[record.analysis_label.value] += 1

        if record.sanitized_image_sha256:
            by_hash[record.sanitized_image_sha256].append(record.sample_id)
        if record.sanitized_image_path:
            if not (store.session_dir / record.sanitized_image_path).is_file():
                report.missing_files.append(record.sample_id)

        pre = repeat_stats(record.pre_meal_measurements_g)
        post = repeat_stats(record.post_meal_measurements_g)
        if pre or post:
            report.repeat_spreads.append(
                (
                    record.sample_id,
                    pre.spread if pre else 0.0,
                    post.spread if post else 0.0,
                )
            )

        if record.analysis_label is not AnalysisLabel.APPROVED:
            continue

        try:
            mass = compute_mass(record, tare_profiles.get(record.tare_profile_id or ""))
        except MeasurementError as error:
            report.invalid_measurements.append((record.sample_id, str(error)))
            continue

        served_values.append(mass.served_edible_mass_g)
        leftover_values.append(mass.leftover_edible_mass_g)
        rate_values.append(mass.waste_rate)
        if record.served_mass_source is ServedMassSource.MEASURED:
            report.primary_candidates += 1

    report.duplicate_images = [
        sorted(sample_ids) for sample_ids in by_hash.values() if len(sample_ids) > 1
    ]
    report.served = Distribution.of(served_values)
    report.leftover = Distribution.of(leftover_values)
    report.rate = Distribution.of(rate_values)
    report.near_zero = sum(1 for rate in rate_values if rate <= EDGE_WIDTH)
    report.near_one = sum(1 for rate in rate_values if rate >= 1 - EDGE_WIDTH)

    report.menu_count = len({record.menu_id for record in latest})
    report.participant_count = len(
        {record.participant_group_id for record in latest if record.participant_group_id}
    )
    report.capture_session_count = len({record.capture_session_id for record in latest})
    report.tare_profiles = [
        (
            profile.tare_profile_id,
            profile.measurement_count,
            profile.tare_weight_g,
            profile.spread_g,
        )
        for profile in sorted(tare_profiles.values(), key=lambda p: p.tare_profile_id)
    ]
    return report
