"""식수 예측 흐름.

요청 → 요일 계산 → 과거 feature(예측일 이전 기록만) → 모델 입력 행 →
metadata feature 순서 검증 → Pipeline.predict() → 0 이상 정수.

아무것도 저장하지 않는다. 예측 이력의 보관은 서비스 백엔드가 맡는다.
"""

from __future__ import annotations

import datetime as dt
import logging
import math

from babplanner_ai.config import AppEnv, Settings
from babplanner_ai.demand_forecast.artifact import ArtifactError, ModelArtifactLoader
from babplanner_ai.demand_forecast.features import (
    BUILDABLE_FEATURES,
    FeatureSchemaError,
    build_feature_row,
    order_for_model,
)
from babplanner_ai.demand_forecast.history import (
    DemandHistoryRepository,
    FutureRecordError,
    HistoryFeatures,
    HistoryUnavailableError,
    SyntheticDemandHistoryRepository,
    compute_history_features,
)
from babplanner_ai.demand_forecast.models import (
    DISCLAIMER_REAL,
    DISCLAIMER_SYNTHETIC,
    DemandForecastRequest,
    DemandForecastResponse,
    DemandForecastStatus,
    ForecastConditions,
    ForecastModelInfo,
    ForecastReference,
)

log = logging.getLogger(__name__)

WEEKDAY_NAMES = ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")


class ForecastUnavailableError(RuntimeError):
    """예측을 수행할 수 없다(모델·이력·추론 문제). 503으로 나간다."""


def previous_weekday(day: dt.date) -> dt.date:
    day -= dt.timedelta(days=1)
    while day.weekday() >= 5:
        day -= dt.timedelta(days=1)
    return day


def clamp_count(raw: float) -> int:
    """반올림(0.5는 올림)하고 0 아래는 0으로 막는다. 음수 식수는 없다."""
    return max(0, math.floor(raw + 0.5))


class DemandForecastService:
    def __init__(self, loader: ModelArtifactLoader, history: DemandHistoryRepository) -> None:
        self._loader = loader
        self._history = history

    def warm_up(self) -> None:
        """기동할 때 한 번 로드해 둔다. 실패해도 서버는 뜬다."""
        self._loader.get()
        try:
            self._history.check_available()
        except HistoryUnavailableError as error:
            log.warning("식수 이력을 쓸 수 없습니다. reason=%s", error)

    def status(self) -> DemandForecastStatus:
        state = self._loader.get()
        try:
            self._history.check_available()
            history_ok, history_reason = True, None
        except HistoryUnavailableError as error:
            history_ok, history_reason = False, str(error)

        model = state.model
        warnings = list(model.warnings) if model else []
        return DemandForecastStatus(
            available=model is not None and history_ok,
            modelLoaded=model is not None,
            historyAvailable=history_ok,
            modelName=model.metadata.model_name if model else None,
            modelVersion=model.version if model else None,
            target=model.metadata.target if model else None,
            syntheticData=model.metadata.synthetic_data if model else None,
            trainingPeriod=list(model.metadata.training_period)
            if model and model.metadata.training_period
            else None,
            reason=state.reason or history_reason,
            warnings=warnings or None,
        )

    def predict(self, request: DemandForecastRequest) -> DemandForecastResponse:
        state = self._loader.get()
        if state.model is None:
            raise ForecastUnavailableError(state.reason or "model unavailable")
        model = state.model
        target = request.date

        try:
            meals = self._history.meals_before(target)
            cctv = self._history.cctv_before(target)
        except HistoryUnavailableError as error:
            raise ForecastUnavailableError(str(error)) from error
        try:
            history = compute_history_features(meals, cctv, target)
        except FutureRecordError as error:
            # 저장소가 예측일 이후 기록을 넘겼다. 걸러서 쓰지 않고 멈춘다.
            log.error("이력 저장소가 예측일 이후 기록을 돌려줬습니다. target=%s", target)
            raise ForecastUnavailableError("history source returned records on or after the forecast date") from error
        # InsufficientHistoryError 는 요청 날짜의 문제라 그대로 올린다(422).

        row = build_feature_row(request, history)
        try:
            ordered = order_for_model(row, model.metadata.features)
            raw = model.predictor.predict_one(ordered, model.metadata.features)
        except (FeatureSchemaError, ArtifactError) as error:
            log.error("모델 입력이 metadata와 맞지 않습니다: %s", error)
            raise ForecastUnavailableError("feature schema mismatch") from error
        except Exception as error:  # noqa: BLE001 - 추론 실패를 숫자로 위장하지 않는다
            log.exception("식수 예측 추론에 실패했습니다.")
            raise ForecastUnavailableError("model inference failed") from error
        if not math.isfinite(raw):
            raise ForecastUnavailableError("model returned a non-finite value")

        metadata = model.metadata
        return DemandForecastResponse(
            forecastDate=target,
            predictedMealCount=clamp_count(raw),
            reference=ForecastReference(
                previousDay=round(history.lag_1),
                sameWeekdayPreviousWeek=round(history.lag_7),
                rolling7DayAverage=round(history.rolling_mean_7),
                historyAsOf=history.meal_history_last_date,
            ),
            conditions=ForecastConditions(
                weekday=WEEKDAY_NAMES[target.weekday()],
                semester=request.semester,
                semesterWeek=request.semesterWeek,
                examPeriod=request.examPeriod,
                holiday=request.holiday,
                campusEvent=request.campusEvent,
                menuCategory=request.menuCategory,
                mainMenu=request.mainMenu,
                menuPopularity=request.menuPopularity,
                specialMenu=request.specialMenu,
                temperature=request.temperature,
                rainfall=request.rainfall,
                humidity=request.humidity,
            ),
            model=ForecastModelInfo(
                name=metadata.model_name,
                version=model.version,
                syntheticData=metadata.synthetic_data,
                trainedAt=metadata.trained_at,
            ),
            warnings=self._warnings(request, history, model.metadata),
            disclaimer=DISCLAIMER_SYNTHETIC if metadata.synthetic_data else DISCLAIMER_REAL,
        )

    @staticmethod
    def _warnings(request: DemandForecastRequest, history: HistoryFeatures, metadata) -> list[str]:
        warnings: list[str] = []
        expected = previous_weekday(request.date)
        last = min(history.meal_history_last_date, history.cctv_history_last_date)
        if last < expected:
            warnings.append(
                f"직전 평일({expected}) 기록이 이력에 없어 가장 최근 기록({last})을 기준으로 "
                "과거 feature를 만들었습니다."
            )
        known_menus = metadata.categorical_levels.get("main_menu")
        if known_menus and request.mainMenu not in known_menus:
            warnings.append(
                f"'{request.mainMenu}'은(는) 모델이 학습 중 본 적 없는 메뉴입니다. "
                "메뉴 이름의 효과 없이 예측했습니다."
            )
        if metadata.training_period and request.date <= metadata.training_period[1]:
            warnings.append("예측일이 모델 학습 데이터 기간 안에 있습니다. 성능 근거로 쓰지 마십시오.")
        return warnings


def build_demand_forecast_service(settings: Settings) -> DemandForecastService:
    loader = ModelArtifactLoader(
        settings.demand_forecast_model_path,
        settings.demand_forecast_metadata_path,
        buildable_features=BUILDABLE_FEATURES,
        allow_synthetic=settings.app_env is not AppEnv.PRODUCTION,
    )
    history = SyntheticDemandHistoryRepository(
        settings.demand_history_path, settings.demand_cctv_log_path
    )
    return DemandForecastService(loader, history)
