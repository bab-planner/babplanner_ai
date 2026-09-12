"""다음 날 중식 식수 예측 모델 학습·평가 (합성데이터 PoC).

실행:
    python src/train.py

하는 일:
    1. 날짜순 분할: 앞 70% train / 다음 15% validation / 마지막 15% test
    2. baseline(lag_7, rolling_mean_7)과 RandomForest, gradient boosting 비교
    3. validation MAE로 최종 모델을 고르고, train+validation으로 다시 학습해 test 1회 평가
    4. feature group ablation (M1~M4)
    5. 최종 모델을 전체 데이터로 다시 학습해 models/에 저장

**여기서 나오는 성능은 합성데이터 위의 숫자다.** 생성 규칙을 모델이 얼마나
되찾는지를 잰 것이지, 실제 급식소에서의 예측 정확도가 아니다.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_GROUP_LABELS,
    FEATURE_GROUPS,
    MEAL_DATA_PATH,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    SYNTHETIC_NOTICE,
    TARGET,
    TRACK_ROOT,
    load_meal_data,
    regression_metrics,
    temporal_split,
)

SEED = 42
INTERVAL_QUANTILES = (0.10, 0.90)  # predict.py의 참고 범위(80%)


# ---------------------------------------------------------------------------
# 모델 정의
# ---------------------------------------------------------------------------
def resolve_gradient_boosting() -> tuple[str, str]:
    """설치된 것 중 LightGBM → XGBoost → sklearn 순으로 고른다. 새로 설치하지 않는다."""
    try:
        import lightgbm

        return "lightgbm", lightgbm.__version__
    except ImportError:
        pass
    try:
        import xgboost

        return "xgboost", xgboost.__version__
    except ImportError:
        return "hist_gradient_boosting", sklearn.__version__


GBM_NAME, GBM_VERSION = resolve_gradient_boosting()


def make_estimator(name: str):
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.5,
                                     random_state=SEED, n_jobs=-1)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                             min_child_samples=10, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
                             n_jobs=1, deterministic=True, force_row_wise=True, verbose=-1)
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
                            colsample_bytree=0.8, random_state=SEED, n_jobs=1)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=10, random_state=SEED)
    if name == "ridge":
        return Ridge(alpha=1.0)
    raise ValueError(name)


def make_pipeline(name: str, features: list[str]) -> Pipeline:
    categorical = [f for f in features if f in CATEGORICAL_FEATURES]
    numeric = [f for f in features if f not in CATEGORICAL_FEATURES]
    preprocess = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numeric", StandardScaler() if name == "ridge" else "passthrough", numeric),
    ])
    return Pipeline([("preprocess", preprocess), ("model", make_estimator(name))])


def fit_predict(name: str, features: list[str], fit_df: pd.DataFrame, eval_df: pd.DataFrame):
    pipe = make_pipeline(name, features).fit(fit_df[features], fit_df[TARGET])
    return pipe, pipe.predict(eval_df[features])


def metric_row(model: str, kind: str, split: str, part: pd.DataFrame, pred) -> dict:
    m = regression_metrics(part[TARGET], pred)
    normal = (part["holiday"] == 0).to_numpy()
    m_normal = regression_metrics(part[TARGET][normal], np.asarray(pred)[normal])
    return {"model": model, "kind": kind, "split": split, "n": len(part),
            **{k: round(v, 2) for k, v in m.items()},
            "MAE_excl_holiday": round(m_normal["MAE"], 2)}


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------
def print_table(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(table.to_string(index=False))


def plot_test_predictions(test: pd.DataFrame, pred: np.ndarray, model_name: str) -> None:
    from eda import BLUE, INK, MUTED, ORANGE, save, setup_style
    import matplotlib.pyplot as plt

    setup_style()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    # x축은 운영일 순번. 방학 기간을 빈칸으로 늘어놓지 않는다.
    frame = test.reset_index(drop=True).assign(pred=pred, x=lambda d: np.arange(len(d)))
    semester_id = (frame["date"].diff().dt.days > 30).cumsum()
    for i, (_, part) in enumerate(frame.groupby(semester_id)):
        first = i == 0
        ax.plot(part["x"], part[TARGET], color=INK, linewidth=1.3, marker="o", markersize=3,
                label="실제(합성) 식수" if first else None)
        ax.plot(part["x"], part["pred"], color=BLUE, linewidth=1.8,
                label=f"{model_name} 예측" if first else None)
        ax.plot(part["x"], part["lag_7"], color=ORANGE, linewidth=1.1, alpha=0.9,
                label="lag7 baseline" if first else None)
        if not first:
            ax.axvline(part["x"].iloc[0] - 0.5, color=INK, linewidth=0.8)
            ax.annotate("방학 이후", (part["x"].iloc[0] - 0.5, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(4, -4), textcoords="offset points", fontsize=9, color=MUTED, va="top")
    for x in frame.loc[frame["holiday"] == 1, "x"]:
        ax.axvline(x, color=MUTED, linewidth=0.8, alpha=0.6, zorder=0)
    ticks = frame.groupby(frame["date"].dt.to_period("M"))["x"].min()
    ax.set_xticks(ticks.to_numpy(), [str(p) for p in ticks.index])
    ax.set_title("test 구간: 실제 vs 예측 (x축 = 운영일 순서, 회색 세로선 = 공휴일)")
    ax.set_ylabel("meal_count (명)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=3)
    save(fig, OUTPUT_DIR / "test_predictions.png")


def file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
def main() -> None:
    df = load_meal_data()
    splits = temporal_split(df)
    train, val, test = splits["train"], splits["validation"], splits["test"]
    train_val = pd.concat([train, val])

    print(SYNTHETIC_NOTICE)
    print(f"\n데이터: {MEAL_DATA_PATH.relative_to(TRACK_ROOT).as_posix()} ({len(df)}행)")
    for name, part in splits.items():
        print(f"  {name:<10} {len(part):>3}행  {part['date'].min().date()} ~ {part['date'].max().date()}"
              f"  (공휴일 {int(part['holiday'].sum())}일)")
    print(f"gradient boosting 구현: {GBM_NAME} {GBM_VERSION}")

    rows: list[dict] = []
    # --- baseline: 학습 없음 -------------------------------------------------
    baselines = [("lag7_baseline", "lag_7", "baseline"),
                 ("rolling7_baseline", "rolling_mean_7", "baseline"),
                 ("lag1_baseline", "lag_1", "reference")]
    for model_name, column, kind in baselines:
        for split_name, part in [("validation", val), ("test", test)]:
            rows.append(metric_row(model_name, kind, split_name, part, part[column]))

    # --- 학습 모델 -----------------------------------------------------------
    # validation 점수: train으로 학습. test 점수: train+validation으로 다시 학습.
    candidates = ["random_forest", GBM_NAME]
    test_predictions: dict[str, np.ndarray] = {}
    for model_name, kind in [(c, "model") for c in candidates] + [("ridge", "reference")]:
        _, val_pred = fit_predict(model_name, ALL_FEATURES, train, val)
        _, test_pred = fit_predict(model_name, ALL_FEATURES, train_val, test)
        test_predictions[model_name] = test_pred
        rows.append(metric_row(model_name, kind, "validation", val, val_pred))
        rows.append(metric_row(model_name, kind, "test", test, test_pred))

    metrics = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")

    val_mae = metrics[(metrics["split"] == "validation") & metrics["model"].isin(candidates)]
    selected = val_mae.sort_values("MAE").iloc[0]["model"]

    cols = ["model", "kind", "MAE", "RMSE", "MAPE", "bias", "MAE_excl_holiday"]
    print_table(f"[validation] n={len(val)}  (train으로 학습)",
                metrics[metrics["split"] == "validation"][cols])
    print_table(f"[test] n={len(test)}  (train+validation으로 학습, 1회 평가)",
                metrics[metrics["split"] == "test"][cols])
    print("\nkind: baseline=요청된 비교 기준, model=최종 후보, reference=참고용(선택 대상 아님)")
    print("bias = 평균(예측-실제). 양수 = 과대 예측(음식이 남음), 음수 = 과소 예측(사람이 못 먹음)")
    print(f"최종 모델: {selected} (validation MAE 기준, 후보 {candidates})")

    # --- ablation ------------------------------------------------------------
    ablation_rows = []
    for model_name in candidates:
        for group, features in FEATURE_GROUPS.items():
            _, val_pred = fit_predict(model_name, features, train, val)
            _, test_pred = fit_predict(model_name, features, train_val, test)
            for split_name, part, pred in [("validation", val, val_pred), ("test", test, test_pred)]:
                row = metric_row(model_name, "ablation", split_name, part, pred)
                ablation_rows.append({"feature_set": group, "description": FEATURE_GROUP_LABELS[group],
                                      "n_features": len(features), **row})
    ablation = pd.DataFrame(ablation_rows).drop(columns="kind")
    ablation.to_csv(OUTPUT_DIR / "ablation_metrics.csv", index=False, encoding="utf-8-sig")
    view = ablation.pivot_table(index=["feature_set", "description"], columns=["model", "split"],
                                values="MAE").round(2)
    print("\n[ablation] MAE (validation: train으로 학습 / test: train+validation으로 학습)")
    print(view.to_string())
    print("합성데이터에서 설계한 관계를 모델이 feature group별로 활용하는지 보는 점검이다. "
          "실제 feature 효과의 증거가 아니다.")

    # --- test 예측 저장과 그림 -------------------------------------------------
    test_out = test[["date", "weekday", "holiday", "menu_category", "main_menu", TARGET,
                     "lag_7", "rolling_mean_7"]].assign(prediction=np.round(test_predictions[selected], 1))
    test_out.assign(date=test_out["date"].dt.strftime("%Y-%m-%d")).to_csv(
        OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    plot_test_predictions(test, test_predictions[selected], selected)

    residual = test[TARGET].to_numpy() - test_predictions[selected]
    residual_q = np.quantile(residual, INTERVAL_QUANTILES)

    # --- 최종 모델: 전체 데이터로 재학습해 저장 ----------------------------------
    final_pipe = make_pipeline(selected, ALL_FEATURES).fit(df[ALL_FEATURES], df[TARGET])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipe, MODEL_PATH)

    selected_test = metrics[(metrics["model"] == selected) & (metrics["split"] == "test")].iloc[0]
    metadata = {
        "synthetic_data": True,
        "notice": SYNTHETIC_NOTICE,
        "purpose": "시연용 PoC. 실제 운영 정확도를 주장하지 않는다.",
        "model_name": selected,
        "model_class": type(final_pipe.named_steps["model"]).__name__,
        "target": TARGET,
        "prediction_task": "D-1 시점에서 다음 운영일 중식 식수 예측",
        "features": ALL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        # 학습 데이터에 나온 범주값. 서빙 쪽이 sklearn 내부 속성을 들추지 않고
        # "모델이 본 적 없는 메뉴"를 알아챌 수 있게 한다.
        "categorical_levels": {c: sorted(map(str, df[c].unique())) for c in CATEGORICAL_FEATURES},
        "numeric_features": NUMERIC_FEATURES,
        "training_date": dt.datetime.now().isoformat(timespec="seconds"),
        "training_rows": len(df),
        "training_period": [str(df["date"].min().date()), str(df["date"].max().date())],
        "saved_model_fit_on": "전체 합성 데이터(train+validation+test). test 지표는 이 모델이 아니라 "
                              "train+validation으로 학습한 같은 설정의 모델에서 잰 것이다.",
        "split": {name: [str(p["date"].min().date()), str(p["date"].max().date()), len(p)]
                  for name, p in splits.items()},
        "selection": f"validation MAE 최소, 후보 {candidates}",
        "synthetic_test_metrics": {k: float(selected_test[k]) for k in ["MAE", "RMSE", "MAPE", "bias"]},
        "synthetic_test_residual_quantiles": {
            "quantiles": list(INTERVAL_QUANTILES),
            "values": [round(float(v), 1) for v in residual_q],
            "note": f"test {len(test)}행 잔차(실제-예측) 분위수. 참고 범위 표시에만 쓴다.",
        },
        "random_seed": SEED,
        "data_sha256": file_sha256(MEAL_DATA_PATH),
        # 서빙 쪽은 이 값이 모델 파일과 맞을 때만 로드한다. metadata와 모델이 서로
        # 다른 학습 회차에서 오면 feature 목록 검증이 엉뚱한 모델을 보증하게 된다.
        "model_sha256": file_sha256(MODEL_PATH),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, GBM_NAME: GBM_VERSION},
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n저장: {(OUTPUT_DIR / 'model_metrics.csv').relative_to(TRACK_ROOT).as_posix()}")
    print(f"저장: {(OUTPUT_DIR / 'ablation_metrics.csv').relative_to(TRACK_ROOT).as_posix()}")
    print(f"저장: {(OUTPUT_DIR / 'test_predictions.csv').relative_to(TRACK_ROOT).as_posix()}")
    print(f"저장: {MODEL_PATH.relative_to(TRACK_ROOT).as_posix()} ({selected}, 전체 {len(df)}행으로 재학습)")
    print(f"저장: {METADATA_PATH.relative_to(TRACK_ROOT).as_posix()}")


if __name__ == "__main__":
    sys.exit(main())
