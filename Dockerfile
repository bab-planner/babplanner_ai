# syntax=docker/dockerfile:1
#
# 밥플래너 AI 서버(식수 예측) 이미지.
#
# 식수 예측 PoC artifact(모델 + 합성 이력)는 git에 없다. 이미지를 **만들 때 한 번**
# 합성 데이터를 생성하고 학습해서 runtime 이미지에 넣는다. 서버가 기동할 때 학습하지
# 않는다. 생성은 seed가 고정이라 같은 소스에서는 같은 artifact가 나온다.
#
#   docker build -t babplanner-ai .
#
# 실행 시 환경 변수(이미지에 넣지 않는다. 비밀은 특히 그렇다):
#   APP_ENV                운영은 production
#   AI_SERVICE_TOKEN       32자 이상. Spring이 X-Service-Token으로 보내는 값과 같아야 한다
#   DEMAND_FORECAST_MODE   poc  (합성 PoC 모델은 이것을 명시해야만 서비스된다. 기본 strict)
#
#   docker run --rm -p 8000:8000 -e APP_ENV=production -e AI_SERVICE_TOKEN=... \
#       -e DEMAND_FORECAST_MODE=poc babplanner-ai
#
# 실데이터 모델이 생기면 build 단계(generate/train)를 실제 artifact 복사로 바꾼다.

# ---------------------------------------------------------------- 1. artifact 생성
FROM python:3.12-slim AS builder

# lightgbm이 OpenMP 런타임(libgomp)을 쓴다.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    MPLBACKEND=Agg

WORKDIR /build
COPY pyproject.toml ./
COPY src ./src
COPY research/demand_forecasting ./research/demand_forecasting

# 서비스 + 추론용 ML 런타임(버전 고정). 학습 그래프용 matplotlib만 더한다.
RUN pip install ".[demand-model]" matplotlib

RUN python research/demand_forecasting/src/generate_synthetic_data.py \
    && python research/demand_forecasting/src/train.py

# ---------------------------------------------------------------- 2. runtime
FROM python:3.12-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --uid 10001 aiuser

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
# 학습 라이브러리(matplotlib 등)는 넣지 않는다. 추론에 필요한 고정 버전만 설치한다.
RUN pip install ".[demand-model]"

# build 단계가 만든 artifact만 가져온다(모델 + metadata + 합성 이력).
COPY --from=builder /build/research/demand_forecasting/models /app/artifacts/models
COPY --from=builder /build/research/demand_forecasting/data /app/artifacts/data

ENV DEMAND_FORECAST_MODEL_PATH=/app/artifacts/models/meal_demand_model.joblib \
    DEMAND_FORECAST_METADATA_PATH=/app/artifacts/models/metadata.json \
    DEMAND_HISTORY_PATH=/app/artifacts/data/synthetic_meal_demand.csv \
    DEMAND_CCTV_LOG_PATH=/app/artifacts/data/synthetic_cctv_daily.csv

USER aiuser
EXPOSE 8000

# /health는 토큰 없이 200이다. 모델 사용 가능 여부는 status가 답한다.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "babplanner_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
