# babplanner-ai

밥플래너 잔반 이미지 분석 서비스.

서비스 백엔드(Spring)만 호출한다. 사용자 앱이 직접 호출하는 endpoint는 없다.

## 현재 상태

실제 비전 모델은 아직 없다. `FixtureWasteEstimator`는 이미지를 판독하지 않고
디코딩 가능 여부만 확인한 뒤 고정값을 돌려주는 **개발 fixture**다. 이 값을
분석 결과로 읽으면 안 된다.

교체 지점은 `WasteEstimator` 하나다. HTTP 계층과 인증은 그대로 두고 구현만
바꾸면 된다.

## 실행

환경 변수는 셸에만 둔다. 파일로 남기지 않는다.

    APP_ENV=local
    WASTE_MODEL_MODE=fixture
    AI_SERVICE_TOKEN=<secret>

    uvicorn babplanner_ai.main:app --host 127.0.0.1 --port 8001

`WASTE_MODEL_MODE`에 기본값은 없다. 운영에서 `fixture`를 고르면 기동에
실패한다. 실제 모델 어댑터가 없는 동안에는 그것이 정상이다.

## 테스트

    pytest
