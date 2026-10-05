"""서비스 토큰."""

from __future__ import annotations

import inspect

from fastapi.testclient import TestClient

from babplanner_ai import security
from babplanner_ai.security import SERVICE_TOKEN_HEADER, tokens_match
from tests.conftest import TEST_SERVICE_TOKEN

# 서비스 토큰을 요구하는 내부 endpoint 중 하나. 모델이 없어도 200으로 상태를 답한다.
PATH = "/internal/v1/ai/demand-forecast/status"


def _post(client: TestClient, token: str | None) -> object:
    headers = {} if token is None else {SERVICE_TOKEN_HEADER: token}
    return client.get(PATH, headers=headers)


def test_missing_token_is_unauthorized(client: TestClient) -> None:
    assert _post(client, None).status_code == 401


def test_wrong_token_is_unauthorized(client: TestClient) -> None:
    assert _post(client, "wrong-token-that-is-long-enough-0123456789").status_code == 401


def test_correct_token_passes(client: TestClient) -> None:
    assert _post(client, TEST_SERVICE_TOKEN).status_code == 200


def test_empty_token_is_unauthorized(client: TestClient) -> None:
    assert _post(client, "").status_code == 401


def test_prefix_of_the_real_token_is_rejected(client: TestClient) -> None:
    # 앞자리만 맞아도 통과하면 한 글자씩 맞춰 나갈 수 있다.
    assert _post(client, TEST_SERVICE_TOKEN[:-1]).status_code == 401


def test_missing_and_wrong_look_the_same(client: TestClient) -> None:
    missing = _post(client, None)
    wrong = _post(client, "wrong-token-that-is-long-enough-0123456789")

    assert missing.status_code == wrong.status_code
    assert missing.json() == wrong.json()


def test_response_does_not_leak_the_expected_token(client: TestClient) -> None:
    body = _post(client, "wrong-token-that-is-long-enough-0123456789").text

    assert TEST_SERVICE_TOKEN not in body


def test_comparison_is_constant_time() -> None:
    source = inspect.getsource(security)

    assert "hmac.compare_digest" in source
    # == 비교로 되돌아가면 응답 시간으로 토큰이 새어 나간다.
    assert "presented == expected" not in source


def test_tokens_match_handles_missing_value() -> None:
    assert tokens_match(None, TEST_SERVICE_TOKEN) is False
    assert tokens_match(TEST_SERVICE_TOKEN, TEST_SERVICE_TOKEN) is True
    assert tokens_match("other", TEST_SERVICE_TOKEN) is False


def test_token_is_never_logged() -> None:
    source = inspect.getsource(security)

    for forbidden in ("print(", "log.info", "logger.info", "request.headers"):
        assert forbidden not in source
