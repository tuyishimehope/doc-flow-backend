"""Endpoint tests for POST /api/v1/chat. No database, no network: the LLM is a fake."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.v1.chat.router import get_llm_picker
from app.main import app
from app.prompts import load_prompt
from app.service.auth.auth import get_current_user
from app.service.llm_client.llm_client import (
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
    Message,
)

URL = "/api/v1/chat"


class FakeLLM:
    def __init__(self, text: str = "Paris.", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[list[Message]] = []

    async def complete(self, messages: list[Message]) -> LLMResponse:
        self.calls.append(messages)
        if self.error:
            raise self.error
        return LLMResponse(
            text=self.text,
            model="fake/model",
            input_tokens=10,
            output_tokens=2,
            latency_ms=5,
            attempts=1,
        )


class FakeUser:
    id = 1


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
async def chat_client(fake_llm: FakeLLM):
    app.dependency_overrides[get_llm_picker] = lambda: (lambda model: fake_llm)
    app.dependency_overrides[get_current_user] = lambda: FakeUser()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_returns_the_answer(chat_client: AsyncClient):
    response = await chat_client.post(URL, json={"message": "Capital of France?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "Paris.", "model": "fake/model"}


@pytest.mark.anyio
async def test_sends_system_prompt_then_user_message(chat_client: AsyncClient, fake_llm: FakeLLM):
    await chat_client.post(URL, json={"message": "Capital of France?"})
    assert fake_llm.calls == [
        [
            {"role": "system", "content": load_prompt("chat_v1")},
            {"role": "user", "content": "Capital of France?"},
        ]
    ]


@pytest.mark.anyio
async def test_message_is_trimmed_before_it_reaches_the_llm(
    chat_client: AsyncClient, fake_llm: FakeLLM
):
    await chat_client.post(URL, json={"message": "   hello \n"})
    assert fake_llm.calls[0][-1]["content"] == "hello"


@pytest.mark.anyio
async def test_accepts_exactly_4000_characters(chat_client: AsyncClient):
    response = await chat_client.post(URL, json={"message": "x" * 4000})
    assert response.status_code == 200


@pytest.mark.anyio
async def test_length_limit_applies_after_trimming(chat_client: AsyncClient):
    response = await chat_client.post(URL, json={"message": "  " + "x" * 4000 + "  "})
    assert response.status_code == 200


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"message": ""},
        {"message": "   \n\t "},
        {"message": "x" * 4001},
        {"message": 123},
        {"message": None},
        {"message": "hi", "unexpected": True},
    ],
    ids=["missing", "empty", "whitespace", "too-long", "not-a-string", "null", "extra-field"],
)
async def test_invalid_bodies_are_rejected_before_calling_the_llm(
    chat_client: AsyncClient, fake_llm: FakeLLM, body: dict
):
    response = await chat_client.post(URL, json=body)
    assert response.status_code == 422
    assert fake_llm.calls == []


@pytest.mark.anyio
async def test_message_must_be_in_the_json_body_not_the_query_string(
    chat_client: AsyncClient, fake_llm: FakeLLM
):
    response = await chat_client.post(f"{URL}?message=hi")
    assert response.status_code == 422
    assert fake_llm.calls == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (LLMTimeoutError("secret internal detail"), 504),
        (LLMRateLimitError("secret internal detail", retry_after=3), 503),
        (LLMUnavailableError("secret internal detail"), 503),
        (LLMError("secret internal detail"), 502),
    ],
    ids=["timeout", "rate-limited", "unavailable", "other"],
)
async def test_llm_failures_become_clear_errors_without_leaking_details(
    chat_client: AsyncClient, fake_llm: FakeLLM, error: Exception, expected_status: int
):
    fake_llm.error = error
    response = await chat_client.post(URL, json={"message": "hi"})
    assert response.status_code == expected_status
    assert "secret" not in response.text
    assert response.json()["detail"]


@pytest.mark.anyio
async def test_rate_limited_response_tells_the_client_when_to_retry(
    chat_client: AsyncClient, fake_llm: FakeLLM
):
    fake_llm.error = LLMRateLimitError("provider 429", retry_after=2.2)
    response = await chat_client.post(URL, json={"message": "hi"})
    assert response.headers["retry-after"] == "3"


@pytest.mark.anyio
async def test_requires_authentication(fake_llm: FakeLLM):
    app.dependency_overrides[get_llm_picker] = lambda: (lambda model: fake_llm)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(URL, json={"message": "hi"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 401
    assert fake_llm.calls == []
