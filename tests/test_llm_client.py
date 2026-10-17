"""Unit tests for the OpenAI-compatible client: retries, timeouts, logging. No network."""

import asyncio
import logging
from types import SimpleNamespace

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from app.service.llm_client.llm_client import (
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.service.llm_client.openai_compatible import OpenAICompatibleClient

REQUEST = httpx.Request("POST", "https://openrouter.test/chat/completions")
MESSAGES = [{"role": "user", "content": "secret user text"}]


def status_error(cls, code: int, headers: dict | None = None):
    response = httpx.Response(code, request=REQUEST, headers=headers or {})
    return cls("boom", response=response, body=None)


def completion(text: str | None = "Paris", model: str = "vendor/real-model"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3),
        model=model,
    )


class FakeOpenAI:
    """Plays back a script: an exception is raised, anything else is returned."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = 0
        self.last_kwargs: dict = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return await item()
        return item


def make_client(
    fake: FakeOpenAI,
    *,
    max_retries: int = 2,
    total_timeout: float = 5,
    stream_timeout: float = 120,
    sleeps: list | None = None,
):
    async def fake_sleep(seconds: float) -> None:
        if sleeps is not None:
            sleeps.append(seconds)

    return OpenAICompatibleClient(
        provider="openrouter",
        client=fake,  # type: ignore[arg-type]
        model="configured/model",
        timeout=1,
        total_timeout=total_timeout,
        max_retries=max_retries,
        max_output_tokens=100,
        stream_timeout=stream_timeout,
        sleep=fake_sleep,
    )


@pytest.mark.anyio
async def test_success_reports_text_model_tokens_and_attempts():
    client = make_client(FakeOpenAI(completion()))
    result = await client.complete(MESSAGES)
    assert result.text == "Paris"
    assert result.model == "vendor/real-model"
    assert (result.input_tokens, result.output_tokens, result.attempts) == (11, 3, 1)


@pytest.mark.anyio
async def test_answer_text_is_trimmed():
    result = await make_client(FakeOpenAI(completion(text="\n\n Paris.  \n"))).complete(MESSAGES)
    assert result.text == "Paris."


@pytest.mark.anyio
async def test_rate_limit_is_retried_and_honours_retry_after():
    fake = FakeOpenAI(
        status_error(RateLimitError, 429, {"retry-after": "1"}),
        status_error(RateLimitError, 429, {"retry-after": "2"}),
        completion(),
    )
    sleeps: list[float] = []
    result = await make_client(fake, sleeps=sleeps).complete(MESSAGES)
    assert result.attempts == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.anyio
async def test_rate_limit_gives_up_after_max_retries_with_a_clear_error():
    fake = FakeOpenAI(*[status_error(RateLimitError, 429, {"retry-after": "1"}) for _ in range(3)])
    with pytest.raises(LLMRateLimitError) as caught:
        await make_client(fake, max_retries=2).complete(MESSAGES)
    assert fake.calls == 3  # first try + 2 retries
    assert caught.value.retry_after == 1.0


@pytest.mark.anyio
async def test_does_not_wait_when_the_provider_asks_for_a_very_long_pause():
    fake = FakeOpenAI(status_error(RateLimitError, 429, {"retry-after": "120"}))
    sleeps: list[float] = []
    with pytest.raises(LLMRateLimitError):
        await make_client(fake, sleeps=sleeps).complete(MESSAGES)
    assert fake.calls == 1
    assert sleeps == []


@pytest.mark.anyio
async def test_timeouts_are_retried_then_reported_as_timeout():
    fake = FakeOpenAI(*[APITimeoutError(request=REQUEST) for _ in range(3)])
    with pytest.raises(LLMTimeoutError):
        await make_client(fake).complete(MESSAGES)
    assert fake.calls == 3


@pytest.mark.anyio
async def test_connection_errors_and_5xx_are_retried_then_reported_as_unavailable():
    fake = FakeOpenAI(APIConnectionError(request=REQUEST), status_error(InternalServerError, 503), completion())
    result = await make_client(fake).complete(MESSAGES)
    assert result.attempts == 3

    fake = FakeOpenAI(*[status_error(InternalServerError, 500) for _ in range(3)])
    with pytest.raises(LLMUnavailableError):
        await make_client(fake).complete(MESSAGES)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (status_error(AuthenticationError, 401), LLMUnavailableError),
        (status_error(BadRequestError, 400), LLMError),
    ],
    ids=["bad-key", "bad-request"],
)
async def test_client_errors_are_not_retried(error, expected):
    fake = FakeOpenAI(error)
    with pytest.raises(expected):
        await make_client(fake).complete(MESSAGES)
    assert fake.calls == 1


@pytest.mark.anyio
@pytest.mark.parametrize("empty", [None, "", "   "])
async def test_empty_answer_is_an_error(empty):
    with pytest.raises(LLMError):
        await make_client(FakeOpenAI(completion(text=empty))).complete(MESSAGES)


@pytest.mark.anyio
async def test_total_time_budget_stops_a_hanging_provider():
    async def hang():
        await asyncio.sleep(10)

    with pytest.raises(LLMTimeoutError):
        await make_client(FakeOpenAI(hang), total_timeout=0.05).complete(MESSAGES)


@pytest.mark.anyio
async def test_every_call_is_logged_once_without_the_message_text(caplog):
    caplog.set_level(logging.INFO, logger="app.service.llm_client.openai_compatible")
    await make_client(FakeOpenAI(completion())).complete(MESSAGES)

    records = [r for r in caplog.records if r.getMessage() == "LLM call finished"]
    assert len(records) == 1
    record = records[0]
    assert (record.provider, record.status, record.attempts) == ("openrouter", "ok", 1)
    assert (record.model, record.input_tokens, record.output_tokens) == ("vendor/real-model", 11, 3)
    assert isinstance(record.latency_ms, int)
    assert all("secret user text" not in str(vars(r)) for r in caplog.records)


@pytest.mark.anyio
async def test_failed_calls_are_logged_with_their_status(caplog):
    caplog.set_level(logging.INFO, logger="app.service.llm_client.openai_compatible")
    fake = FakeOpenAI(*[APITimeoutError(request=REQUEST) for _ in range(3)])
    with pytest.raises(LLMTimeoutError):
        await make_client(fake).complete(MESSAGES)
    final = [r for r in caplog.records if r.getMessage() == "LLM call finished"]
    assert len(final) == 1
    assert (final[0].status, final[0].attempts) == ("timeout", 3)


@pytest.mark.anyio
async def test_the_answer_says_which_provider_gave_it():
    result = await make_client(FakeOpenAI(completion())).complete(MESSAGES)
    assert result.provider == "openrouter"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("cls", "code", "expected"),
    [
        (BadRequestError, 400, LLMError),  # Gemini's answer to an invalid key, Groq's to a retired model
        (BadRequestError, 404, LLMError),  # the model was retired
        (BadRequestError, 422, LLMError),
        (AuthenticationError, 401, LLMUnavailableError),  # our key is wrong
    ],
)
async def test_other_client_errors_are_not_retried_but_leave_room_for_a_fallback(cls, code, expected):
    fake = FakeOpenAI(status_error(cls, code))
    with pytest.raises(expected):
        await make_client(fake).complete(MESSAGES)
    assert fake.calls == 1  # retrying the same provider cannot help
