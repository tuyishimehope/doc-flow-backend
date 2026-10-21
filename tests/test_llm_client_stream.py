"""Streaming behaviour of the OpenAI-compatible client. No network: the provider is faked."""

import asyncio
import logging
from types import SimpleNamespace

import pytest
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from app.service.llm_client.llm_client import (
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
    StreamDelta,
    StreamDone,
)
from tests.test_llm_client import MESSAGES, REQUEST, FakeOpenAI, make_client, status_error

MODEL = "vendor/real-model"


def piece(text: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        model=MODEL, usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
    )


USAGE_CHUNK = SimpleNamespace(
    model=MODEL, usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3), choices=[]
)


class FakeStream:
    """Behaves like ``openai.AsyncStream``: an async iterator and an async context manager."""

    def __init__(
        self, chunks, *, error: Exception | None = None, delay: float = 0, hang: bool = False
    ):
        self.chunks = list(chunks)
        self.error = error
        self.delay = delay
        self.hang = hang
        self.closed = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk
        if self.error:
            raise self.error
        if self.hang:
            await asyncio.sleep(30)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        self.closed = True


def good_stream() -> FakeStream:
    return FakeStream([piece("Par"), piece("is."), USAGE_CHUNK])


async def collect(client) -> list:
    return [event async for event in client.stream(MESSAGES)]


@pytest.mark.anyio
async def test_yields_pieces_then_a_done_event_and_asks_for_a_stream():
    fake = FakeOpenAI(
        FakeStream(
            [piece(None), piece("\n\n"), piece(" "), piece("Par"), piece("is."), USAGE_CHUNK]
        )
    )
    events = await collect(make_client(fake))

    assert events[:2] == [StreamDelta("Par"), StreamDelta("is.")]  # blank lead-in dropped
    done = events[2]
    assert isinstance(done, StreamDone)
    assert (done.model, done.input_tokens, done.output_tokens, done.attempts) == (MODEL, 11, 3, 1)
    assert done.provider == "openrouter"
    assert isinstance(done.latency_ms, int) and isinstance(done.first_piece_ms, int)
    assert fake.last_kwargs["stream"] is True
    assert fake.last_kwargs["stream_options"] == {"include_usage": True}


@pytest.mark.anyio
async def test_retries_before_the_first_piece():
    sleeps: list[float] = []
    fake = FakeOpenAI(status_error(RateLimitError, 429, {"retry-after": "1"}), good_stream())
    events = await collect(make_client(fake, sleeps=sleeps))
    assert events[-1].attempts == 2
    assert sleeps == [1.0]


@pytest.mark.anyio
async def test_an_error_event_inside_an_open_stream_is_retried_if_nothing_was_sent():
    fake = FakeOpenAI(
        FakeStream([], error=APIError("boom", REQUEST, body=None)),
        good_stream(),
    )
    events = await collect(make_client(fake))
    assert events[-1].attempts == 2


@pytest.mark.anyio
async def test_never_retries_after_text_was_sent():
    fake = FakeOpenAI(
        FakeStream([piece("Par")], error=APIConnectionError(request=REQUEST)), good_stream()
    )
    seen = []
    with pytest.raises(LLMUnavailableError):
        async for event in make_client(fake).stream(MESSAGES):
            seen.append(event)
    assert seen == [StreamDelta("Par")]
    assert fake.calls == 1  # a second answer would be glued onto the first


@pytest.mark.anyio
async def test_timeouts_before_any_text_are_retried_then_reported():
    fake = FakeOpenAI(*[APITimeoutError(request=REQUEST) for _ in range(3)])
    with pytest.raises(LLMTimeoutError):
        await collect(make_client(fake))
    assert fake.calls == 3


@pytest.mark.anyio
async def test_does_not_retry_when_the_wait_would_blow_the_time_budget():
    sleeps: list[float] = []
    fake = FakeOpenAI(status_error(RateLimitError, 429, {"retry-after": "5"}))
    with pytest.raises(LLMRateLimitError):
        await collect(make_client(fake, total_timeout=1, sleeps=sleeps))
    assert (fake.calls, sleeps) == (1, [])


@pytest.mark.anyio
async def test_an_answer_with_no_text_is_an_error():
    with pytest.raises(LLMError):
        await collect(make_client(FakeOpenAI(FakeStream([piece(""), piece(None), USAGE_CHUNK]))))


@pytest.mark.anyio
async def test_a_stream_that_runs_too_long_is_cut_off():
    fake = FakeOpenAI(FakeStream([piece("a"), piece("b"), piece("c")], delay=0.06))
    seen = []
    with pytest.raises(LLMTimeoutError):
        async for event in make_client(fake, stream_timeout=0.1).stream(MESSAGES):
            seen.append(event)
    assert seen == [StreamDelta("a")]


@pytest.mark.anyio
async def test_closing_the_stream_early_closes_the_provider_and_logs_cancelled(caplog):
    caplog.set_level(logging.INFO, logger="app.service.llm_client.openai_compatible")
    provider_stream = FakeStream([piece("a"), piece("b")], hang=True)
    events = make_client(FakeOpenAI(provider_stream)).stream(MESSAGES)

    assert await events.__anext__() == StreamDelta("a")
    await events.aclose()

    assert provider_stream.closed
    final = [r for r in caplog.records if r.getMessage() == "LLM call finished"]
    assert [r.status for r in final] == ["cancelled"]


@pytest.mark.anyio
async def test_cancelling_the_listener_closes_the_provider_stream(caplog):
    caplog.set_level(logging.INFO, logger="app.service.llm_client.openai_compatible")
    provider_stream = FakeStream([piece("a")], hang=True)
    client = make_client(FakeOpenAI(provider_stream))
    got_first_piece = asyncio.Event()

    async def listen():
        async for _ in client.stream(MESSAGES):
            got_first_piece.set()

    task = asyncio.create_task(listen())
    await asyncio.wait_for(got_first_piece.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider_stream.closed
    assert [r.status for r in caplog.records if r.getMessage() == "LLM call finished"] == [
        "cancelled"
    ]


@pytest.mark.anyio
async def test_a_finished_stream_is_logged_once_with_time_to_first_piece(caplog):
    caplog.set_level(logging.INFO, logger="app.service.llm_client.openai_compatible")
    secret = SimpleNamespace(role="user", content="secret user text")
    await collect(make_client(FakeOpenAI(good_stream())))

    records = [r for r in caplog.records if r.getMessage() == "LLM call finished"]
    assert len(records) == 1
    record = records[0]
    assert (record.status, record.attempts, record.model) == ("ok", 1, MODEL)
    assert (record.input_tokens, record.output_tokens) == (11, 3)
    assert isinstance(record.first_piece_ms, int)
    assert all(secret.content not in str(vars(r)) for r in caplog.records)
