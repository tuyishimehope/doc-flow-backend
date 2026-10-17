"""FallbackLLMClient with scripted fake providers. No network."""

import asyncio
import logging

import pytest

from app.service.llm_client.fallback import FallbackLLMClient
from app.service.llm_client.llm_client import (
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
    StreamDelta,
    StreamDone,
)

MESSAGES = [{"role": "user", "content": "hi"}]


class Scripted:
    """A fake provider that answers, fails, or hangs on command and counts its calls."""

    def __init__(
        self,
        provider: str,
        *,
        error: Exception | None = None,
        error_after_piece: Exception | None = None,
        hang: bool = False,
        empty: bool = False,
    ) -> None:
        self.provider = provider
        self.error = error
        self.error_after_piece = error_after_piece
        self.hang = hang
        self.empty = empty
        self.calls = 0
        self.closed = False

    async def complete(self, messages):
        self.calls += 1
        if self.hang:
            await asyncio.sleep(30)
        if self.error:
            raise self.error
        return LLMResponse(
            text=f"from {self.provider}", model=f"{self.provider}/model", input_tokens=1,
            output_tokens=1, latency_ms=1, attempts=1, provider=self.provider,
        )

    async def stream(self, messages):
        self.calls += 1
        try:
            if self.hang:
                await asyncio.sleep(30)
            if self.error:
                raise self.error
            if self.empty:
                return
            yield StreamDelta(f"from {self.provider}")
            if self.error_after_piece:
                raise self.error_after_piece
            yield StreamDone(
                model=f"{self.provider}/model", input_tokens=1, output_tokens=1, latency_ms=1,
                first_piece_ms=1, attempts=1, provider=self.provider,
            )
        finally:
            self.closed = True


def chain(*clients: Scripted, total_timeout: float = 5) -> FallbackLLMClient:
    return FallbackLLMClient(list(clients), total_timeout=total_timeout)


async def collect(client) -> list:
    return [event async for event in client.stream(MESSAGES)]


FALLBACK_ERRORS = [
    LLMTimeoutError("slow"),
    LLMRateLimitError("429", retry_after=1),
    LLMUnavailableError("down or bad key"),
    LLMError("a 4xx: Gemini's 400 for a bad key, a retired model, an empty answer"),
]
FALLBACK_IDS = ["timeout", "rate-limited", "unavailable", "other-4xx"]


def test_needs_at_least_one_client_and_names_the_chain():
    with pytest.raises(ValueError):
        FallbackLLMClient([], total_timeout=1)
    assert chain(Scripted("a"), Scripted("b")).provider == "a+b"


# ---- complete ---------------------------------------------------------------


@pytest.mark.anyio
async def test_the_first_provider_answers_and_the_rest_are_not_touched():
    a, b = Scripted("a"), Scripted("b")
    result = await chain(a, b).complete(MESSAGES)
    assert (result.provider, a.calls, b.calls) == ("a", 1, 0)


@pytest.mark.anyio
@pytest.mark.parametrize("error", FALLBACK_ERRORS, ids=FALLBACK_IDS)
async def test_a_failing_provider_hands_over_to_the_next(error):
    a, b = Scripted("a", error=error), Scripted("b")
    result = await chain(a, b).complete(MESSAGES)
    assert (result.text, result.provider, a.calls, b.calls) == ("from b", "b", 1, 1)


@pytest.mark.anyio
async def test_it_walks_the_whole_chain_and_ends_with_the_last_provider():
    a = Scripted("a", error=LLMTimeoutError("a"))
    b = Scripted("b", error=LLMRateLimitError("b"))
    c = Scripted("c")
    result = await chain(a, b, c).complete(MESSAGES)
    assert (result.provider, a.calls, b.calls, c.calls) == ("c", 1, 1, 1)


@pytest.mark.anyio
async def test_when_every_provider_fails_the_last_error_is_raised():
    a = Scripted("a", error=LLMTimeoutError("a"))
    b = Scripted("b", error=LLMUnavailableError("b"))
    with pytest.raises(LLMUnavailableError):
        await chain(a, b).complete(MESSAGES)
    assert (a.calls, b.calls) == (1, 1)


@pytest.mark.anyio
async def test_one_time_budget_covers_the_whole_chain():
    a, b = Scripted("a", hang=True), Scripted("b")
    with pytest.raises(LLMTimeoutError):
        await chain(a, b, total_timeout=0.05).complete(MESSAGES)
    assert b.calls == 0  # the budget was spent waiting for a


@pytest.mark.anyio
async def test_a_fallback_is_logged_with_who_failed_and_who_is_next(caplog):
    caplog.set_level(logging.WARNING, logger="app.service.llm_client.fallback")
    await chain(Scripted("a", error=LLMTimeoutError("slow")), Scripted("b")).complete(MESSAGES)
    [record] = [r for r in caplog.records if "trying the next" in r.getMessage()]
    assert (record.provider, record.status, record.fallback_to) == ("a", "timeout", "b")


# ---- stream -----------------------------------------------------------------


@pytest.mark.anyio
async def test_a_stream_comes_from_the_first_provider_when_it_works():
    a, b = Scripted("a"), Scripted("b")
    events = await collect(chain(a, b))
    assert [type(e).__name__ for e in events] == ["StreamDelta", "StreamDone"]
    assert (events[-1].provider, b.calls) == ("a", 0)


@pytest.mark.anyio
@pytest.mark.parametrize("error", FALLBACK_ERRORS, ids=FALLBACK_IDS)
async def test_a_stream_that_fails_before_any_text_falls_back(error):
    a, b = Scripted("a", error=error), Scripted("b")
    events = await collect(chain(a, b))
    assert events[0] == StreamDelta("from b")
    assert events[-1].provider == "b"
    assert a.closed


@pytest.mark.anyio
async def test_a_stream_that_ends_without_any_event_falls_back():
    events = await collect(chain(Scripted("a", empty=True), Scripted("b")))
    assert events[-1].provider == "b"


@pytest.mark.anyio
async def test_a_stream_never_falls_back_after_text_was_sent():
    a, b = Scripted("a", error_after_piece=LLMUnavailableError("dropped")), Scripted("b")
    seen = []
    with pytest.raises(LLMUnavailableError):
        async for event in chain(a, b).stream(MESSAGES):
            seen.append(event)
    assert seen == [StreamDelta("from a")]
    assert b.calls == 0  # a second answer would be glued onto the first


@pytest.mark.anyio
async def test_when_every_stream_fails_the_last_error_is_raised():
    a = Scripted("a", error=LLMTimeoutError("a"))
    b = Scripted("b", error=LLMRateLimitError("b"))
    with pytest.raises(LLMRateLimitError):
        await collect(chain(a, b))


@pytest.mark.anyio
async def test_a_provider_that_hangs_before_its_first_piece_uses_up_the_budget():
    a, b = Scripted("a", hang=True), Scripted("b")
    with pytest.raises(LLMTimeoutError):
        await collect(chain(a, b, total_timeout=0.05))
    assert a.closed and b.calls == 0


@pytest.mark.anyio
async def test_closing_the_stream_early_closes_the_provider_that_was_answering():
    a, b = Scripted("a"), Scripted("b")
    events = chain(a, b).stream(MESSAGES)
    assert await events.__anext__() == StreamDelta("from a")
    await events.aclose()
    assert a.closed and b.calls == 0


@pytest.mark.anyio
async def test_cancelling_while_waiting_for_the_first_piece_closes_the_provider():
    a = Scripted("a", hang=True)
    task = asyncio.create_task(collect(chain(a, Scripted("b"))))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert a.closed
