"""``FallbackLLMClient``: try one provider after another behind the ``LLMClient`` interface.

The router and the chat service never know there is more than one provider.

Rules:
* Every failure moves on to the next provider, including 4xx answers. Providers use those
  codes for their own problems (an invalid key, a retired model), and a rejected request
  generates no tokens, so trying the next provider costs nothing extra.
* The whole chain shares one time budget, so four slow providers cannot add up to minutes.
* A stream can only fall back *before its first piece*. After that, text has already
  reached the user, and another provider's answer cannot be glued onto it.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Sequence

from app.service.llm_client.llm_client import (
    LLMClient,
    LLMError,
    LLMResponse,
    LLMTimeoutError,
    Message,
    StreamEvent,
)

logger = logging.getLogger(__name__)


class FallbackLLMClient:
    def __init__(self, clients: Sequence[LLMClient], *, total_timeout: float) -> None:
        if not clients:
            raise ValueError("FallbackLLMClient needs at least one client")
        self._clients = list(clients)
        self._total_timeout = total_timeout

    @property
    def provider(self) -> str:
        return "+".join(_name(client) for client in self._clients)

    async def complete(self, messages: list[Message]) -> LLMResponse:
        started = time.monotonic()
        last_error: LLMError | None = None
        for index, client in enumerate(self._clients):
            remaining = self._remaining(started)
            if last_error is not None and remaining <= 0:
                break
            try:
                return await asyncio.wait_for(
                    client.complete(messages), timeout=max(remaining, 0.01)
                )
            except asyncio.TimeoutError:
                last_error = LLMTimeoutError("The fallback chain ran out of time")
            except LLMError as exc:
                last_error = exc
            self._log_fallback(client, last_error, index)
        assert last_error is not None
        raise last_error

    async def stream(self, messages: list[Message]) -> AsyncGenerator[StreamEvent, None]:
        started = time.monotonic()
        last_error: LLMError | None = None
        for index, client in enumerate(self._clients):
            remaining = self._remaining(started)
            if last_error is not None and remaining <= 0:
                break
            events = client.stream(messages)
            try:
                first = await asyncio.wait_for(events.__anext__(), timeout=max(remaining, 0.01))
            except asyncio.TimeoutError:
                last_error = LLMTimeoutError("The fallback chain ran out of time")
            except StopAsyncIteration:
                last_error = LLMError("Provider ended the stream without any event")
            except LLMError as exc:
                last_error = exc
            except BaseException:
                await events.aclose()  # cancelled while waiting: stop paying for this call
                raise
            else:
                # From here on we are committed to this provider, whatever happens.
                try:
                    yield first
                    async for event in events:
                        yield event
                finally:
                    await events.aclose()
                return
            await events.aclose()
            self._log_fallback(client, last_error, index)
        assert last_error is not None
        raise last_error

    def _remaining(self, started: float) -> float:
        return self._total_timeout - (time.monotonic() - started)

    def _log_fallback(self, client: LLMClient, error: LLMError | None, index: int) -> None:
        following = self._clients[index + 1] if index + 1 < len(self._clients) else None
        # Failing over is routine (a warning). Running out of providers is an outage, so it is
        # logged as an error, which Sentry turns into an event.
        logger.log(
            logging.WARNING if following else logging.ERROR,
            "LLM provider failed, trying the next one"
            if following
            else "LLM provider failed, no providers left",
            extra={
                "provider": _name(client),
                "status": error.status if error else "error",
                "fallback_to": _name(following) if following else None,
            },
        )


def _name(client: LLMClient) -> str:
    return getattr(client, "provider", type(client).__name__)
