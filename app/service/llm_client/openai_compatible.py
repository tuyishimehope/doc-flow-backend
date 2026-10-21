"""``LLMClient`` for every provider that speaks the OpenAI wire format.

OpenRouter, Google Gemini, Groq and OpenAI all accept the same request, so one class
serves them all: only ``base_url``, the key, the model name and the ``provider`` label
differ. The SDK's own retries are switched off (``max_retries=0``) so this class is the
single place that decides what is retried, how long to wait and how long the caller may
be kept waiting.
"""

import asyncio
import logging
import random
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from app.core.observability import set_tag
from app.service.llm_client.llm_client import (
    LLMError,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
    Message,
    StreamDelta,
    StreamDone,
    StreamEvent,
)

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 8.0


@dataclass
class _CallState:
    model: str
    attempts: int = 0
    status: str = "error"
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    first_piece_ms: int | None = None


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        provider: str,
        client: AsyncOpenAI,
        model: str,
        timeout: float,
        total_timeout: float,
        max_retries: int,
        max_output_tokens: int,
        stream_timeout: float = 120,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.provider = provider  # label used in logs and in the answer's metadata
        self._client = client
        self._model = model
        self._timeout = timeout
        self._total_timeout = total_timeout
        self._max_retries = max_retries
        self._max_output_tokens = max_output_tokens
        self._stream_timeout = stream_timeout
        self._sleep = sleep  # injectable so tests do not really wait

    async def complete(self, messages: list[Message]) -> LLMResponse:
        set_tag(
            "llm.provider", self.provider
        )  # the last one set is the one that handled the request
        started = time.monotonic()
        state = _CallState(model=self._model)
        try:
            completion = await asyncio.wait_for(
                self._request_with_retries(messages, state),
                timeout=self._total_timeout,
            )
            text = _extract_text(completion)
            usage = getattr(completion, "usage", None)
            state.model = getattr(completion, "model", None) or self._model
            _apply_usage(state, usage)
            state.status = "ok"
        except asyncio.TimeoutError:
            state.status = LLMTimeoutError.status
            raise LLMTimeoutError("LLM call exceeded its total time budget") from None
        except LLMError as exc:
            state.status = exc.status
            raise
        finally:
            self._log(state, started)
        return LLMResponse(
            text=text,
            model=state.model,
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            latency_ms=_elapsed_ms(started),
            attempts=state.attempts,
            provider=self.provider,
            cached_tokens=state.cached_tokens,
            reasoning_tokens=state.reasoning_tokens,
        )

    async def stream(self, messages: list[Message]) -> AsyncGenerator[StreamEvent, None]:
        """Yield text pieces as they arrive, then one ``StreamDone``.

        Retries are only allowed before the first piece: once text has reached the
        caller it cannot be taken back, so a later failure is raised instead.
        """
        set_tag("llm.provider", self.provider)
        started = time.monotonic()
        state = _CallState(model=self._model)
        sent_text = False
        try:
            for attempt in range(1, self._max_retries + 2):
                state.attempts = attempt
                try:
                    response = await self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,  # type: ignore[arg-type]
                        max_tokens=self._max_output_tokens,
                        # Also the longest allowed silence between two pieces.
                        timeout=self._timeout,
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                    # `async with` closes the provider connection however we leave,
                    # including when the caller stops listening halfway.
                    async with response:
                        async for chunk in response:
                            state.model = getattr(chunk, "model", None) or state.model
                            usage = getattr(chunk, "usage", None)
                            if usage is not None:
                                _apply_usage(state, usage)
                            text = _delta_text(chunk)
                            if not sent_text:
                                text = text.lstrip()  # models often start with blank lines
                            if not text:
                                continue  # role-only chunks and the final usage chunk
                            if time.monotonic() - started > self._stream_timeout:
                                raise LLMTimeoutError("Stream exceeded its maximum duration")
                            if not sent_text:
                                sent_text = True
                                state.first_piece_ms = _elapsed_ms(started)
                            yield StreamDelta(text)
                    if not sent_text:
                        raise LLMError("Provider returned an empty answer")
                    break
                except APIError as exc:
                    await self._sleep(
                        self._plan_retry(exc, attempt, started, can_retry=not sent_text)
                    )
            state.status = "ok"
            yield StreamDone(
                model=state.model,
                input_tokens=state.input_tokens,
                output_tokens=state.output_tokens,
                latency_ms=_elapsed_ms(started),
                first_piece_ms=state.first_piece_ms or 0,
                attempts=state.attempts,
                provider=self.provider,
                cached_tokens=state.cached_tokens,
                reasoning_tokens=state.reasoning_tokens,
            )
        except (asyncio.CancelledError, GeneratorExit):
            # The listener went away (closed tab, shutdown). Not an error.
            if state.status != "ok":
                state.status = "cancelled"
            raise
        except LLMError as exc:
            state.status = exc.status
            raise
        finally:
            self._log(state, started)

    async def _request_with_retries(self, messages: list[Message], state: _CallState) -> Any:
        started = time.monotonic()
        for attempt in range(1, self._max_retries + 2):
            state.attempts = attempt
            try:
                return await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=self._max_output_tokens,
                    timeout=self._timeout,
                )
            except (APIConnectionError, APIStatusError) as exc:
                await self._sleep(self._plan_retry(exc, attempt, started))
        raise LLMError("Retry loop ended without a result")  # pragma: no cover

    def _plan_retry(
        self, exc: APIError, attempt: int, started: float, *, can_retry: bool = True
    ) -> float:
        """Return how long to wait before the next attempt, or raise the mapped error."""
        error, retryable, suggested_delay = _classify(exc)
        delay = suggested_delay if suggested_delay is not None else _backoff(attempt)
        out_of_time = (time.monotonic() - started) + delay > self._total_timeout
        if (
            not (can_retry and retryable)
            or attempt > self._max_retries
            or delay > MAX_BACKOFF_SECONDS
            or out_of_time
        ):
            raise error from exc
        logger.warning(
            "LLM call failed, retrying",
            extra={
                "provider": self.provider,
                "model": self._model,
                "status": error.status,
                "attempts": attempt,
            },
        )
        return delay

    def _log(self, state: _CallState, started: float) -> None:
        # One record per call. The message text is never logged, only metadata.
        logger.log(
            logging.INFO if state.status in ("ok", "cancelled") else logging.WARNING,
            "LLM call finished",
            extra={
                "provider": self.provider,
                "model": state.model,
                "status": state.status,
                "latency_ms": _elapsed_ms(started),
                "first_piece_ms": state.first_piece_ms,
                "input_tokens": state.input_tokens,
                "output_tokens": state.output_tokens,
                "cached_tokens": state.cached_tokens,
                "reasoning_tokens": state.reasoning_tokens,
                "attempts": state.attempts,
            },
        )


def _count(value: Any) -> int | None:
    """A token count, or None when the provider did not send a real number."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _apply_usage(state: _CallState, usage: Any) -> None:
    """Copy the token counts of a ``usage`` object onto the call state.

    Providers differ: some send no usage at all, some leave out the ``*_details`` parts, and
    some send them as ``None``. Every level is therefore optional. Where the numbers live:

    * ``prompt_tokens_details.cached_tokens``: input served from the provider's cache
    * ``completion_tokens_details.reasoning_tokens``: hidden thinking, inside ``completion_tokens``
    """
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    state.input_tokens = _count(getattr(usage, "prompt_tokens", None))
    state.output_tokens = _count(getattr(usage, "completion_tokens", None))
    state.cached_tokens = _count(getattr(prompt_details, "cached_tokens", None))
    state.reasoning_tokens = _count(getattr(completion_details, "reasoning_tokens", None))


def _classify(exc: APIError) -> tuple[LLMError, bool, float | None]:
    """Return (error to raise, whether to retry, delay the server asked for)."""
    if isinstance(exc, APITimeoutError):
        return LLMTimeoutError("Provider call timed out"), True, None
    if isinstance(exc, APIConnectionError):
        return LLMUnavailableError("Provider unreachable"), True, None
    if isinstance(exc, RateLimitError):
        delay = _retry_after_seconds(exc)
        return LLMRateLimitError("Provider rate limit", retry_after=delay), True, delay
    if isinstance(exc, APIStatusError):
        code = exc.status_code
        if code == 408 or code >= 500:
            return LLMUnavailableError(f"Provider error {code}"), True, None
        if code in (401, 402, 403):  # bad key or no credit: retrying cannot help
            return (
                LLMUnavailableError(f"Provider refused our credentials or credit ({code})"),
                False,
                None,
            )
        # Other 4xx: never retried here, but another provider may still succeed. Providers
        # use these codes for their own problems too: Gemini answers 400 to an invalid API key,
        # Groq answers 400 to a retired model, and a retired model is a 404 elsewhere.
        return LLMError(f"Provider could not serve the request ({code})"), False, None
    # A plain APIError is how the SDK reports an error event inside an open stream.
    return LLMUnavailableError("Provider reported an error while streaming"), True, None


def _retry_after_seconds(exc: APIStatusError) -> float | None:
    raw = exc.response.headers.get("retry-after")
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def _backoff(attempt: int) -> float:
    # 0.5s, 1s, 2s ... with jitter so many clients do not retry in lockstep.
    return min(0.5 * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS) * random.uniform(0.8, 1.2)


def _extract_text(completion: Any) -> str:
    choices = getattr(completion, "choices", None) or []
    text = choices[0].message.content if choices else None
    if not text or not text.strip():
        raise LLMError("Provider returned an empty answer")
    return text.strip()


def _delta_text(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None)
    if not choices:  # the last chunk only carries the token counts
        return ""
    return getattr(choices[0].delta, "content", None) or ""


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
