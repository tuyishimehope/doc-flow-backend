"""OpenRouter implementation of ``LLMClient``.

OpenRouter speaks the OpenAI wire format, so we reuse the OpenAI SDK with a
different ``base_url``. The SDK's own retries are switched off (``max_retries=0``)
so this class is the single place that decides what is retried, how long to wait
and how long the caller may be kept waiting.
"""

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from app.core.config import settings
from app.service.llm_client.llm_client import (
    LLMError,
    LLMNotConfiguredError,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    LLMUnavailableError,
    Message,
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


class OpenRouterClient:
    provider = "openrouter"

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        timeout: float,
        total_timeout: float,
        max_retries: int,
        max_output_tokens: int,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._client = client
        self._model = model
        self._timeout = timeout
        self._total_timeout = total_timeout
        self._max_retries = max_retries
        self._max_output_tokens = max_output_tokens
        self._sleep = sleep  # injectable so tests do not really wait

    async def complete(self, messages: list[Message]) -> LLMResponse:
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
            state.input_tokens = getattr(usage, "prompt_tokens", None)
            state.output_tokens = getattr(usage, "completion_tokens", None)
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
        )

    async def _request_with_retries(self, messages: list[Message], state: _CallState) -> Any:
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
                error, retryable, suggested_delay = _classify(exc)
                delay = suggested_delay if suggested_delay is not None else _backoff(attempt)
                out_of_attempts = attempt > self._max_retries
                if not retryable or out_of_attempts or delay > MAX_BACKOFF_SECONDS:
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
                await self._sleep(delay)
        raise LLMError("Retry loop ended without a result")  # pragma: no cover

    def _log(self, state: _CallState, started: float) -> None:
        # One record per call. The message text is never logged, only metadata.
        logger.log(
            logging.INFO if state.status == "ok" else logging.WARNING,
            "LLM call finished",
            extra={
                "provider": self.provider,
                "model": state.model,
                "status": state.status,
                "latency_ms": _elapsed_ms(started),
                "input_tokens": state.input_tokens,
                "output_tokens": state.output_tokens,
                "attempts": state.attempts,
            },
        )


def _classify(exc: APIConnectionError | APIStatusError) -> tuple[LLMError, bool, float | None]:
    """Return (error to raise, whether to retry, delay the server asked for)."""
    if isinstance(exc, APITimeoutError):
        return LLMTimeoutError("Provider call timed out"), True, None
    if isinstance(exc, APIConnectionError):
        return LLMUnavailableError("Provider unreachable"), True, None
    assert isinstance(exc, APIStatusError)
    if isinstance(exc, RateLimitError):
        delay = _retry_after_seconds(exc)
        return LLMRateLimitError("Provider rate limit", retry_after=delay), True, delay
    code = exc.status_code
    if code == 408 or code >= 500:
        return LLMUnavailableError(f"Provider error {code}"), True, None
    if code in (401, 402, 403):  # bad key or no credit: retrying cannot help
        return LLMUnavailableError(f"Provider refused our credentials or credit ({code})"), False, None
    return LLMError(f"Provider rejected the request ({code})"), False, None


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


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


@lru_cache(maxsize=1)
def get_llm_client() -> OpenRouterClient:
    """One shared client, so HTTP connections are reused between requests."""
    api_key = settings.OPENROUTER_API_KEY.get_secret_value()
    if not api_key:
        raise LLMNotConfiguredError("OPENROUTER_API_KEY is not set")
    return OpenRouterClient(
        client=AsyncOpenAI(
            base_url=settings.OPENROUTER_BASE_URL,
            api_key=api_key,
            timeout=settings.CHAT_TIMEOUT_SECONDS,
            max_retries=0,
        ),
        model=settings.CHAT_MODEL,
        timeout=settings.CHAT_TIMEOUT_SECONDS,
        total_timeout=settings.CHAT_TOTAL_TIMEOUT_SECONDS,
        max_retries=settings.CHAT_MAX_RETRIES,
        max_output_tokens=settings.CHAT_MAX_OUTPUT_TOKENS,
    )
