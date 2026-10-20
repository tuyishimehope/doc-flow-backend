"""Speech to text through Groq's OpenAI-compatible endpoint.

Same rules as the chat clients: callers only see ``LLMError`` subclasses, never an SDK
exception or a provider message, and one log line per call carries metadata only. The
audio, its file name and the transcript are never logged.

There are no retries (``max_retries=0``) and no fallback provider. Add them when a real
failure shows they are needed.
"""

import logging
import time
from dataclasses import dataclass
from functools import lru_cache

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.core.config import settings
from app.service.llm_client.llm_client import (
    LLMError,
    LLMNotConfiguredError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

PROVIDER = "groq"
# The formats Groq accepts. They belong to the provider, not to the deployment.
ALLOWED_AUDIO_EXTENSIONS = frozenset(
    {"flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"}
)


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    model: str
    latency_ms: int
    audio_bytes: int


@lru_cache(maxsize=1)
def _client() -> AsyncOpenAI:
    """One HTTP client for every call, so connections are reused."""
    api_key = settings.GROQ_AI_API_KEY.get_secret_value()
    if not api_key:
        raise LLMNotConfiguredError("GROQ_AI_API_KEY is not set")
    return AsyncOpenAI(
        base_url=settings.GROQ_BASE_URL,
        api_key=api_key,
        timeout=settings.TRANSCRIBE_TIMEOUT_SECONDS,
        max_retries=0,
    )


async def transcribe(
    *, user_id: int, filename: str, content: bytes, content_type: str | None
) -> TranscriptionResult:
    started = time.monotonic()
    state = "ok"
    try:
        # The file name is sent so Groq can tell the format; it is never logged.
        response = await _client().audio.transcriptions.create(
            model=settings.TRANSCRIBE_MODEL,
            file=(filename, content, content_type or "application/octet-stream"),
            timeout=settings.TRANSCRIBE_TIMEOUT_SECONDS,
        )
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            raise LLMError("The provider returned no text")
    except LLMError as exc:
        state = exc.status
        raise
    except APITimeoutError:
        state = "timeout"
        raise LLMTimeoutError("Transcription timed out") from None
    except APIConnectionError:
        state = "unavailable"
        raise LLMUnavailableError("Could not reach the transcription provider") from None
    except APIStatusError as exc:
        if exc.status_code == 429:
            state = "rate_limited"
            raise LLMRateLimitError("Transcription provider is rate limiting us") from None
        state = "error"
        raise LLMError(f"Transcription provider answered {exc.status_code}") from None
    finally:
        logger.log(
            logging.INFO if state == "ok" else logging.WARNING,
            "Transcription finished",
            extra={
                "provider": PROVIDER,
                "model": settings.TRANSCRIBE_MODEL,
                "status": state,
                "latency_ms": int((time.monotonic() - started) * 1000),
                "audio_bytes": len(content),
                "user_id": user_id,
            },
        )
    return TranscriptionResult(
        text=text.strip(),
        model=settings.TRANSCRIBE_MODEL,
        latency_ms=int((time.monotonic() - started) * 1000),
        audio_bytes=len(content),
    )
