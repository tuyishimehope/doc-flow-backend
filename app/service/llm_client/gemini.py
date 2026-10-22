"""The one Gemini client, through its OpenAI-compatible endpoint, shared by every feature that
calls Gemini directly (image questions, document questions).

These features are separate from the chat clients on purpose: the chat fallback chain also holds
models that cannot read images or follow a response schema. Each caller passes its own
``timeout=`` on the call, so this client only carries a ceiling for a call that forgets to.
"""

from functools import lru_cache
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.service.llm_client.llm_client import LLMNotConfiguredError

CLIENT_TIMEOUT_SECONDS = 120  # the highest any feature allows; calls set their own, lower, value


@lru_cache(maxsize=1)
def gemini_client() -> AsyncOpenAI:
    """One HTTP client for every call, so connections are reused."""
    return new_gemini_client()


def new_gemini_client() -> AsyncOpenAI:
    """A client that is not shared. A Celery task runs each call in its own event loop
    (``asyncio.run``), and an HTTP client must not be reused across loops."""
    api_key = settings.GEMINI_AI_API_KEY.get_secret_value()
    if not api_key:
        raise LLMNotConfiguredError("GEMINI_AI_API_KEY is not set")
    return AsyncOpenAI(
        base_url=settings.GEMINI_BASE_URL,
        api_key=api_key,
        timeout=CLIENT_TIMEOUT_SECONDS,
        max_retries=0,
    )


def token_count(value: Any) -> int | None:
    """A token count, or None when the provider did not send a real number."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None
