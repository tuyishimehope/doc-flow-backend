"""The contract the rest of the app depends on, independent of any provider.

Routers and services import from here. Only ``openrouter.py`` knows about
OpenRouter, so a second provider (Phase 4) means a new file, not a rewrite.
"""

from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str  # the model that really answered (a router may pick one for us)
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    attempts: int


class LLMError(Exception):
    """Base class. ``status`` is the short word written to the call log."""

    status = "error"


class LLMNotConfiguredError(LLMError):
    status = "not_configured"


class LLMTimeoutError(LLMError):
    status = "timeout"


class LLMRateLimitError(LLMError):
    status = "rate_limited"

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMUnavailableError(LLMError):
    status = "unavailable"


class LLMClient(Protocol):
    async def complete(self, messages: list[Message]) -> LLMResponse: ...
