"""The contract the rest of the app depends on, independent of any provider.

Routers and services import from here. Only the client modules know about a
provider's wire format, so adding a provider means a new entry in the registry,
not a change to any caller.
"""

from collections.abc import AsyncGenerator
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
    provider: str = ""  # which provider answered (matters once there is a fallback)


@dataclass(frozen=True)
class StreamDelta:
    """A piece of the answer, in the order the model wrote it."""

    text: str


@dataclass(frozen=True)
class StreamDone:
    """Always the last event of a stream that finished normally."""

    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    first_piece_ms: int  # the wait before anything appeared: what users feel most
    attempts: int
    provider: str = ""


StreamEvent = StreamDelta | StreamDone


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

    def stream(self, messages: list[Message]) -> AsyncGenerator[StreamEvent, None]:
        """Yield ``StreamDelta`` pieces, then one ``StreamDone``.

        Failures before the first piece are raised as ``LLMError`` subclasses so
        callers can still answer with a proper HTTP status. Failures after that
        are raised too, but the caller has already started its response.
        """
        ...
