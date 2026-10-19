from dataclasses import dataclass
from enum import Enum

DEFAULT_TITLE = "New conversation"
MAX_TITLE_CHARS = 100
AUTO_TITLE_CHARS = 50


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class MessageStatus(str, Enum):
    COMPLETE = "complete"
    CANCELLED = "cancelled"  # the user left before the answer finished
    ERROR = "error"  # the provider failed after some text was already sent


@dataclass(frozen=True)
class CallUsage:
    """What one model call cost. Every field may be missing: providers differ."""

    provider: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_ms: int | None = None
