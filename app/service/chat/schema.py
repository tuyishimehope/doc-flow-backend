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
