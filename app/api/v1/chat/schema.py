from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.service.chat.schema import MAX_TITLE_CHARS

MAX_MESSAGE_CHARS = 4000

Title = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TITLE_CHARS)
]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Whitespace is trimmed first, then the length rules apply to what is left.
    message: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_CHARS),
    ]


class ChatResponse(BaseModel):
    answer: str
    model: str


class StreamChatRequest(ChatRequest):
    # Missing: start a new conversation. Present: continue that one.
    conversation_id: int | None = Field(default=None, ge=1)


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: Title | None = None


class ConversationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: Title


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    status: str
    model: str | None
    created_at: datetime


class ConversationDetailResponse(ConversationResponse):
    messages: list[MessageResponse]


class PaginatedConversationResponse(BaseModel):
    conversations: list[ConversationResponse]
    total: int
    skip: int
    limit: int
    has_more: bool
