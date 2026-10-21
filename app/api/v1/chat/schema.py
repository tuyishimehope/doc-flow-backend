from datetime import date, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.service.chat.schema import MAX_TITLE_CHARS
from app.service.llm_client.registry import ModelChoice

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
    # Which provider to try first. Only names from the allowlist are accepted, never a
    # raw model name, so a caller cannot pick an expensive model we did not offer.
    model: ModelChoice | None = None


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


class ModelOption(BaseModel):
    id: str
    label: str
    model: str | None


class ModelsResponse(BaseModel):
    default: str
    models: list[ModelOption]


class DailyUsage(BaseModel):
    day: date
    answers: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    reasoning_tokens: int


class LatencyStats(BaseModel):
    samples: int  # how many finished answers the percentiles are based on
    p50_ms: int | None
    p95_ms: int | None


class ChatStatsResponse(BaseModel):
    """The current user's own usage over the last ``days`` days (today included)."""

    days: int
    answers: int
    complete: int
    errors: int
    cancelled: int
    error_rate: float  # errors / answers; cancelled answers are the user's choice, not errors
    latency: LatencyStats
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    per_day: list[DailyUsage]


class TranscriptionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    text: str
    model: str
    created_at: datetime


class ImageAnswerResponse(BaseModel):
    conversation_id: int
    answer: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
