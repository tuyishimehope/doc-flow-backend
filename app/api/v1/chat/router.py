import json
import logging
import math
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.v1.chat.schema import ChatRequest, ChatResponse
from app.core.config import settings
from app.core.rate_limit import check_rate_limit
from app.service.auth.auth import CurrentUser
from app.service.chat.chat import answer_message, stream_answer
from app.service.llm_client.llm_client import (
    LLMClient,
    LLMError,
    LLMNotConfiguredError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
    StreamDelta,
    StreamDone,
    StreamEvent,
)
from app.service.llm_client.openrouter import get_llm_client

logger = logging.getLogger(__name__)

CHAT_ROUTER = APIRouter(prefix="/api/v1/chat", tags=["chat"])

UNAVAILABLE = "The assistant is temporarily unavailable. Please try again shortly."

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    429: {"description": "Too many chat requests from this user"},
    502: {"description": "The AI provider rejected the request or returned nothing"},
    503: {"description": "The AI provider is unavailable or rate limiting us"},
    504: {"description": "The AI provider took too long"},
}


def get_chat_llm() -> LLMClient:
    try:
        return get_llm_client()
    except LLMNotConfiguredError:
        logger.error("Chat is disabled because OPENROUTER_API_KEY is not set")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=UNAVAILABLE) from None


def to_http_error(exc: LLMError) -> HTTPException:
    """Turn an LLM failure into a safe, generic HTTP error. The client logged the cause."""
    if isinstance(exc, LLMTimeoutError):
        return HTTPException(
            status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The assistant took too long to answer. Please try again.",
        )
    if isinstance(exc, LLMRateLimitError):
        wait = exc.retry_after if exc.retry_after is not None else 10
        return HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=UNAVAILABLE,
            headers={"Retry-After": str(max(1, math.ceil(wait)))},
        )
    if isinstance(exc, LLMUnavailableError):
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=UNAVAILABLE)
    return HTTPException(
        status.HTTP_502_BAD_GATEWAY,
        detail="The assistant could not answer that. Please try again.",
    )


async def _rate_limit(current_user: CurrentUser) -> None:
    await check_rate_limit(
        key=f"chat:{current_user.id}",
        limit=settings.CHAT_RATE_LIMIT_PER_MINUTE,
        window_seconds=60,
    )


@CHAT_ROUTER.post("", response_model=ChatResponse, responses=ERROR_RESPONSES)
async def chat_endpoint(
    body: ChatRequest,
    current_user: CurrentUser,
    llm: Annotated[LLMClient, Depends(get_chat_llm)],
) -> ChatResponse:
    await _rate_limit(current_user)
    try:
        result = await answer_message(body.message, llm)
    except LLMError as exc:
        raise to_http_error(exc) from None
    return ChatResponse(answer=result.text, model=result.model)


@CHAT_ROUTER.post(
    "/stream",
    response_class=StreamingResponse,
    responses={
        **ERROR_RESPONSES,
        200: {
            "description": "Server-Sent Events: `token`, then `done` or `error`",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        },
    },
)
async def chat_stream_endpoint(
    body: ChatRequest,
    current_user: CurrentUser,
    llm: Annotated[LLMClient, Depends(get_chat_llm)],
) -> StreamingResponse:
    await _rate_limit(current_user)
    events = stream_answer(body.message, llm)
    # Wait for the first piece *before* answering. Until then a failure can still
    # be a real HTTP status (503, 504...). After it, the 200 is already on the wire.
    try:
        first = await events.__anext__()
    except LLMError as exc:
        raise to_http_error(exc) from None
    except StopAsyncIteration:
        raise to_http_error(LLMError("Stream ended without any event")) from None
    return StreamingResponse(
        _event_stream(first, events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # stops nginx-style proxies from holding pieces back
        },
    )


async def _event_stream(
    first: StreamEvent, rest: AsyncGenerator[StreamEvent, None]
) -> AsyncIterator[str]:
    try:
        yield _format(first)
        async for event in rest:
            yield _format(event)
    except LLMError as exc:
        # The status line is gone, so the failure travels as an event instead.
        yield _sse("error", {"code": exc.status, "message": to_http_error(exc).detail})
    except Exception:
        logger.exception("Unexpected failure while streaming a chat answer")
        yield _sse("error", {"code": "error", "message": "Something went wrong. Please try again."})
    finally:
        # Also runs when the client disconnects (the task is cancelled here), so the
        # provider connection is closed and we stop paying for words nobody reads.
        await rest.aclose()


def _format(event: StreamEvent) -> str:
    if isinstance(event, StreamDelta):
        return _sse("token", {"text": event.text})
    assert isinstance(event, StreamDone)
    return _sse(
        "done",
        {
            "model": event.model,
            "input_tokens": event.input_tokens,
            "output_tokens": event.output_tokens,
            "latency_ms": event.latency_ms,
            "first_piece_ms": event.first_piece_ms,
        },
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    # JSON on a single line, so a newline inside the text can never end the event.
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
