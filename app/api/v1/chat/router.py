import json
import logging
import math
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.chat.schema import ChatRequest, ChatResponse, ModelsResponse, StreamChatRequest
from app.core.config import settings
from app.core.observability import set_tag, set_user_id
from app.core.rate_limit import check_rate_limit
from app.db.dependencies import get_db_session
from app.db.session import SessionLocal
from app.models.schema import Conversation
from app.service.auth.auth import CurrentUser
from app.service.chat.chat import answer_message, auto_title, stream_answer, trim_history
from app.service.chat.conversation import SessionFactory, save_while_streaming
from app.service.chat.crud import (
    add_message,
    count_messages,
    create_conversation,
    get_conversation,
    get_recent_messages,
    rename_conversation,
)
from app.service.chat.schema import DEFAULT_TITLE, MessageRole
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
from app.service.llm_client.registry import AUTO, ModelUnavailableError, get_llm_client, list_models

logger = logging.getLogger(__name__)

CHAT_ROUTER = APIRouter(prefix="/api/v1/chat", tags=["chat"])

UNAVAILABLE = "The assistant is temporarily unavailable. Please try again shortly."

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    429: {"description": "Too many chat requests from this user"},
    502: {"description": "The AI provider rejected the request or returned nothing"},
    503: {"description": "The AI provider is unavailable or rate limiting us"},
    504: {"description": "The AI provider took too long"},
}


LLMPicker = Callable[[str | None], LLMClient]


def get_llm_picker() -> LLMPicker:
    """Returns the function that turns the request's ``model`` into an LLM client."""

    def pick(model: str | None) -> LLMClient:
        try:
            return get_llm_client(model)
        except LLMNotConfiguredError:
            logger.error("Chat is disabled because no LLM provider API key is set")
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=UNAVAILABLE) from None
        except ModelUnavailableError:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, detail="That model is not available right now."
            ) from None

    return pick


def get_session_factory() -> SessionFactory:
    """The streaming answer is saved with its own session, because the request's is gone."""
    return SessionLocal


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


def _tag_request(user_id: int, model: str | None) -> None:
    """Label error reports with who (numeric id only) and which model choice."""
    set_user_id(user_id)
    set_tag("chat.model_choice", model or AUTO)


async def _rate_limit(current_user: CurrentUser) -> None:
    await check_rate_limit(
        key=f"chat:{current_user.id}",
        limit=settings.CHAT_RATE_LIMIT_PER_MINUTE,
        window_seconds=60,
    )


@CHAT_ROUTER.get("/models", response_model=ModelsResponse)
async def models_endpoint(current_user: CurrentUser) -> ModelsResponse:
    """The choices for the ``model`` field of a chat request."""
    return ModelsResponse(default=AUTO, models=list_models())  # type: ignore[arg-type]


@CHAT_ROUTER.post("", response_model=ChatResponse, responses=ERROR_RESPONSES)
async def chat_endpoint(
    body: ChatRequest,
    current_user: CurrentUser,
    pick_llm: Annotated[LLMPicker, Depends(get_llm_picker)],
) -> ChatResponse:
    await _rate_limit(current_user)
    _tag_request(current_user.id, body.model)
    llm = pick_llm(body.model)
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
        404: {"description": "The conversation does not exist or belongs to someone else"},
        200: {
            "description": (
                "Server-Sent Events: `conversation`, then `token`s, then `done` or `error`"
            ),
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        },
    },
)
async def chat_stream_endpoint(
    body: StreamChatRequest,
    current_user: CurrentUser,
    pick_llm: Annotated[LLMPicker, Depends(get_llm_picker)],
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
    session_factory: Annotated[SessionFactory, Depends(get_session_factory)],
) -> StreamingResponse:
    await _rate_limit(current_user)
    _tag_request(current_user.id, body.model)
    llm = pick_llm(body.model)  # before any database work, so a bad choice creates nothing
    conversation = await _conversation_for(body, current_user.id, db_session)

    # The earlier turns come first; the new message is saved right after, so it
    # is never lost even if the model call fails or the server dies.
    earlier = await get_recent_messages(conversation.id, settings.CHAT_HISTORY_MESSAGES, db_session)
    history = trim_history(
        [{"role": m.role, "content": m.content} for m in earlier],  # type: ignore[misc]
        settings.CHAT_HISTORY_MAX_CHARS,
    )
    await add_message(
        conversation_id=conversation.id,
        role=MessageRole.USER,
        content=body.message,
        db_session=db_session,
    )

    events = stream_answer(body.message, llm, history)
    # Wait for the first piece *before* answering. Until then a failure can still
    # be a real HTTP status (503, 504...). After it, the 200 is already on the wire.
    try:
        first = await events.__anext__()
    except LLMError as exc:
        raise to_http_error(exc) from None
    except StopAsyncIteration:
        raise to_http_error(LLMError("Stream ended without any event")) from None
    return StreamingResponse(
        _event_stream(conversation.id, first, events, session_factory),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # stops nginx-style proxies from holding pieces back
        },
    )


async def _conversation_for(
    body: StreamChatRequest, user_id: int, db_session: AsyncSession
) -> Conversation:
    if body.conversation_id is None:
        return await create_conversation(user_id, auto_title(body.message), db_session)
    conversation = await get_conversation(body.conversation_id, user_id, db_session)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    if conversation.title == DEFAULT_TITLE and await count_messages(conversation.id, db_session) == 0:
        conversation = await rename_conversation(conversation, auto_title(body.message), db_session)
    return conversation


async def _event_stream(
    conversation_id: int,
    first: StreamEvent,
    rest: AsyncGenerator[StreamEvent, None],
    session_factory: SessionFactory,
) -> AsyncIterator[str]:
    saving = save_while_streaming(
        first, rest, conversation_id=conversation_id, session_factory=session_factory
    )
    try:
        yield _sse("conversation", {"conversation_id": conversation_id})
        async for event in saving:
            yield _format(event)
    except LLMError as exc:
        # The status line is gone, so the failure travels as an event instead.
        yield _sse("error", {"code": exc.status, "message": to_http_error(exc).detail})
    except Exception:
        logger.exception("Unexpected failure while streaming a chat answer")
        yield _sse("error", {"code": "error", "message": "Something went wrong. Please try again."})
    finally:
        # Also runs when the client disconnects: the provider connection is closed
        # and the part of the answer written so far is saved as "cancelled".
        await saving.aclose()


def _format(event: StreamEvent) -> str:
    if isinstance(event, StreamDelta):
        return _sse("token", {"text": event.text})
    assert isinstance(event, StreamDone)
    return _sse(
        "done",
        {
            "provider": event.provider,
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
