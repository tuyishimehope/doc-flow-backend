import asyncio
import json
import logging
import math
import os
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Path, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import StringConstraints
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.chat.schema import (
    ChatRequest,
    ChatResponse,
    ChatStatsResponse,
    ImageAnswerResponse,
    MAX_MESSAGE_CHARS,
    ModelsResponse,
    StreamChatRequest,
    TranscriptionResponse,
)
from app.core.config import settings
from app.core.observability import set_tag, set_user_id
from app.core.rate_limit import check_rate_limit
from app.core.stream_limit import acquire_stream_slot, release_stream_slot
from app.db.dependencies import get_db_session
from app.db.session import SessionLocal
from app.models.schema import Conversation
from app.service.auth.auth import CurrentUser
from app.service.chat.budget import DailyBudgetExceeded, check_daily_budget
from app.service.chat.chat import CHAT_PROMPT, answer_image_question, answer_message, auto_title, get_transcription, stream_answer, transcribe_audio_file, trim_history
from app.service.chat.conversation import SessionFactory, save_while_streaming
from app.service.chat.crud import (
    add_message,
    count_messages,
    create_conversation,
    get_conversation,
    get_recent_messages,
    rename_conversation,
)
from app.service.chat.schema import DEFAULT_STATS_DAYS, DEFAULT_TITLE, MAX_STATS_DAYS, MessageRole
from app.service.chat.stats import get_chat_stats
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
from app.service.llm_client.transcriber import ALLOWED_AUDIO_EXTENSIONS
from app.utils.chat_utils import convert_mb_to_bytes
from app.utils.image import clean_image, image_policy
from app.utils.upload import UploadRejected, read_checked_upload

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
    if isinstance(exc, (LLMUnavailableError, LLMNotConfiguredError)):
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


async def _token_budget(current_user: CurrentUser, db_session: AsyncSession) -> None:
    """429 when the user has used up today's tokens. Runs before any model call."""
    try:
        await check_daily_budget(current_user.id, db_session)
    except DailyBudgetExceeded as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"You have used your daily limit of {exc.limit:,} tokens. "
                f"It resets at {exc.resets_at:%H:%M} UTC (in {_hours_minutes(exc.retry_after)})."
            ),
            headers={"Retry-After": str(exc.retry_after)},
        ) from None


def _hours_minutes(seconds: int) -> str:
    hours, minutes = divmod(max(seconds, 0) // 60, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m"


# How long past the stream time limit a ticket is still trusted. After that the user is not
# locked out by a stream whose server died without handing the ticket back.
STREAM_SLOT_MARGIN_SECONDS = 30


async def _take_stream_slot(current_user: CurrentUser) -> str | None:
    """A ticket for one running stream, or 429 when the user already has the maximum.

    ``None`` means no ticket is held (limits are switched off), so there is nothing to give back.
    """
    if not settings.rate_limit_enabled:
        return None
    limit = settings.CHAT_MAX_CONCURRENT_STREAMS
    slot_id = await acquire_stream_slot(
        current_user.id,
        limit=limit,
        ttl_seconds=math.ceil(settings.CHAT_STREAM_TIMEOUT_SECONDS) + STREAM_SLOT_MARGIN_SECONDS,
    )
    if slot_id is None:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"You already have the maximum of {limit} answers streaming at once. "
            "Wait for one to finish, then try again.",
            headers={"Retry-After": "5"},
        )
    return slot_id


@CHAT_ROUTER.get("/stats", response_model=ChatStatsResponse)
async def stats_endpoint(
    current_user: CurrentUser,
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
    days: Annotated[int, Query(ge=1, le=MAX_STATS_DAYS)] = DEFAULT_STATS_DAYS,
) -> ChatStatsResponse:
    """The current user's own usage: tokens per day, p50/p95 latency and error rate.

    Only the user's own answers are counted. A view over everyone's usage needs an admin
    role, which the app does not have yet.
    """
    return ChatStatsResponse(**await get_chat_stats(current_user.id, days, db_session))


@CHAT_ROUTER.get("/models", response_model=ModelsResponse)
async def models_endpoint(current_user: CurrentUser) -> ModelsResponse:
    """The choices for the ``model`` field of a chat request."""
    return ModelsResponse(default=AUTO, models=list_models())  # type: ignore[arg-type]


@CHAT_ROUTER.post("", response_model=ChatResponse, responses=ERROR_RESPONSES)
async def chat_endpoint(
    body: ChatRequest,
    current_user: CurrentUser,
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
    pick_llm: Annotated[LLMPicker, Depends(get_llm_picker)],
) -> ChatResponse:
    await _rate_limit(current_user)
    await _token_budget(current_user, db_session)
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
    await _token_budget(current_user, db_session)
    _tag_request(current_user.id, body.model)
    llm = pick_llm(body.model)  # before any database work, so a bad choice creates nothing
    slot_id = await _take_stream_slot(current_user)  # after the cheap checks: a refusal creates nothing
    try:
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
            _event_stream(
                conversation.id,
                first,
                events,
                session_factory,
                current_user.id,
                slot_id,
                prompt_version=CHAT_PROMPT,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # stops nginx-style proxies from holding pieces back
            },
        )
    except BaseException:
        # The stream never started, so nothing else will give the ticket back.
        await release_stream_slot(current_user.id, slot_id)
        raise


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
    user_id: int | None = None,
    slot_id: str | None = None,
    prompt_version: str | None = None,
) -> AsyncIterator[str]:
    saving = save_while_streaming(
        first,
        rest,
        conversation_id=conversation_id,
        session_factory=session_factory,
        prompt_version=prompt_version,
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
        try:
            await saving.aclose()
        finally:
            if user_id is not None:
                await release_stream_slot(user_id, slot_id)  # the ticket goes back on every exit


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

TRANSCRIBE_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    **ERROR_RESPONSES,
    413: {"description": "The audio file is bigger than TRANSCRIBE_MAX_FILE_MB"},
    415: {"description": "The file extension is not an audio format the provider accepts"},
    422: {"description": "No file, or an empty file"},
    504: {"description": "The provider took longer than TRANSCRIBE_TIMEOUT_SECONDS"},
}


@CHAT_ROUTER.post("/transcribe", response_model=TranscriptionResponse, responses=TRANSCRIBE_ERROR_RESPONSES)
async def transcribe_audio(
    current_user: CurrentUser,
    file: UploadFile,
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
):
    await _rate_limit(current_user=current_user)
    await _token_budget(current_user=current_user, db_session=db_session)
    _tag_request(current_user.id, settings.TRANSCRIBE_MODEL)
    _validate_audio_upload(file)
    content = await file.read()
    try:
        saved = await transcribe_audio_file(
            user_id=current_user.id,
            filename=file.filename or "",
            content=content,
            content_type=file.content_type,
            db_session=db_session,
        )
    except LLMError as exc:
        raise to_http_error(exc) from None
    return saved


def _validate_audio_upload(file: UploadFile) -> None:
    """Cheap checks before anything is sent to the provider. Extension only, not content."""
    if not file.filename:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Please upload an audio file.")
    extension = os.path.splitext(file.filename)[1].lower().lstrip(".")
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported audio format. Use one of: " + ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS)) + ".",
        )
    if not file.size:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="The audio file is empty.")
    if file.size > convert_mb_to_bytes(settings.TRANSCRIBE_MAX_FILE_MB):
        raise HTTPException(
            413,
            detail=f"The audio file is too big. The limit is {settings.TRANSCRIBE_MAX_FILE_MB} MB.",
        )

@CHAT_ROUTER.get(
    "/transcribe/{transcription_id}",
    response_model=TranscriptionResponse,
    responses={404: {"description": "No such transcription, or it belongs to someone else"}},
)
async def get_transcription_endpoint(
    transcription_id: Annotated[int, Path(ge=1, le=2_147_483_647)],
    current_user: CurrentUser,
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
):
    transcription = await get_transcription(current_user.id, transcription_id, db_session)
    if transcription is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Transcription not found")
    return transcription


IMAGE_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    **ERROR_RESPONSES,
    413: {"description": "The image is bigger than VISION_MAX_IMAGE_MB, or has too many pixels"},
    415: {"description": "Not a jpg, jpeg, png or webp, or the content does not match the extension"},
    422: {"description": "No file, an empty file, a file that is not a valid image, or a bad question"},
    504: {"description": "The AI provider took longer than VISION_TIMEOUT_SECONDS"},
}


@CHAT_ROUTER.post("/image", response_model=ImageAnswerResponse, responses=IMAGE_ERROR_RESPONSES)
async def answer_with_image(
    current_user: CurrentUser,
    file: UploadFile,
    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_CHARS),
        Form(),
    ],
    db_session: Annotated[AsyncSession, Depends(get_db_session)],
):
    """One image and one question in, one answer out. The image is checked, cleaned and not kept."""
    await _rate_limit(current_user=current_user)
    await _token_budget(current_user=current_user, db_session=db_session)
    _tag_request(current_user.id, settings.VISION_MODEL)
    try:
        upload = await read_checked_upload(file, image_policy())
        image = await asyncio.to_thread(clean_image, upload.content)
    except UploadRejected as exc:
        raise HTTPException(exc.status_code, detail=exc.detail) from None
    try:
        return await answer_image_question(
            user_id=current_user.id,
            question=question,
            upload=upload,
            image=image,
            db_session=db_session,
        )
    except LLMError as exc:
        raise to_http_error(exc) from None
