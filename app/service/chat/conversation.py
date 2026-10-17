"""Saving the assistant's answer while it streams."""

import asyncio
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.service.chat.crud import add_message
from app.service.chat.schema import MessageRole, MessageStatus
from app.service.llm_client.llm_client import StreamDelta, StreamDone, StreamEvent

logger = logging.getLogger(__name__)

# Anything that opens a database session: ``async with session_factory() as session``.
SessionFactory = Callable[[], Any]

# Saves that must finish even though the request that started them was cancelled.
_background: set[asyncio.Task[Any]] = set()


async def _save(
    session_factory: SessionFactory,
    conversation_id: int,
    text: str,
    status: MessageStatus,
    model: str | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    # A new session: the request's own session may already be closed by now.
    async with session_factory() as session:
        session: AsyncSession
        await add_message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=text,
            status=status,
            model=model,
            meta=meta,
            db_session=session,
        )


async def _finish_even_if_cancelled(work: Awaitable[None]) -> None:
    """Run ``work`` as its own task, so cancelling the caller cannot cancel the save."""
    task = asyncio.ensure_future(work)
    _background.add(task)
    task.add_done_callback(_background.discard)
    try:
        await asyncio.shield(task)
    except Exception:
        logger.exception("Could not save the cut-off assistant message")


async def save_while_streaming(
    first: StreamEvent,
    rest: AsyncGenerator[StreamEvent, None],
    *,
    conversation_id: int,
    session_factory: SessionFactory,
) -> AsyncGenerator[StreamEvent, None]:
    """Pass events through unchanged and store the assistant's message.

    * The stream finishes: saved as ``complete`` *before* the ``done`` event goes out,
      so a client that saw ``done`` knows the answer is stored.
    * The provider fails after some text: the text so far is saved as ``error``.
    * The client leaves mid-answer: the text so far is saved as ``cancelled``.
    """
    parts: list[str] = []
    saved = False
    outcome = MessageStatus.CANCELLED  # what happened, if we are interrupted
    try:
        event = first
        while True:
            if isinstance(event, StreamDelta):
                parts.append(event.text)
            elif isinstance(event, StreamDone):
                await _save(
                    session_factory,
                    conversation_id,
                    "".join(parts),
                    MessageStatus.COMPLETE,
                    model=event.model,
                    meta={
                        "provider": event.provider,
                        "input_tokens": event.input_tokens,
                        "output_tokens": event.output_tokens,
                        "latency_ms": event.latency_ms,
                        "first_piece_ms": event.first_piece_ms,
                    },
                )
                saved = True
            yield event
            if saved:
                return
            try:
                event = await rest.__anext__()
            except StopAsyncIteration:
                outcome = MessageStatus.ERROR  # ended without a ``done`` event
                return
    except Exception:
        outcome = MessageStatus.ERROR
        raise
    finally:
        try:
            await rest.aclose()  # stop the provider call first: it is the part that costs money
        finally:
            if not saved and parts:
                await _finish_even_if_cancelled(
                    _save(session_factory, conversation_id, "".join(parts), outcome)
                )
