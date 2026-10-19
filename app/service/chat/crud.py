"""Database access for conversations and messages.

Every conversation query is scoped by ``user_id``: asking for someone else's
conversation looks exactly like asking for one that does not exist.
"""

from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema import Conversation, Message
from app.service.chat.schema import CallUsage, MessageRole, MessageStatus


async def create_conversation(user_id: int, title: str, db_session: AsyncSession) -> Conversation:
    conversation = Conversation(user_id=user_id, title=title)
    db_session.add(conversation)
    await db_session.flush()
    await db_session.refresh(conversation)  # load the server-generated timestamps
    await db_session.commit()
    return conversation


async def get_conversation(id: int, user_id: int, db_session: AsyncSession) -> Conversation | None:
    result = await db_session.execute(
        select(Conversation).where(Conversation.id == id, Conversation.user_id == user_id)
    )
    return result.scalar_one_or_none()


async def list_conversations(
    user_id: int, skip: int, limit: int, db_session: AsyncSession
) -> list[Conversation]:
    result = await db_session.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .offset(skip)
        .limit(limit)
    )
    return list(result.scalars().all())


async def count_conversations(user_id: int, db_session: AsyncSession) -> int:
    result = await db_session.execute(
        select(func.count()).select_from(Conversation).where(Conversation.user_id == user_id)
    )
    return result.scalar_one()


async def rename_conversation(
    conversation: Conversation, title: str, db_session: AsyncSession
) -> Conversation:
    conversation.title = title
    await db_session.flush()
    await db_session.refresh(conversation)
    await db_session.commit()
    return conversation


async def delete_conversation(id: int, user_id: int, db_session: AsyncSession) -> bool:
    conversation = await get_conversation(id, user_id, db_session)
    if conversation is None:
        return False
    # Explicit, so it does not depend on the database enforcing ON DELETE CASCADE.
    await db_session.execute(delete(Message).where(Message.conversation_id == id))
    await db_session.execute(delete(Conversation).where(Conversation.id == id))
    await db_session.commit()
    return True


async def add_message(
    *,
    conversation_id: int,
    role: MessageRole,
    content: str,
    db_session: AsyncSession,
    status: MessageStatus = MessageStatus.COMPLETE,
    model: str | None = None,
    meta: dict[str, Any] | None = None,
) -> Message:
    usage: CallUsage | None = None,
    message = Message(
    usage = usage or CallUsage()
        conversation_id=conversation_id,
        role=role.value,
        content=content,
        status=status.value,
        model=model,
        meta=meta,
        provider=usage.provider or None,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_tokens=usage.cached_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        latency_ms=usage.latency_ms,
    )
    db_session.add(message)
    await db_session.execute(
        update(Conversation).where(Conversation.id == conversation_id).values(updated_at=func.now())
    )
    await db_session.flush()
    await db_session.refresh(message)
    await db_session.commit()
    return message


async def count_messages(conversation_id: int, db_session: AsyncSession) -> int:
    result = await db_session.execute(
        select(func.count()).select_from(Message).where(Message.conversation_id == conversation_id)
    )
    return result.scalar_one()


async def get_messages(conversation_id: int, db_session: AsyncSession) -> list[Message]:
    result = await db_session.execute(
        select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
    )
    return list(result.scalars().all())


async def get_recent_messages(
    conversation_id: int, limit: int, db_session: AsyncSession
) -> list[Message]:
    """The latest ``limit`` complete messages, oldest first. Cut-off answers are left out."""
    if limit <= 0:
        return []
    result = await db_session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id, Message.status == MessageStatus.COMPLETE.value)
        .order_by(Message.id.desc())
        .limit(limit)
    )
    return list(reversed(result.scalars().all()))
