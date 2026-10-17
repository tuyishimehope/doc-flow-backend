from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.chat.schema import (
    ConversationCreate,
    ConversationDetailResponse,
    ConversationResponse,
    ConversationUpdate,
    MessageResponse,
    PaginatedConversationResponse,
)
from app.db.dependencies import get_db_session
from app.service.auth.auth import CurrentUser
from app.service.chat.crud import (
    count_conversations,
    create_conversation,
    delete_conversation,
    get_conversation,
    get_messages,
    list_conversations,
    rename_conversation,
)
from app.service.chat.schema import DEFAULT_TITLE

CONVERSATIONS_ROUTER = APIRouter(prefix="/api/v1/chat/conversations", tags=["chat"])

DbSession = Annotated[AsyncSession, Depends(get_db_session)]
NOT_FOUND = HTTPException(status.HTTP_404_NOT_FOUND, detail="Conversation not found")


@CONVERSATIONS_ROUTER.post("", status_code=status.HTTP_201_CREATED, response_model=ConversationResponse)
async def create_conversation_endpoint(
    body: ConversationCreate, current_user: CurrentUser, db_session: DbSession
):
    """Start an empty conversation. Without a title it is named after its first message."""
    return await create_conversation(current_user.id, body.title or DEFAULT_TITLE, db_session)


@CONVERSATIONS_ROUTER.get("", response_model=PaginatedConversationResponse)
async def list_conversations_endpoint(
    current_user: CurrentUser,
    db_session: DbSession,
    skip: int = Query(default=0, ge=0, description="The items to skip"),
    limit: int = Query(default=10, ge=1, le=50, description="Items per page"),
):
    """The caller's conversations, most recently active first."""
    rows = await list_conversations(current_user.id, skip, limit, db_session)
    total = await count_conversations(current_user.id, db_session)
    return PaginatedConversationResponse(
        conversations=[ConversationResponse.model_validate(row) for row in rows],
        total=total,
        skip=skip,
        limit=limit,
        has_more=skip + len(rows) < total,
    )


@CONVERSATIONS_ROUTER.get("/{id}", response_model=ConversationDetailResponse)
async def get_conversation_endpoint(id: int, current_user: CurrentUser, db_session: DbSession):
    conversation = await get_conversation(id, current_user.id, db_session)
    if conversation is None:
        raise NOT_FOUND
    messages = await get_messages(conversation.id, db_session)
    return ConversationDetailResponse(
        **ConversationResponse.model_validate(conversation).model_dump(),
        messages=[MessageResponse.model_validate(message) for message in messages],
    )


@CONVERSATIONS_ROUTER.patch("/{id}", response_model=ConversationResponse)
async def rename_conversation_endpoint(
    id: int, body: ConversationUpdate, current_user: CurrentUser, db_session: DbSession
):
    conversation = await get_conversation(id, current_user.id, db_session)
    if conversation is None:
        raise NOT_FOUND
    return await rename_conversation(conversation, body.title, db_session)


@CONVERSATIONS_ROUTER.delete("/{id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation_endpoint(id: int, current_user: CurrentUser, db_session: DbSession):
    if not await delete_conversation(id, current_user.id, db_session):
        raise NOT_FOUND
    return Response(status_code=status.HTTP_204_NO_CONTENT)
