import logging
import math
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.chat.schema import ChatRequest, ChatResponse
from app.core.config import settings
from app.core.rate_limit import check_rate_limit
from app.service.auth.auth import CurrentUser
from app.service.chat.chat import answer_message
from app.service.llm_client.llm_client import (
    LLMClient,
    LLMError,
    LLMNotConfiguredError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.service.llm_client.openrouter import get_llm_client

logger = logging.getLogger(__name__)

CHAT_ROUTER = APIRouter(prefix="/api/v1/chat", tags=["chat"])

UNAVAILABLE = "The assistant is temporarily unavailable. Please try again shortly."


def get_chat_llm() -> LLMClient:
    try:
        return get_llm_client()
    except LLMNotConfiguredError:
        logger.error("Chat is disabled because OPENROUTER_API_KEY is not set")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=UNAVAILABLE) from None


@CHAT_ROUTER.post(
    "",
    response_model=ChatResponse,
    responses={
        429: {"description": "Too many chat requests from this user"},
        502: {"description": "The AI provider rejected the request or returned nothing"},
        503: {"description": "The AI provider is unavailable or rate limiting us"},
        504: {"description": "The AI provider took too long"},
    },
)
async def chat_endpoint(
    body: ChatRequest,
    current_user: CurrentUser,
    llm: Annotated[LLMClient, Depends(get_chat_llm)],
) -> ChatResponse:
    await check_rate_limit(
        key=f"chat:{current_user.id}",
        limit=settings.CHAT_RATE_LIMIT_PER_MINUTE,
        window_seconds=60,
    )
    # The client logs the real cause. Callers only get a safe, generic message.
    try:
        result = await answer_message(body.message, llm)
    except LLMTimeoutError:
        raise HTTPException(
            status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The assistant took too long to answer. Please try again.",
        ) from None
    except LLMRateLimitError as exc:
        wait = exc.retry_after if exc.retry_after is not None else 10
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=UNAVAILABLE,
            headers={"Retry-After": str(max(1, math.ceil(wait)))},
        ) from None
    except LLMUnavailableError:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=UNAVAILABLE) from None
    except LLMError:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail="The assistant could not answer that. Please try again.",
        ) from None
    return ChatResponse(answer=result.text, model=result.model)
