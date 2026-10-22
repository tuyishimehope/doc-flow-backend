"""Ask questions about a PDF that was stored once (plan_4, part B).

``POST /documents/qa`` keeps the PDF and its page text; ``POST /documents/{id}/ask`` answers
from the stored pages with checked citations; ``GET /documents/{id}/file`` gives the original
back so a viewer can open the cited page. Only the owner can use any of them.
"""

import asyncio
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from minio.error import S3Error
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.chat.router import (
    ERROR_RESPONSES,
    _rate_limit,
    _tag_request,
    _token_budget,
    to_http_error,
)
from app.api.v1.chat.schema import CitationResponse, DocumentAnswerResponse
from app.api.v1.document.qa_schema import AskDocumentRequest, StoredDocumentResponse
from app.core.config import settings
from app.db.dependencies import get_db_session
from app.service.auth.auth import CurrentUser
from app.service.chat.chat import ask_stored_document
from app.service.chat.crud import get_conversation
from app.service.document.crud import get_document_by_id
from app.service.document.pages import get_pages
from app.service.document.qa import store_pdf
from app.service.file.file import get_file_by_id
from app.service.llm_client.llm_client import LLMError
from app.utils.http import attachment_disposition
from app.utils.pdf import extract_pages, pdf_policy
from app.utils.scan import scan_upload
from app.utils.upload import UploadRejectedError, read_checked_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

DbSession = Annotated[AsyncSession, Depends(get_db_session)]

NOT_FOUND = "Document not found"
NOT_PREPARED = (
    "This document was not stored for questions. "
    "Upload the PDF again with POST /api/v1/documents/qa."
)

STORE_RESPONSES: dict[int | str, dict[str, Any]] = {
    429: ERROR_RESPONSES[429],
    413: {
        "description": (
            "The PDF is bigger than DOCUMENT_QA_MAX_MB, "
            "or has more than DOCUMENT_QA_MAX_PAGES pages"
        )
    },
    415: {"description": "Not a .pdf, or the content does not match the extension"},
    422: {
        "description": (
            "No file, an empty file, a PDF that is encrypted, damaged or scanned (no text), "
            "or a file rejected by the virus scan"
        )
    },
}

ASK_RESPONSES: dict[int | str, dict[str, Any]] = {
    **ERROR_RESPONSES,
    404: {"description": "No such document or conversation for this user"},
    409: {
        "description": (
            "The document was not stored for questions, "
            "or the conversation is about a different document"
        )
    },
    429: {"description": "Too many requests, or the daily token budget is used up"},
}


@router.post(
    "/qa",
    status_code=status.HTTP_201_CREATED,
    response_model=StoredDocumentResponse,
    responses=STORE_RESPONSES,
)
async def store_document_for_questions(
    current_user: CurrentUser, file: UploadFile, db_session: DbSession
):
    """Upload a PDF once. Its text is read page by page and kept, so it can be asked about."""
    await _rate_limit(current_user=current_user)
    _tag_request(current_user.id, settings.DOCUMENT_MODEL)
    try:
        upload = await read_checked_upload(file, pdf_policy())
        await scan_upload(upload.content)
        pages = await asyncio.to_thread(
            extract_pages, upload.content, settings.DOCUMENT_QA_MAX_PAGES
        )
    except UploadRejectedError as exc:
        raise HTTPException(exc.status_code, detail=exc.detail) from None
    document = await store_pdf(
        user_id=current_user.id, upload=upload, pages=pages, db_session=db_session
    )
    return StoredDocumentResponse(
        document_id=document.id,
        file_id=document.file_id,
        name=document.name,
        pages=len(pages),
    )


@router.post("/{id}/ask", response_model=DocumentAnswerResponse, responses=ASK_RESPONSES)
async def ask_document(
    id: int, body: AskDocumentRequest, current_user: CurrentUser, db_session: DbSession
):
    """Ask a question about a stored PDF; every citation was checked against the page text.

    Send ``conversation_id`` to ask a follow-up in the same conversation.
    """
    await _rate_limit(current_user=current_user)
    await _token_budget(current_user=current_user, db_session=db_session)
    _tag_request(current_user.id, settings.DOCUMENT_MODEL)

    document = await get_document_by_id(id=id, db_session=db_session, user_id=current_user.id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=NOT_FOUND)
    pages = await get_pages(document.id, db_session)
    if not pages:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=NOT_PREPARED)

    conversation = None
    if body.conversation_id is not None:
        conversation = await get_conversation(body.conversation_id, current_user.id, db_session)
        if conversation is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        if conversation.document_id != document.id:
            raise HTTPException(
                status.HTTP_409_CONFLICT, detail="That conversation is about a different document."
            )

    try:
        answer = await ask_stored_document(
            user_id=current_user.id,
            document_id=document.id,
            question=body.question,
            pages=pages,
            conversation=conversation,
            db_session=db_session,
        )
    except LLMError as exc:
        raise to_http_error(exc) from None
    return DocumentAnswerResponse(
        conversation_id=answer.conversation_id,
        answer=answer.answer,
        found=answer.found,
        grounded=answer.grounded,
        citations=[CitationResponse(page=c.page, quote=c.quote) for c in answer.citations],
        pages=answer.pages,
        model=answer.model,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
    )


@router.get(
    "/{id}/file",
    responses={200: {"content": {"application/pdf": {}}}, 404: {"description": "Not found"}},
)
async def get_document_file(id: int, current_user: CurrentUser, db_session: DbSession):
    """The original file, for its owner only (a viewer can open it from a blob URL)."""
    document = await get_document_by_id(id=id, db_session=db_session, user_id=current_user.id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=NOT_FOUND)
    try:
        result = await get_file_by_id(
            id=document.file_id, current_user=current_user, db_session=db_session
        )
    except S3Error:
        logger.exception("Stored file is missing", extra={"document_id": document.id})
        result = None
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="File not found")
    return StreamingResponse(
        content=result["content"],
        media_type=result["content_type"],
        headers={
            "Content-Disposition": attachment_disposition(result["name"]),
            "X-Content-Type-Options": "nosniff",
        },
    )
