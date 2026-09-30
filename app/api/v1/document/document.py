from typing import Literal

from fastapi import APIRouter, Depends, UploadFile, HTTPException, status, Body, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.dependencies import get_db_session
from app.service.document.crud import delete_document_by_id, get_all_documents, get_document_by_id, get_jobs, get_processing_requests_by_document, get_total_no_of_documents, update_document_status
from app.service.document.document import process_document, queue_processing_request
from app.service.document.schema import DocumentProcessingResponse, DocumentUpdate, Document_Status, PaginatedDocumentResponse, ProcessRequest, Processing_Type, DocumentResponse, ProcessingJobResponse, ProcessingRequestResponse
from app.utils.document import valid_type_document, validate_document_content
from app.service.auth.auth import CurrentUser
from app.core.config import settings

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=DocumentProcessingResponse)
async def post_document_endpoint(file: UploadFile, current_user: CurrentUser, processing_type: Processing_Type = Body(), instructions: str | None = Body(default=None), db_session: AsyncSession = Depends(get_db_session)):
    result = valid_type_document(file=file)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File Format Not Accepted")

    content = await file.read(settings.MAX_UPLOAD_SIZE_BYTES + 1)
    if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.MAX_UPLOAD_SIZE_BYTES} byte upload limit",
        )
    if not validate_document_content(file, content):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File contents do not match the declared supported format",
        )
    await file.seek(0)
    response = await process_document(id=current_user.id, file=file, processing_type=processing_type, instructions=instructions, db_session=db_session)
    return response


@router.get("/{id}", response_model=DocumentResponse)
async def get_document_endpoint(id: int, current_user: CurrentUser, db_session: AsyncSession = Depends(get_db_session)):
    document = await get_document_by_id(id=id, db_session=db_session, user_id=current_user.id)
    if document is not None:
        return document

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                        detail="Document Not Found")


@router.get("", response_model=PaginatedDocumentResponse)
async def get_documents_endpoint(
    current_user: CurrentUser,
    skip: int = Query(default=0, ge=0, title="skip page", description="The items to skip "),
    limit: int = Query(default=10, title="limit", description="limit of items per page", ge=1, le=50),
    q: str | None = Query(default=None, max_length=255, description="Case-insensitive search on the document name"),
    status_filter: Literal[Document_Status.ACTIVE, Document_Status.ARCHIVED] | None = Query(default=None, alias="status"),
    db_session: AsyncSession = Depends(get_db_session),
):
    result = await get_all_documents(skip=skip, limit=limit, db_session=db_session, user_id=current_user.id, q=q, status=status_filter)
    total = await get_total_no_of_documents(db_session=db_session, user_id=current_user.id, q=q, status=status_filter)

    return PaginatedDocumentResponse(documents=[DocumentResponse.model_validate(document) for document in result],
                                     total=total, skip=skip, limit=limit, has_more=skip + len(result) < total)


@router.patch("/{id}", response_model=DocumentResponse)
async def update_document_endpoint(id: int, body: DocumentUpdate, current_user: CurrentUser, db_session: AsyncSession = Depends(get_db_session)):
    """Archive (status=ARCHIVED) or restore (status=ACTIVE) a document."""
    document = await update_document_status(id=id, status=body.status, current_user=current_user, db_session=db_session)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Document Not Found")
    return document


@router.post("/{id}/process", status_code=status.HTTP_201_CREATED, response_model=DocumentProcessingResponse)
async def reprocess_document_endpoint(id: int, body: ProcessRequest, current_user: CurrentUser, db_session: AsyncSession = Depends(get_db_session)):
    document = await get_document_by_id(id=id, db_session=db_session, user_id=current_user.id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Document Not Found")
    processing_request = await queue_processing_request(
        document=document, processing_type=body.processing_type, instructions=body.instructions, db_session=db_session)
    return DocumentProcessingResponse(
        document_id=document.id, processing_request_id=processing_request.id, status=processing_request.status)


@router.delete("/{id}", response_model=DocumentResponse)
async def delete_document_endpoint(current_user: CurrentUser, id: int, db_session: AsyncSession = Depends(get_db_session)):
    document = await delete_document_by_id(id=id, db_session=db_session, current_user=current_user)
    if document is not None:
        return document

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                        detail="Document Not Found")


@router.get("/{id}/jobs", response_model=list[ProcessingJobResponse])
async def get_status_jobs_endpoint(id: int, current_user: CurrentUser, db_session: AsyncSession = Depends(get_db_session)):
    response = await get_jobs(id=id, current_user=current_user, db_session=db_session)
    if response is not None:
        return [{"attempt": data.attempt_number, "status": data.status, "created_at": data.started_at, "completed_at": data.completed_at, "failure_reason": data.failure_reason} for data in response]
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")


@router.get("/{id}/processing-requests", response_model=list[ProcessingRequestResponse])
async def get_processing_requests_endpoint(id: int, current_user: CurrentUser, db_session: AsyncSession = Depends(get_db_session)):
    response = await get_processing_requests_by_document(id=id, current_user=current_user, db_session=db_session)
    if response is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Document Not Found")
    return response
