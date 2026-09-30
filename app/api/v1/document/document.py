from fastapi import APIRouter, Depends, UploadFile, HTTPException, status, Body, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.dependencies import get_db_session
from app.service.document.crud import get_total_no_of_documents
from app.service.document.document import delete_document, get_document, get_documents, get_status_jobs, process_document
from app.service.document.schema import DocumentProcessingResponse, PaginatedDocumentResponse, Processing_Type, DocumentResponse, ProcessingJobResponse
from app.utils.document import valid_type_document, validate_document_content
from app.service.auth.auth import CurrentUser
from app.core.config import settings

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=DocumentProcessingResponse)
async def post_document_endpoint(file: UploadFile, current_user: CurrentUser, processing_type: Processing_Type = Body(), instructions: str = Body(), db_session: AsyncSession = Depends(get_db_session)):
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
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Not authorized to view document")

    document = await get_document(id=id, db_session=db_session, user_id=current_user.id)
    if document is not None:
        return DocumentResponse(id=document.id, name=document.name, status=document.status, file_id=document.file_id, user_id=document.user_id, created_at=document.created_at, updated_at=document.updated_at)

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                        detail="Document Not Found")


@router.get("", response_model=PaginatedDocumentResponse)
async def get_documents_endpoint(current_user: CurrentUser, skip: int = Query(default=0, ge=0, le=50, title="skip page", description="The items to skip "), limit: int = Query(default=10, title="limit", description="limit of items per page", gt=1, le=50),  db_session: AsyncSession = Depends(get_db_session)):
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Not authorized to view document")
    result = await get_documents(skip=skip, limit=limit, db_session=db_session, user_id=current_user.id)
    total = await get_total_no_of_documents(db_session=db_session, user_id=current_user.id)

    return PaginatedDocumentResponse(documents=[DocumentResponse.model_validate(document) for document in result],
                                     total=total, skip=skip, limit=limit, has_more=skip + len(result) < total)


@router.delete("/{id}", response_model=DocumentResponse)
async def delete_document_endpoint(current_user: CurrentUser, id: int, db_session: AsyncSession = Depends(get_db_session)):
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Not authorized to view document")

    document = await delete_document(id=id, db_session=db_session, current_user=current_user)
    if document is not None:
        return DocumentResponse(id=document.id, name=document.name, status=document.status, file_id=document.file_id, user_id=document.user_id, created_at=document.created_at, updated_at=document.updated_at)

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                        detail="Document Not Found")


@router.get("/{id}/jobs", response_model=list[ProcessingJobResponse])
async def get_status_jobs_endpoint(id: int, current_user: CurrentUser, db_session: AsyncSession = Depends(get_db_session)):
    response = await get_status_jobs(id=id, current_user=current_user, db_session=db_session)
    if response is not None:
        return [{"attempt": data.attempt_number, "status": data.status, "created_at": data.started_at, "completed_at": data.completed_at} for data in response]
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
