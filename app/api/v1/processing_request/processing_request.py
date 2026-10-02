from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.dependencies import get_db_session
from app.service.auth.auth import CurrentUser
from app.service.document.crud import cancel_processing_request, get_document_by_id, get_processing_request_by_id
from app.service.document.document import export_result, get_processing_result, get_processing_status, queue_processing_request
from app.service.document.schema import RETRYABLE_STATUSES, DocumentProcessingResponse, ExportFormat, Processing_status, ProcessingRequestResponse, ProcessingResultResponse, ProcessingStatusResponse
from app.utils.http import attachment_disposition


router = APIRouter(prefix="/api/v1/processing-requests",
                   tags=["processing-request"])


@router.get("/status/{id}", response_model=ProcessingStatusResponse)
async def get_status_endpoint(
    id: int,
    current_user: CurrentUser,
    db_session: AsyncSession = Depends(get_db_session)
):
    result = await get_processing_status(
        id,
        current_user,
        db_session
    )
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Processing request with id: {id} not found")
    return result


@router.get("/result/{id}", response_model=ProcessingResultResponse)
async def get_result_endpoint(
    id: int,
    current_user: CurrentUser,
    db_session: AsyncSession = Depends(get_db_session)
):
    result = await get_processing_result(
        id,
        current_user,
        db_session
    )
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Processing request with id: {id} not found")
    return result


@router.get("/{id}", response_model=ProcessingRequestResponse)
async def get_processing_request_endpoint(id: int, current_user: CurrentUser, db_session: AsyncSession = Depends(get_db_session)):
    response = await get_processing_request_by_id(id=id, current_user=current_user, db_session=db_session)
    if response is not None:
        return response
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                        detail="Request not found")


@router.post("/{id}/cancel", response_model=ProcessingStatusResponse)
async def cancel_processing_request_endpoint(id: int, current_user: CurrentUser, db_session: AsyncSession = Depends(get_db_session)):
    processing_request = await get_processing_request_by_id(id=id, current_user=current_user, db_session=db_session)
    if processing_request is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Request not found")
    if not await cancel_processing_request(id=id, db_session=db_session):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Only requests that have not started can be cancelled")
    return ProcessingStatusResponse(id=id, status=Processing_status.CANCELLED)


@router.post("/{id}/retry", status_code=status.HTTP_201_CREATED, response_model=DocumentProcessingResponse)
async def retry_processing_request_endpoint(id: int, current_user: CurrentUser, db_session: AsyncSession = Depends(get_db_session)):
    """Queue a new request with the same type and instructions as a failed or cancelled one."""
    processing_request = await get_processing_request_by_id(id=id, current_user=current_user, db_session=db_session)
    if processing_request is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Request not found")
    if processing_request.status not in RETRYABLE_STATUSES:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Only failed or cancelled requests can be retried")

    document = await get_document_by_id(id=processing_request.document_id, db_session=db_session, user_id=current_user.id)
    new_request = await queue_processing_request(
        document=document,
        processing_type=processing_request.processing_type,
        instructions=processing_request.instructions,
        db_session=db_session,
    )
    return DocumentProcessingResponse(
        document_id=document.id, processing_request_id=new_request.id, status=new_request.status)


@router.get("/{id}/export", response_class=Response)
async def export_result_endpoint(
    id: int,
    current_user: CurrentUser,
    export_format: ExportFormat = Query(default=ExportFormat.JSON, alias="format"),
    db_session: AsyncSession = Depends(get_db_session),
):
    result = await get_processing_result(id, current_user, db_session)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="No result for this processing request")
    content, media_type, filename = export_result(id, result["result"], export_format)
    return Response(content=content, media_type=media_type,
                    headers={"Content-Disposition": attachment_disposition(filename)})
