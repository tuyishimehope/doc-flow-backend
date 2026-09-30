from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.dependencies import get_db_session
from app.service.auth.auth import CurrentUser
from app.service.document.crud import get_processing_request_by_id
from app.service.document.document import get_processing_result, get_processing_status
from app.service.document.schema import ProcessingRequestResponse, ProcessingResultResponse, ProcessingStatusResponse


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
