from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.dependencies import get_db_session
from app.service.auth.auth import CurrentUser
from app.service.file.crud import get_count
from app.service.file.file import get_all_files, get_file_by_id, delete_file_by_id
from app.service.file.schema import FileResponse, PaginatedFileResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/files", tags=["files"])


class FileDeleteResponse(BaseModel):
    file_id: int
    status: str


@router.get("/{id}", status_code=status.HTTP_200_OK, summary="Get a file by id", description="You can get a file by id", responses={
    200: {
            "content": {
                "application/pdf": {}
            }
            }
})
async def get_file_endpoint(id: int, current_user: CurrentUser, db_session: AsyncSession = Depends(get_db_session)):
    """
    Get a file by id. The binary response may be PDF, DOCX, or an image.

    Args:
        id (int): id of the file
        db_session (AsyncSession, optional): db session. Defaults to Depends(get_db_session).

    Raises:
        HTTPException: return expection when the file not found

    Returns:
        file: Returns a file 
    """
    result = await get_file_by_id(id=id, current_user=current_user, db_session=db_session)

    if result:
        return Response(
            content=result["content"],
            media_type=result["content_type"],
            headers={
                "Content-Disposition": f'attachment; filename="{result["name"]}"'
            }
        )

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                        detail="file not found")


@router.get("", response_model=PaginatedFileResponse)
async def get_files(current_user: CurrentUser, db_session: Annotated[AsyncSession, Depends(get_db_session)], limit: int = Query(default=10, ge=1, le=50), skip: int = Query(default=0, ge=0, le=50)):
    result = await get_all_files(limit=limit, skip=skip, user_id=current_user.id, db_session=db_session)
    total = await get_count(user_id=current_user.id,db_session=db_session)
    return PaginatedFileResponse(files=[FileResponse.model_validate(file) for file in result],
                                 total=total,
                                 skip=skip,
                                 limit=limit,
                                 has_more = skip + len(result) < total)


@router.delete("/{id}", response_model=FileDeleteResponse)
async def delete_file(id: int, current_user: CurrentUser, db_session: Annotated[AsyncSession, Depends(get_db_session)]):
    result = await delete_file_by_id(id=id, current_user=current_user, db_session=db_session)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return FileDeleteResponse(file_id=result, status="DELETED")
