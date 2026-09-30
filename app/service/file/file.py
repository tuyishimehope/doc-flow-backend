import logging

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool
import datetime

from app.core.minio import minio_client
from app.service.auth.auth import CurrentUser
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.service.file.crud import get_file_id, get_all_files_by_user
from app.core.config import settings
from app.models.schema import Document
from app.service.document.schema import Document_Status

logger = logging.getLogger(__name__)

BUCKET_NAME = settings.MINIO_BUCKET


async def post_file(file: UploadFile, file_id: str):

    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)

    # The MinIO client is blocking; keep it off the event loop.
    await run_in_threadpool(
        minio_client.put_object,
        bucket_name=BUCKET_NAME,
        object_name=file_id,
        data=file.file,
        length=size,
        content_type=file.content_type or "",
    )

    return {
        "file_id": file_id
    }


def get_file(file_id: int):
    return minio_client.get_object(BUCKET_NAME, str(file_id))


def delete_file(file_id: int):
    minio_client.remove_object(BUCKET_NAME, str(file_id))


async def get_file_by_id(id: int, current_user: CurrentUser, db_session: AsyncSession):

    file_record = await get_file_id(id=id, current_user=current_user, db_session=db_session)

    if not file_record:
        return None

    response = await run_in_threadpool(get_file, id)

    def stream_content():
        try:
            yield from response.stream(64 * 1024)
        finally:
            response.close()
            response.release_conn()

    return {
        "name": file_record.name,
        "content": stream_content(),
        "content_type": file_record.content_type
    }


async def delete_file_by_id(id: int, current_user: CurrentUser, db_session: AsyncSession):
    file_record = await get_file_id(
        id=id,
        current_user=current_user,
        db_session=db_session,
        include_deleted=True,
    )

    if not file_record:
        return None

    deleted_at = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    file_record.deleted_at = file_record.deleted_at or deleted_at
    document_result = await db_session.execute(
        select(Document).where(
            Document.file_id == id,
            Document.user_id == current_user.id,
        )
    )
    document = document_result.scalar_one_or_none()
    if document is not None:
        document.status = Document_Status.DELETED
        document.deleted_at = document.deleted_at or deleted_at

    try:
        await db_session.commit()
    except Exception:
        await db_session.rollback()
        raise

    try:
        await run_in_threadpool(delete_file, id)
    except Exception:
        # Repeating DELETE retries object cleanup after the records are tombstoned.
        logger.exception(
            "Could not remove file object from storage",
            extra={"file_id": id, "document_id": document.id if document else None},
        )
    logger.info(
        "File and parent document soft-deleted",
        extra={"file_id": id, "document_id": document.id if document else None, "user_id": current_user.id},
    )
    return id
    
async def get_all_files(limit: int, skip: int, user_id: int, db_session: AsyncSession):
    result = await get_all_files_by_user(limit=limit, skip=skip, user_id=user_id, db_session=db_session)
    return result
