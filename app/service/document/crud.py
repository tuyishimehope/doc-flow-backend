from datetime import datetime, timezone
import logging

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema import Document, Extracted_Result, File, Processing_Request, Processing_Job, User
from app.service.auth.auth import CurrentUser
from app.service.document.schema import Document_Status, Processing_status
from app.service.file.file import delete_file

logger = logging.getLogger(__name__)


async def save_file(file, file_name, db_session: AsyncSession):
    file_object = File(name=file.filename,
                       content_type=file.content_type, extension=file_name)

    db_session.add(file_object)
    await db_session.flush()

    return file_object


async def save_document(user_id, file, file_object, db_session: AsyncSession):
    document_object = Document(
        user_id=user_id, name=file.filename, file=file_object)

    db_session.add(document_object)
    await db_session.flush()

    return document_object


async def save_processing_request(document_object, processing_type, instructions, db_session: AsyncSession):
    processing_request_object = Processing_Request(
        document_id=document_object.id, processing_type=processing_type, instructions=instructions, status=Processing_status.PENDING)

    db_session.add(processing_request_object)
    await db_session.commit()

    return processing_request_object


async def get_processing_request_status(processing_request_id: int, current_user: CurrentUser, db_session: AsyncSession):

    request = (
        Select(Processing_Request)
        .join(Document)
        .where(
            Processing_Request.id == processing_request_id,
            Document.user_id == current_user.id,
            Document.deleted_at.is_(None),
            Document.status != Document_Status.DELETED,
        )
    )

    result = await db_session.execute(request)
    response = result.scalar_one_or_none()
    return response


async def get_processing_request_result(processing_request_id: int, current_user: CurrentUser, db_session: AsyncSession):
    request = (
        Select(Extracted_Result)
        .join(Processing_Request)
        .join(Document)
        .where(
            Extracted_Result.processing_request_id
            == processing_request_id,
            Document.user_id == current_user.id,
            Document.deleted_at.is_(None),
            Document.status != Document_Status.DELETED,
        )
    )
    result = await db_session.execute(request)
    response = result.scalar_one_or_none()
    return response


async def get_document_by_id(
    id: int,
    db_session: AsyncSession,
    user_id: int,
    include_deleted: bool = False,
) -> Document | None:
    statement = Select(Document).where(Document.id == id, Document.user_id == user_id)
    if not include_deleted:
        statement = statement.where(
            Document.deleted_at.is_(None),
            Document.status != Document_Status.DELETED,
        )
    result = await db_session.execute(statement)
    response = result.scalar_one_or_none()
    return response


async def get_all_documents(skip: int, limit: int, db_session: AsyncSession, user_id: int) -> list[Document] :
    statement = (
        Select(Document)
        .where(
            Document.user_id == user_id,
            Document.deleted_at.is_(None),
            Document.status != Document_Status.DELETED,
        )
        .offset(skip)
        .limit(limit)
    )
    result = await db_session.execute(statement)
    response = result.scalars().all()
    return list(response)


async def get_total_no_of_documents(db_session: AsyncSession, user_id: int) -> int:
    statement = Select(func.count(Document.id)).where(
        Document.user_id == user_id,
        Document.deleted_at.is_(None),
        Document.status != Document_Status.DELETED,
    )
    result = await db_session.execute(statement)
    response = result.scalar_one()
    return response


async def delete_document_by_id(id: int, current_user: CurrentUser, db_session: AsyncSession):
    document = await get_document_by_id(
        id=id,
        db_session=db_session,
        user_id=current_user.id,
        include_deleted=True,
    )
    if document is None:
        return None

    deleted_at = datetime.now(timezone.utc).replace(tzinfo=None)
    document.status = Document_Status.DELETED
    document.deleted_at = document.deleted_at or deleted_at
    file_record = await db_session.get(File, document.file_id)
    if file_record is not None:
        file_record.deleted_at = file_record.deleted_at or deleted_at

    try:
        await db_session.commit()
    except Exception:
        await db_session.rollback()
        raise

    try:
        delete_file(document.file_id)
    except Exception:
        # A repeat DELETE can retry this idempotent MinIO cleanup.
        logger.exception(
            "Could not remove document object from storage",
            extra={"document_id": document.id, "file_id": document.file_id},
        )
    logger.info(
        "Document soft-deleted",
        extra={"document_id": document.id, "file_id": document.file_id, "user_id": current_user.id},
    )
    return document



async def get_jobs(id: int, current_user: CurrentUser, db_session: AsyncSession) -> list[Processing_Job] | None:
    owned_document = await db_session.execute(
        select(Document.id).where(
            Document.id == id,
            Document.user_id == current_user.id,
            Document.deleted_at.is_(None),
            Document.status != Document_Status.DELETED,
        )
    )
    if owned_document.scalar_one_or_none() is None:
        return None

    stmt = select(Processing_Job).join(Processing_Request).join(Document).where(
        Processing_Request.document_id == id,
        Document.user_id == current_user.id,
    )
    processing_job_record = await db_session.execute(stmt)
    result = processing_job_record.scalars().all()
    return list(result)


async def get_processing_request_by_id(id: int, current_user: CurrentUser, db_session: AsyncSession):
    stmt = Select(Processing_Request).join(Document).where(
        Processing_Request.id == id,
        Document.user_id == current_user.id,
        Document.deleted_at.is_(None),
        Document.status != Document_Status.DELETED,
    )
    record = await db_session.execute(stmt)
    result = record.scalar_one_or_none()
    return result
