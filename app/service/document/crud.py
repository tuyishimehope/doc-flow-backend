from datetime import datetime, timezone
import logging

from sqlalchemy import Select, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.models.schema import Document, Extracted_Result, File, Processing_Request, Processing_Job
from app.service.auth.auth import CurrentUser
from app.service.document.schema import DocumentStats, Document_Status, ProcessingStats, Processing_Job_Status, Processing_status, Processing_Type, RecentFailure, UserStatsResponse
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
            Document.is_active(),
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
            Document.is_active(),
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
        statement = statement.where(Document.is_active())
    result = await db_session.execute(statement)
    response = result.scalar_one_or_none()
    return response


def _document_list_filters(user_id: int, q: str | None, status: Document_Status | None) -> list:
    filters = [Document.user_id == user_id, Document.is_active()]
    if q:
        filters.append(Document.name.icontains(q, autoescape=True))
    if status is not None:
        filters.append(Document.status == status)
    return filters


async def get_all_documents(
    skip: int,
    limit: int,
    db_session: AsyncSession,
    user_id: int,
    q: str | None = None,
    status: Document_Status | None = None,
) -> list[Document]:
    statement = (
        Select(Document)
        .where(*_document_list_filters(user_id, q, status))
        .order_by(Document.created_at.desc(), Document.id.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db_session.execute(statement)
    response = result.scalars().all()
    return list(response)


async def get_total_no_of_documents(
    db_session: AsyncSession,
    user_id: int,
    q: str | None = None,
    status: Document_Status | None = None,
) -> int:
    statement = Select(func.count(Document.id)).where(*_document_list_filters(user_id, q, status))
    result = await db_session.execute(statement)
    response = result.scalar_one()
    return response


async def update_document(
    id: int,
    changes: dict,
    current_user: CurrentUser,
    db_session: AsyncSession,
) -> Document | None:
    """Apply a rename and/or archive change, e.g. {"name": ..., "status": ...}."""
    document = await get_document_by_id(id=id, db_session=db_session, user_id=current_user.id)
    if document is None:
        return None

    for field, value in changes.items():
        setattr(document, field, value)
    await db_session.commit()
    # updated_at is set by the database, so reload it before the response reads it.
    await db_session.refresh(document)
    return document


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
    # updated_at is set by the database, so reload it before the response reads it.
    await db_session.refresh(document)

    try:
        await run_in_threadpool(delete_file, document.file_id)
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
    document = await get_document_by_id(id=id, db_session=db_session, user_id=current_user.id)
    if document is None:
        return None

    stmt = select(Processing_Job).join(Processing_Request).join(Document).where(
        Processing_Request.document_id == id,
        Document.user_id == current_user.id,
    )
    processing_job_record = await db_session.execute(stmt)
    result = processing_job_record.scalars().all()
    return list(result)


async def get_processing_requests_by_document(id: int, current_user: CurrentUser, db_session: AsyncSession) -> list[Processing_Request] | None:
    document = await get_document_by_id(id=id, db_session=db_session, user_id=current_user.id)
    if document is None:
        return None

    stmt = (
        select(Processing_Request)
        .where(Processing_Request.document_id == id)
        .order_by(Processing_Request.created_at.desc(), Processing_Request.id.desc())
    )
    result = await db_session.execute(stmt)
    return list(result.scalars().all())


async def get_processing_request_by_id(id: int, current_user: CurrentUser, db_session: AsyncSession):
    stmt = Select(Processing_Request).join(Document).where(
        Processing_Request.id == id,
        Document.user_id == current_user.id,
        Document.is_active(),
    )
    record = await db_session.execute(stmt)
    result = record.scalar_one_or_none()
    return result


async def cancel_processing_request(id: int, db_session: AsyncSession) -> bool:
    """Cancel a request that has not started. Returns False if it already started or finished."""
    # A single conditional UPDATE, so it cannot race with the worker picking the request up.
    result = await db_session.execute(
        update(Processing_Request)
        .where(
            Processing_Request.id == id,
            Processing_Request.status.in_([Processing_status.PENDING, Processing_status.QUEUED]),
        )
        .values(status=Processing_status.CANCELLED, updated_at=func.now())
    )
    await db_session.commit()
    return result.rowcount == 1


async def get_user_stats(user_id: int, db_session: AsyncSession, recent_failure_limit: int = 5) -> UserStatsResponse:
    document_counts = dict((await db_session.execute(
        select(Document.status, func.count())
        .where(Document.user_id == user_id, Document.is_active())
        .group_by(Document.status)
    )).all())

    request_rows = (await db_session.execute(
        select(Processing_Request.status, Processing_Request.processing_type, func.count())
        .join(Document)
        .where(Document.user_id == user_id, Document.is_active())
        .group_by(Processing_Request.status, Processing_Request.processing_type)
    )).all()
    # Start every key at 0 so clients get a stable shape.
    by_status = dict.fromkeys(Processing_status, 0)
    by_type = dict.fromkeys(Processing_Type, 0)
    for request_status, processing_type, count in request_rows:
        by_status[request_status] += count
        by_type[processing_type] += count

    failure_rows = (await db_session.execute(
        select(
            Processing_Request.id,
            Document.id,
            Document.name,
            Processing_Request.processing_type,
            Processing_Job.failure_reason,
            Processing_Job.completed_at,
        )
        .select_from(Processing_Job)
        .join(Processing_Request)
        .join(Document)
        .where(
            Document.user_id == user_id,
            Document.is_active(),
            Processing_Request.status == Processing_status.FAILED,
            Processing_Job.status == Processing_Job_Status.FAILED,
        )
        .order_by(Processing_Job.completed_at.desc(), Processing_Job.id.desc())
        .limit(recent_failure_limit)
    )).all()

    return UserStatsResponse(
        documents=DocumentStats(
            total=sum(document_counts.values()),
            active=document_counts.get(Document_Status.ACTIVE, 0),
            archived=document_counts.get(Document_Status.ARCHIVED, 0),
        ),
        processing_requests=ProcessingStats(
            total=sum(by_status.values()), by_status=by_status, by_type=by_type),
        recent_failures=[
            RecentFailure(
                processing_request_id=request_id,
                document_id=document_id,
                document_name=document_name,
                processing_type=processing_type,
                failure_reason=failure_reason,
                failed_at=failed_at,
            )
            for request_id, document_id, document_name, processing_type, failure_reason, failed_at in failure_rows
        ],
    )
