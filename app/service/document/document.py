import csv
import io
import json
import logging

from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool


from app.service.auth.auth import CurrentUser
from app.tasks.document_task import start_processing
from app.utils.document import get_file_extension
from app.models.schema import Document, Processing_Request
from app.service.document.schema import ExportFormat, Processing_status, Processing_Type
from app.service.file.file import post_file, delete_file
from app.service.document.crud import get_processing_request_result, get_processing_request_status, save_document, save_file, save_processing_request
from app.service.auth.crud import get_user_by_id

logger = logging.getLogger(__name__)


async def process_document(id: int, file: UploadFile, processing_type: Processing_Type, instructions: str | None, db_session: AsyncSession):
    stored_object_id = None
    try:
        file_name = get_file_extension(file)

        file_object = await save_file(file=file, file_name=file_name, db_session=db_session)
        stored_object_id = str(file_object.id)

        await post_file(file=file, file_id=str(file_object.id))

        user = await get_user_by_id(db_session=db_session, id=id)

        if user is None:
            raise ValueError("Authenticated user was not found")

        document_object = await save_document(user_id=user.id, file=file, file_object=file_object, db_session=db_session)

        processing_request_object = await save_processing_request(document_object=document_object, processing_type=processing_type, instructions=instructions, db_session=db_session)

        if processing_request_object is None:
            raise RuntimeError("Could not create the processing request")

        processing_request_object.status = Processing_status.QUEUED
        await db_session.commit()

        try:
            start_processing.delay(processing_request_id=processing_request_object.id)
        except Exception:
            await db_session.delete(document_object)
            await db_session.delete(file_object)
            await db_session.commit()
            raise

        logger.info(
            "Document processing request queued",
            extra={
                "document_id": document_object.id,
                "processing_request_id": processing_request_object.id,
                "user_id": user.id,
            },
        )

        return {"document_id": document_object.id, "processing_request_id": processing_request_object.id, "status": processing_request_object.status}
    except Exception:
        await db_session.rollback()
        logger.exception(
            "Document upload or queueing failed",
            extra={"user_id": id, "stored_file_id": stored_object_id},
        )
        if stored_object_id is not None:
            try:
                await run_in_threadpool(delete_file, int(stored_object_id))
            except Exception:
                # Preserve the original failure; orphan cleanup can be retried operationally.
                pass
        raise


async def get_processing_status(
    processing_request_id: int,
    current_user: CurrentUser,
    db_session: AsyncSession
):
    result = await get_processing_request_status(processing_request_id=processing_request_id, current_user=current_user, db_session=db_session)

    if not result:
        return None

    return {
        "id": result.id,
        "status": result.status
    }


async def get_processing_result(
    processing_request_id: int,
    current_user: CurrentUser,
    db_session: AsyncSession
):
    result = await get_processing_request_result(processing_request_id=processing_request_id, current_user=current_user, db_session=db_session)

    if not result:
        return None

    return result.content_json


async def queue_processing_request(
    document: Document,
    processing_type: Processing_Type,
    instructions: str | None,
    db_session: AsyncSession,
) -> Processing_Request:
    """Run another processing pass over an already uploaded document."""
    latest_version = await db_session.scalar(
        select(func.max(Processing_Request.version)).where(
            Processing_Request.document_id == document.id)
    )
    processing_request = Processing_Request(
        document_id=document.id,
        processing_type=processing_type,
        instructions=instructions,
        status=Processing_status.QUEUED,
        version=(latest_version or 0) + 1,
    )
    db_session.add(processing_request)
    await db_session.commit()

    try:
        start_processing.delay(processing_request_id=processing_request.id)
    except Exception:
        await db_session.delete(processing_request)
        await db_session.commit()
        raise

    logger.info(
        "Document reprocessing request queued",
        extra={"document_id": document.id, "processing_request_id": processing_request.id},
    )
    return processing_request


def _flatten(value, prefix: str = ""):
    """Yield (field, value) rows: nested keys become dotted paths, list items get [i]."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list) and any(isinstance(item, (dict, list)) for item in value):
        for index, item in enumerate(value):
            yield from _flatten(item, f"{prefix}[{index}]")
    elif isinstance(value, list):
        yield prefix, "; ".join(str(item) for item in value)
    else:
        yield prefix or "summary", "" if value is None else value


def export_result(processing_request_id: int, result, export_format: ExportFormat) -> tuple[bytes, str, str]:
    """Return (content, media type, filename) for a completed result."""
    if export_format == ExportFormat.JSON:
        content = json.dumps(result, indent=2, ensure_ascii=False).encode()
        media_type = "application/json"
    else:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["field", "value"])
        writer.writerows(_flatten(result))
        # BOM so Excel opens UTF-8 text correctly.
        content = buffer.getvalue().encode("utf-8-sig")
        media_type = "text/csv"
    return content, media_type, f"processing-request-{processing_request_id}.{export_format.value}"
