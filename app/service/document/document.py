from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
import logging


from app.service.auth.auth import CurrentUser
from app.tasks.document_task import start_processing
from app.utils.document import get_file_extension
from .schema import Processing_Type
from app.service.document.schema import Processing_status, Processing_Type
from app.service.file.file import post_file, get_file, delete_file
from app.service.document.crud import delete_document_by_id, get_all_documents, get_document_by_id, get_jobs, get_processing_request_result, get_processing_request_status, save_document, save_file, save_processing_request, get_total_no_of_documents, get_processing_request_by_id
from app.service.auth.crud import get_user_by_id

logger = logging.getLogger(__name__)


async def process_document(id: int, file: UploadFile, processing_type: Processing_Type, instructions: str, db_session: AsyncSession):
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
                delete_file(int(stored_object_id))
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


async def get_document(id: int, db_session: AsyncSession, user_id: int):
    result = await get_document_by_id(id=id, db_session=db_session, user_id=user_id)
    return result


async def get_documents(skip: int, limit: int, db_session: AsyncSession, user_id: int):
    result = await get_all_documents(skip=skip, limit=limit, db_session=db_session, user_id=user_id)
    return result


async def delete_document(id: int, current_user: CurrentUser, db_session: AsyncSession):
    return await delete_document_by_id(id=id, current_user=current_user, db_session=db_session)




async def get_status_jobs(id: int, current_user: CurrentUser, db_session: AsyncSession):
    response = await get_jobs(id=id, current_user=current_user, db_session=db_session)
    return response


async def get_processing_request(id: int, current_user: CurrentUser, db_session: AsyncSession):
    response = await get_processing_request_by_id(id, current_user, db_session)
    return response
