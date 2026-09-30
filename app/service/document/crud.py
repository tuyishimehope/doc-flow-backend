from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema import Document, Extracted_Result, File, Processing_Request, Processing_Job, User
from app.service.auth.auth import CurrentUser
from app.service.document.schema import Processing_status
from app.service.file.file import delete_file_by_id


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
            Document.user_id == current_user.id
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
            Document.user_id == current_user.id
        )
    )
    result = await db_session.execute(request)
    response = result.scalar_one_or_none()
    return response


async def get_document_by_id(id: int, db_session: AsyncSession, user_id: int) -> Document | None:
    statement = Select(Document).where(
        Document.id == id, Document.user_id == user_id)
    result = await db_session.execute(statement)
    response = result.scalar_one_or_none()
    return response


async def get_all_documents(skip: int, limit: int, db_session: AsyncSession, user_id: int) -> list[Document] :
    statement = Select(Document).where(Document.user_id ==
                                       user_id).offset(skip).limit(limit)
    result = await db_session.execute(statement)
    response = result.scalars().all()
    return list(response)


async def get_total_no_of_documents(db_session: AsyncSession, user_id: int) -> int:
    statement = Select(func.count(Document.id)).where(
        Document.user_id == user_id)
    result = await db_session.execute(statement)
    response = result.scalar_one()
    return response


async def delete_document_by_id(id: int, current_user: CurrentUser, db_session: AsyncSession):
    try:
        document = await get_document_by_id(id=id, db_session=db_session, user_id=current_user.id)
        if document:
            await db_session.delete(document)
            await delete_file_by_id(id=document.file_id, current_user=current_user, db_session=db_session)
            await db_session.commit()
            return document
    except:
        await db_session.rollback()



async def get_jobs(id: int, current_user: CurrentUser, db_session: AsyncSession) -> list[Processing_Job] | None:
    owned_document = await db_session.execute(
        select(Document.id).where(
            Document.id == id,
            Document.user_id == current_user.id,
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
    stmt = Select(Processing_Request).join(Document).where(Processing_Request.id == id, Document.user_id == current_user.id)
    record = await db_session.execute(stmt)
    result = record.scalar_one_or_none()
    return result
