from sqlalchemy import func, select

from app.models.schema import Document, File
from app.service.document.schema import Document_Status
from app.service.auth.auth import CurrentUser
from sqlalchemy.ext.asyncio import AsyncSession


async def get_file_id(
    id: int,
    current_user: CurrentUser,
    db_session: AsyncSession,
    include_deleted: bool = False,
):
    stmt = select(File).join(Document).where(
        File.id == id,
        Document.user_id == current_user.id,
    )
    if not include_deleted:
        stmt = stmt.where(
            File.deleted_at.is_(None),
            Document.deleted_at.is_(None),
            Document.status != Document_Status.DELETED,
        )

    file_record = await db_session.execute(stmt)

    result = file_record.scalar_one_or_none()

    return result


async def get_all_files_by_user(skip: int, limit: int, user_id: int, db_session:AsyncSession):
    statement = (
        select(File)
        .join(Document, Document.file_id == File.id)
        .where(
            Document.user_id == user_id,
            Document.deleted_at.is_(None),
            Document.status != Document_Status.DELETED,
            File.deleted_at.is_(None),
        )
        .order_by(File.created_at.desc(), File.id.desc())
        .offset(skip)
        .limit(limit)
    )

    result = await db_session.execute(statement)
    return result.scalars().all()


async def get_count(user_id: int, db_session: AsyncSession) -> int:
    statement = (
        select(func.count(Document.id))
        .join(File, Document.file_id == File.id)
        .where(
            Document.user_id == user_id,
            Document.deleted_at.is_(None),
            Document.status != Document_Status.DELETED,
            File.deleted_at.is_(None),
        )
    )

    result = await db_session.execute(statement)
    return result.scalar_one()
