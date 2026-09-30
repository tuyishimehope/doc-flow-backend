from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema import User


async def get_user_by_email(email: str, db_session: AsyncSession) -> User | None:
    statement = select(User).where(func.lower(User.email) == email.lower())
    results = await db_session.execute(statement)
    user_obj = results.scalar_one_or_none()
    return user_obj


async def get_user_by_id(id: int, db_session: AsyncSession) -> User | None:
    statement = select(User).where(User.id == id)
    results = await db_session.execute(statement)
    user_obj = results.scalar_one_or_none()
    return user_obj
