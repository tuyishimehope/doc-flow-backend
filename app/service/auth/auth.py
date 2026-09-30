import hashlib
import logging
import secrets
from datetime import timezone, datetime, timedelta
from typing import Annotated
from alembic.util import status
from fastapi import Depends, HTTPException
import jwt
from pwdlib import PasswordHash
from fastapi.security import OAuth2PasswordBearer
from argon2 import PasswordHasher
from sqlalchemy import Select
from sqlalchemy.ext.asyncio import AsyncSession


from app.db.dependencies import get_db_session
from app.service.auth.schema import LoginRequest, UserUpdate
from app.core.config import settings
from app.models.schema import User
from app.service.auth.crud import get_count_users, get_user_by_email, get_users


password_hash = PasswordHash.recommended()
logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/users/token")


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_hash.verify(password=plain_password, hash=hashed_password)


def generate_reset_token() -> str:
    return secrets.token_urlsafe(32)


def hash_reset_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(
            timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key.get_secret_value(), algorithm=settings.algorithm)
    return encoded_jwt


def verify_access_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, settings.secret_key.get_secret_value(), algorithms=[
                             settings.algorithm], options={"require": ["exp", "sub"]})
    except jwt.InvalidTokenError:
        return None
    else:
        return payload.get("sub")


async def authenticate(login_info: LoginRequest, db_session: AsyncSession):
    try:
        user_obj = await get_user_by_email(email=login_info.email, db_session=db_session)
        if user_obj is None:
            raise

        ph = PasswordHasher()

        result = ph.verify(user_obj.password_hash,
                           password=login_info.password)

        if result == False:
            raise

        if login_info.email == user_obj.email:
            return user_obj.email
    except Exception as e:
        raise e


async def create_user(first_name: str, last_name: str, email: str, password: str, db_session: AsyncSession,):
    try:
        user = await get_user_by_email(email, db_session)
        if user is not None:
            return False

        password_hash = hash_password(password)
        user = User(first_name=first_name, last_name=last_name,
                    email=email.lower(), password_hash=password_hash)
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        return user
    except Exception:
        logger.exception("User creation failed")
        await db_session.rollback()
        raise


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> User:
    user_id = verify_access_token(token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id_int = int(user_id)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    result = await db.execute(
        Select(User).where(User.id == user_id_int),
    )
    user = result.scalars().first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_all_users(limit: int, skip: int, db_session: AsyncSession) -> list[User]:
    users = await get_users(limit, skip, db_session)
    return users


async def update_user(
    id: int,
    user_info: UserUpdate,
    current_user: CurrentUser,
    db_session: AsyncSession,
):
    if id != current_user.id:
        return None

    statement = Select(User).where(User.id == current_user.id)

    result = await db_session.execute(statement)
    user = result.scalar_one_or_none()

    if user is None:
        return None

    update_data = user_info.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(user, field, value)

    await db_session.commit()
    await db_session.refresh(user)

    return user


async def delete_user_by_id(id: int,
                            current_user: CurrentUser,
                            db_session: AsyncSession,
                            ):
    if id != current_user.id:
        return None

    statement = Select(User).where(User.id == current_user.id)

    result = await db_session.execute(statement)
    user = result.scalar_one_or_none()

    if user is None:
        return None

    user.deleted_at = datetime.now(timezone.utc)
    await db_session.commit()
