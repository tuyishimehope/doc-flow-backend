from typing import Annotated
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select
from sqlalchemy import delete as sql_delete

from app.service.auth.auth import create_user, delete_user_by_id, generate_reset_token, hash_password, hash_reset_token, update_user
from app.db.dependencies import get_db_session
from app.service.auth.schema import ChangePasswordRequest, CreateUserRequest, ForgotPasswordRequest, MessageResponse, ResetPasswordRequest, UserResponse, UserUpdate
from app.service.auth.auth import create_access_token, verify_password
from app.service.auth.schema import Token
from app.models import schema
from app.service.auth.auth import CurrentUser
from app.core.config import settings
from app.utils.email_utils import send_password_reset_email


router = APIRouter(prefix="/api/v1/users", tags=["users"])


# @router.post("/login")
# async def login(login_info: LoginRequest, db_session: Annotated[AsyncSession, Depends(get_db_session)]):
#     result = await authenticate(login_info, db_session)
#     if result is None:
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
#                             detail="Email or Password is not correct")
#     return result


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    # Look up user by email (case-insensitive)
    # Note: OAuth2PasswordRequestForm uses "username" field, but we treat it as email
    result = await db.execute(
        select(schema.User).where(
            func.lower(schema.User.email) == form_data.username.lower(),
            schema.User.deleted_at.is_(None),
        ),
    )
    user = result.scalars().first()

    # Verify user exists and password is correct
    # Don't reveal which one failed (security best practice)
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token with user id as subject
    access_token_expires = timedelta(
        minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=access_token_expires,
    )
    return Token(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
async def get_current_user(
    current_user: CurrentUser
):
    return current_user


@router.post("/signup", status_code=status.HTTP_201_CREATED, response_model=UserResponse)
async def signup(user: CreateUserRequest, db_session: AsyncSession = Depends(get_db_session)):
    result = await create_user(db_session=db_session, first_name=user.first_name, last_name=user.last_name, email=user.email, password=user.password)

    if result:
        return result

    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Email already exists")


@router.patch("/{id}", response_model=UserResponse)
async def update_user_info(id: int, user_info: UserUpdate, current_user: CurrentUser, db_session: Annotated[AsyncSession, Depends(get_db_session)]):
    result = await update_user(id, user_info, current_user, db_session)
    if result is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="You are not allowed to perform this action")
    return result


@router.delete("/{id}", response_model=UserResponse)
async def delete_user(id: int, current_user: CurrentUser, db_session: Annotated[AsyncSession, Depends(get_db_session)]):
    result = await delete_user_by_id(id, current_user, db_session)
    if result is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="You are not allowed to perform this action")
    return result


@router.post("/forgot-password", status_code=status.HTTP_202_ACCEPTED, response_model=MessageResponse)
async def forgot_password(
    request_data: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    result = await db.execute(
        select(schema.User).where(
            func.lower(schema.User.email) == request_data.email.lower(),
            schema.User.deleted_at.is_(None),
        ),
    )
    user = result.scalars().first()

    if user:
        await db.execute(
            sql_delete(schema.PasswordResetToken).where(
                schema.PasswordResetToken.user_id == user.id,
            ),
        )

        token = generate_reset_token()
        token_hash = hash_reset_token(token)
        expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=settings.reset_token_expire_minutes,
        )

        reset_token = schema.PasswordResetToken(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=expires_at,
        )
        db.add(reset_token)
        await db.commit()

        background_tasks.add_task(
            send_password_reset_email,
            to_email=user.email,
            username=user.first_name,
            token=token,
        )

    return {
        "message": "If an account exists with this email, you will receive password reset instructions.",
    }


@router.post("/reset-password", status_code=status.HTTP_200_OK, response_model=MessageResponse)
async def reset_password(
    request_data: ResetPasswordRequest,
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    token_hash = hash_reset_token(request_data.token)

    result = await db.execute(
        select(schema.PasswordResetToken).where(
            schema.PasswordResetToken.token_hash == token_hash,
        ),
    )
    reset_token = result.scalars().first()

    if not reset_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    if reset_token.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        await db.delete(reset_token)
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    result = await db.execute(
        select(schema.User).where(
            schema.User.id == reset_token.user_id,
            schema.User.deleted_at.is_(None),
        ),
    )
    user = result.scalars().first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    user.password_hash = hash_password(request_data.new_password)

    await db.execute(
        sql_delete(schema.PasswordResetToken).where(
            schema.PasswordResetToken.user_id == user.id,
        ),
    )

    await db.commit()
    return {
        "message": "Password reset successfully. You can now log in with your new password.",
    }


@router.patch("/me/password", status_code=status.HTTP_200_OK, response_model=MessageResponse)
async def change_password(
    password_data: ChangePasswordRequest,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db_session)],
):
    if not verify_password(password_data.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    current_user.password_hash = hash_password(password_data.new_password)

    await db.execute(
        sql_delete(schema.PasswordResetToken).where(
            schema.PasswordResetToken.user_id == current_user.id,
        ),
    )

    await db.commit()
    return {"message": "Password changed successfully"}
