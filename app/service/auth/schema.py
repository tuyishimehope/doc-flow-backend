from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, ConfigDict, Field
from fastapi import Body


class LoginRequest(BaseModel):
    email: EmailStr = Body()
    password: str = Body()


class UserBase(BaseModel):
    first_name: str = Body()
    last_name: str = Body()
    email: EmailStr = Body()

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[EmailStr] = None

class CreateUserRequest(UserBase):
    password: str = Body()


class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime
    
class PaginatedUserResponse(BaseModel):
    users: list[UserResponse]
    total: int
    skip: int
    limit: int
    has_more: bool
    


class Token(BaseModel):
    access_token: str
    token_type: str

class ForgotPasswordRequest(BaseModel):
    email: EmailStr = Field(max_length=120)


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=8)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)


class MessageResponse(BaseModel):
    message: str
