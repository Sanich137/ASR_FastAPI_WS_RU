"""Pydantic-модели для аутентификации и авторизации."""

from pydantic import BaseModel, EmailStr, Field


class UserRegisterRequest(BaseModel):
    """Запрос на регистрацию по email/пароль."""

    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: str | None = None


class UserLoginRequest(BaseModel):
    """Запрос на вход по email/пароль."""

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Ответ с access-токеном (refresh — в httpOnly cookie)."""

    access_token: str
    token_type: str = "bearer"


class ChangePasswordRequest(BaseModel):
    """Запрос на смену пароля."""

    old_password: str
    new_password: str = Field(..., min_length=6)


class TelegramAuthRequest(BaseModel):
    """Запрос на аутентификацию через Telegram Web App."""

    init_data: str
