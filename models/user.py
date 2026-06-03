"""Pydantic-модели для пользовательского кабинета."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class UserProfileResponse(BaseModel):
    """Ответ с профилем пользователя."""

    id: str
    email: str
    full_name: Optional[str] = None
    phone: Optional[str] = None
    role: str
    is_active: bool
    telegram_linked: bool = False


class UserProfileUpdateRequest(BaseModel):
    """Запрос на обновление профиля."""

    full_name: Optional[str] = None
    phone: Optional[str] = None


class UserQuotaResponse(BaseModel):
    """Ответ с квотой пользователя."""

    plan_name: Optional[str] = None
    max_requests_per_minute: int = 0
    requests_used_this_minute: int = 0
    max_audio_duration_sec: Optional[int] = None


class UserSubscriptionResponse(BaseModel):
    """Ответ с данными подписки."""

    status: str
    plan_name: Optional[str] = None
    started_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    auto_renew: bool = False


class ASRSessionItem(BaseModel):
    """Элемент списка ASR-сессий."""

    id: str
    session_type: str
    status: str
    audio_duration_sec: Optional[float] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class UserStatsResponse(BaseModel):
    """Агрегированная статистика пользователя."""

    total_sessions: int = 0
    total_audio_hours: float = 0.0
    sessions_this_month: int = 0


class ApiKeyResponse(BaseModel):
    """Ответ с данными API-ключа (без plain key)."""

    id: str
    name: str
    is_active: bool
    created_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None


class ApiKeyCreateRequest(BaseModel):
    """Запрос на создание API-ключа."""

    name: str


class ApiKeyCreateResponse(ApiKeyResponse):
    """Ответ при создании ключа (с plain key, один раз)."""

    plain_key: str


class TelegramLinkStatus(BaseModel):
    """Статус привязки Telegram."""

    linked: bool
    telegram_username: Optional[str] = None


class TelegramUnlinkRequest(BaseModel):
    """Запрос на отвязку Telegram."""

    password: Optional[str] = None
