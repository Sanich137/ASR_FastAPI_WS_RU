"""Pydantic-модели для админ-панели."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class PaginationParams(BaseModel):
    """Параметры пагинации для списков."""

    page: int = Field(1, ge=1)
    per_page: int = Field(20, ge=1, le=100)


class AdminUserListItem(BaseModel):
    """Элемент списка пользователей в админке."""

    id: str
    email: str
    full_name: Optional[str] = None
    role: str
    is_active: bool
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None
    telegram_linked: bool = False


class AdminUserDetailResponse(AdminUserListItem):
    """Детальная информация о пользователе."""

    phone: Optional[str] = None
    telegram_id: Optional[int] = None
    telegram_username: Optional[str] = None


class AdminUserUpdateRequest(BaseModel):
    """Запрос на обновление пользователя админом."""

    full_name: Optional[str] = None
    phone: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class AdminPlanCreateUpdateRequest(BaseModel):
    """Запрос на создание/обновление тарифа."""

    code: str
    name: str
    description: Optional[str] = None
    max_requests_per_minute: int = 60
    max_audio_duration_sec: Optional[int] = None
    price_per_month: Optional[float] = None
    is_active: bool = True


class AdminPlanResponse(AdminPlanCreateUpdateRequest):
    """Ответ с тарифом."""

    id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AdminSubscriptionResponse(BaseModel):
    """Ответ с подпиской."""

    id: str
    user_id: str
    plan_id: str
    plan_name: Optional[str] = None
    status: str
    started_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    auto_renew: bool = False


class AdminTransactionResponse(BaseModel):
    """Ответ с транзакцией."""

    id: str
    user_id: str
    subscription_id: Optional[str] = None
    amount: Optional[float] = None
    currency: str
    status: str
    payment_provider: str
    external_payment_id: Optional[str] = None
    created_at: Optional[datetime] = None


class AdminSystemLogResponse(BaseModel):
    """Ответ с системным логом."""

    id: str
    level: str
    component: str
    message: str
    meta: Optional[dict] = None
    created_at: Optional[datetime] = None


class AdminAuditLogResponse(BaseModel):
    """Ответ с аудит-логом."""

    id: str
    admin_id: str
    action: str
    target_type: Optional[str] = None
    target_id: Optional[str] = None
    details: Optional[dict] = None
    created_at: Optional[datetime] = None


class AdminApiKeyResponse(BaseModel):
    """Ответ с API-ключом (админка)."""

    id: str
    user_id: str
    user_email: Optional[str] = None
    name: str
    is_active: bool
    created_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None


class AdminMetricsResponse(BaseModel):
    """Ответ с текущими метриками системы."""

    active_connections: int = 0
    active_tasks: int = 0
    cpu_percent: Optional[float] = None
    gpu_utilization: Optional[float] = None
    queue_depth: int = 0
    uptime_seconds: float = 0.0


class AdminMaintenanceToggle(BaseModel):
    """Запрос на переключение режима обслуживания."""

    enabled: bool


class AdminTelegramConfig(BaseModel):
    """Конфигурация Telegram-бота."""

    webapp_url: Optional[str] = None
    is_active: bool = True


class AdminTelegramStats(BaseModel):
    """Статистика Telegram Web App."""

    total_telegram_users: int = 0
    total_webapp_sessions: int = 0


class AdminBroadcastRequest(BaseModel):
    """Запрос на рассылку сообщения."""

    message: str = Field(..., min_length=1, max_length=4096)
