"""Перечисления (Enum) для SQLAlchemy-моделей."""

import enum


class UserRole(str, enum.Enum):
    """Роли пользователей."""

    user = "user"
    admin = "admin"
    superadmin = "superadmin"


class SubscriptionStatus(str, enum.Enum):
    """Статусы подписки."""

    active = "active"
    expired = "expired"
    cancelled = "cancelled"


class TransactionStatus(str, enum.Enum):
    """Статусы платежной транзакции."""

    pending = "pending"
    succeeded = "succeeded"
    cancelled = "cancelled"


class ASRSessionStatus(str, enum.Enum):
    """Статусы сессии распознавания."""

    processing = "processing"
    completed = "completed"
    failed = "failed"


class ASRSessionType(str, enum.Enum):
    """Типы сессии распознавания."""

    url = "url"
    file = "file"
    websocket = "websocket"


class SystemLogLevel(str, enum.Enum):
    """Уровни системного лога."""

    info = "info"
    warning = "warning"
    error = "error"
