from enum import Enum


class Role(str, Enum):
    """Роли пользователей для RBAC."""
    guest = "guest"
    user = "user"
    admin = "admin"
    superadmin = "superadmin"


class SubscriptionType(str, Enum):
    """Типы подписок."""
    free = "free"
    pro = "pro"
    enterprise = "enterprise"


class SubscriptionStatus(str, Enum):
    """Статусы подписки."""
    active = "active"
    expired = "expired"
    cancelled = "cancelled"


class TransactionStatus(str, Enum):
    """Статусы транзакции."""
    pending = "pending"
    completed = "completed"
    failed = "failed"


class PaymentMethod(str, Enum):
    """Способы оплаты."""
    card = "card"
    crypto = "crypto"
    bank_transfer = "bank_transfer"
