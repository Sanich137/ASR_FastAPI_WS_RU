"""SQLAlchemy 2.0 ORM-модели для ASR-сервиса."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import BigInteger, JSON, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base
from db.enums import (
    ASRSessionStatus,
    ASRSessionType,
    SubscriptionStatus,
    SystemLogLevel,
    TransactionStatus,
    UserRole,
)


class User(Base):
    """Пользователь системы."""

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(Text)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    role: Mapped[UserRole] = mapped_column(String(20), default=UserRole.user)
    is_active: Mapped[bool] = mapped_column(default=True)
    email_verified: Mapped[bool] = mapped_column(default=False)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    # Telegram Web App fields
    telegram_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, unique=True, nullable=True, index=True
    )
    telegram_username: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    telegram_first_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    telegram_last_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    telegram_photo_url: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )
    telegram_auth_date: Mapped[Optional[datetime]] = mapped_column(
        nullable=True
    )

    # relationships
    subscriptions: Mapped[list["Subscription"]] = relationship(
        back_populates="user", lazy="selectin"
    )
    api_keys: Mapped[list["ApiKey"]] = relationship(
        back_populates="user", lazy="selectin"
    )
    asr_sessions: Mapped[list["ASRSession"]] = relationship(
        back_populates="user", lazy="selectin"
    )
    admin_audit_logs: Mapped[list["AdminAuditLog"]] = relationship(
        back_populates="admin", lazy="selectin"
    )


class Plan(Base):
    """Тарифный план (справочник)."""

    __tablename__ = "plans"

    code: Mapped[str] = mapped_column(String(50), unique=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    max_requests_per_minute: Mapped[int] = mapped_column(Integer, default=60)
    max_audio_duration_sec: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    price_per_month: Mapped[Optional[int]] = mapped_column(
        Numeric(10, 2), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(default=True)

    subscriptions: Mapped[list["Subscription"]] = relationship(
        back_populates="plan", lazy="selectin"
    )


class Subscription(Base):
    """Подписка пользователя."""

    __tablename__ = "subscriptions"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), index=True
    )
    plan_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("plans.id"), index=True
    )
    status: Mapped[SubscriptionStatus] = mapped_column(
        String(20), default=SubscriptionStatus.active
    )
    started_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc)
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    auto_renew: Mapped[bool] = mapped_column(default=False)
    yookassa_payment_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )

    user: Mapped["User"] = relationship(back_populates="subscriptions")
    plan: Mapped["Plan"] = relationship(back_populates="subscriptions")
    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="subscription", lazy="selectin"
    )


class Transaction(Base):
    """Платёжная транзакция."""

    __tablename__ = "transactions"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), index=True
    )
    subscription_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("subscriptions.id"), nullable=True, index=True
    )
    amount: Mapped[Optional[int]] = mapped_column(Numeric(10, 2), nullable=True)
    currency: Mapped[str] = mapped_column(String(3), default="RUB")
    status: Mapped[TransactionStatus] = mapped_column(
        String(20), default=TransactionStatus.pending
    )
    payment_provider: Mapped[str] = mapped_column(String(50), default="yookassa")
    external_payment_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, index=True
    )
    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    subscription: Mapped[Optional["Subscription"]] = relationship(
        back_populates="transactions"
    )


class ASRSession(Base):
    """Сессия распознавания речи."""

    __tablename__ = "asr_sessions"

    user_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=True, index=True
    )
    session_type: Mapped[ASRSessionType] = mapped_column(String(20))
    status: Mapped[ASRSessionStatus] = mapped_column(
        String(20), default=ASRSessionStatus.processing
    )
    audio_duration_sec: Mapped[Optional[float]] = mapped_column(nullable=True)
    processing_duration_sec: Mapped[Optional[float]] = mapped_column(nullable=True)
    cost: Mapped[Optional[int]] = mapped_column(Numeric(10, 2), nullable=True)
    result_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    request_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    user: Mapped[Optional["User"]] = relationship(back_populates="asr_sessions")


class ApiKey(Base):
    """API-ключ для программного доступа."""

    __tablename__ = "api_keys"

    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    permissions: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    rate_limit_override: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    user: Mapped["User"] = relationship(back_populates="api_keys")


class SystemLog(Base):
    """Системный лог."""

    __tablename__ = "system_logs"

    level: Mapped[SystemLogLevel] = mapped_column(String(20), index=True)
    component: Mapped[str] = mapped_column(String(100), index=True)
    message: Mapped[str] = mapped_column(Text)
    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


class AdminAuditLog(Base):
    """Аудит действий администраторов."""

    __tablename__ = "admin_audit_logs"

    admin_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), index=True
    )
    action: Mapped[str] = mapped_column(String(100))
    target_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    target_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    admin: Mapped["User"] = relationship(back_populates="admin_audit_logs")


class TelegramBotConfig(Base):
    """Конфигурация Telegram-бота (опционально, для админ-панели)."""

    __tablename__ = "telegram_bot_configs"

    bot_token_hash: Mapped[str] = mapped_column(Text)
    webapp_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
