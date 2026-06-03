"""Бизнес-логика админ-панели."""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.security import create_access_token  # type: ignore[import-untyped]
from db.enums import SubscriptionStatus, UserRole
from db.models import (
    AdminAuditLog,
    ApiKey,
    ASRSession,
    Plan,
    Subscription,
    SystemLog,
    TelegramBotConfig,
    Transaction,
    User,
)


# In-memory флаг режима обслуживания
_maintenance_mode: bool = False


def is_maintenance_mode() -> bool:
    """Возвращает True, если включён режим обслуживания."""
    return _maintenance_mode


def set_maintenance_mode(enabled: bool) -> None:
    """Включает/выключает режим обслуживания."""
    global _maintenance_mode
    _maintenance_mode = enabled


async def get_users_list(
    db: AsyncSession,
    page: int = 1,
    per_page: int = 20,
    search: Optional[str] = None,
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
) -> tuple[list[User], int]:
    """Возвращает список пользователей с пагинацией и общее количество."""
    query = select(User)
    count_query = select(func.count(User.id))

    if search:
        pattern = f"%{search}%"
        query = query.where(
            (User.email.ilike(pattern)) | (User.full_name.ilike(pattern))
        )
        count_query = count_query.where(
            (User.email.ilike(pattern)) | (User.full_name.ilike(pattern))
        )

    if role:
        query = query.where(User.role == role)
        count_query = count_query.where(User.role == role)

    if is_active is not None:
        query = query.where(User.is_active.is_(is_active))
        count_query = count_query.where(User.is_active.is_(is_active))

    query = query.order_by(User.created_at.desc())
    query = query.offset((page - 1) * per_page).limit(per_page)

    result = await db.execute(query)
    users = result.scalars().all()

    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    return list(users), total


async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
    """Возвращает пользователя по ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def update_user(
    db: AsyncSession,
    user: User,
    data: dict[str, Any],
) -> User:
    """Обновляет поля пользователя."""
    for field in ("full_name", "phone", "role", "is_active"):
        if field in data and data[field] is not None:
            setattr(user, field, data[field])
    await db.commit()
    await db.refresh(user)
    return user


async def get_user_sessions(
    db: AsyncSession,
    user_id: str,
    limit: int = 50,
) -> list[ASRSession]:
    """Возвращает ASR-сессии пользователя."""
    result = await db.execute(
        select(ASRSession)
        .where(ASRSession.user_id == user_id)
        .order_by(ASRSession.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def impersonate_user(user: User) -> str:
    """Генерирует access token от имени пользователя (superadmin only)."""
    return create_access_token({"sub": user.id, "role": user.role.value})


async def get_plans(db: AsyncSession) -> list[Plan]:
    """Возвращает все тарифные планы."""
    result = await db.execute(select(Plan).order_by(Plan.created_at.desc()))
    return list(result.scalars().all())


async def create_plan(db: AsyncSession, data: dict[str, Any]) -> Plan:
    """Создаёт новый тарифный план."""
    plan = Plan(**data)
    db.add(plan)
    await db.commit()
    await db.refresh(plan)
    return plan


async def update_plan(
    db: AsyncSession,
    plan: Plan,
    data: dict[str, Any],
) -> Plan:
    """Обновляет тарифный план."""
    for field in (
        "code",
        "name",
        "description",
        "max_requests_per_minute",
        "max_audio_duration_sec",
        "price_per_month",
        "is_active",
    ):
        if field in data and data[field] is not None:
            setattr(plan, field, data[field])
    await db.commit()
    await db.refresh(plan)
    return plan


async def delete_plan(db: AsyncSession, plan: Plan) -> None:
    """Деактивирует тарифный план."""
    plan.is_active = False
    await db.commit()


async def get_subscriptions(
    db: AsyncSession,
    page: int = 1,
    per_page: int = 20,
) -> tuple[list[Subscription], int]:
    """Возвращает подписки с пагинацией."""
    result = await db.execute(
        select(Subscription)
        .order_by(Subscription.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
    )
    total_result = await db.execute(select(func.count(Subscription.id)))
    return list(result.scalars().all()), total_result.scalar() or 0


async def extend_subscription(
    db: AsyncSession,
    subscription: Subscription,
    days: int = 30,
) -> Subscription:
    """Ручное продление подписки."""
    now = datetime.now(timezone.utc)
    if subscription.expires_at:
        subscription.expires_at = subscription.expires_at + timedelta(days=days)
    else:
        subscription.expires_at = now + timedelta(days=days)
    subscription.status = SubscriptionStatus.active
    await db.commit()
    await db.refresh(subscription)
    return subscription


async def cancel_subscription_admin(
    db: AsyncSession,
    subscription: Subscription,
) -> None:
    """Ручная отмена подписки админом."""
    subscription.status = SubscriptionStatus.cancelled
    subscription.auto_renew = False
    await db.commit()


async def get_transactions(
    db: AsyncSession,
    page: int = 1,
    per_page: int = 20,
) -> tuple[list[Transaction], int]:
    """Возвращает транзакции с пагинацией."""
    result = await db.execute(
        select(Transaction)
        .order_by(Transaction.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
    )
    total_result = await db.execute(select(func.count(Transaction.id)))
    return list(result.scalars().all()), total_result.scalar() or 0


async def get_system_logs(
    db: AsyncSession,
    page: int = 1,
    per_page: int = 50,
    level: Optional[str] = None,
    component: Optional[str] = None,
) -> tuple[list[SystemLog], int]:
    """Возвращает системные логи."""
    query = select(SystemLog).order_by(SystemLog.created_at.desc())
    count_query = select(func.count(SystemLog.id))

    if level:
        query = query.where(SystemLog.level == level)
        count_query = count_query.where(SystemLog.level == level)
    if component:
        query = query.where(SystemLog.component == component)
        count_query = count_query.where(SystemLog.component == component)

    query = query.offset((page - 1) * per_page).limit(per_page)
    result = await db.execute(query)
    total_result = await db.execute(count_query)
    return list(result.scalars().all()), total_result.scalar() or 0


async def get_audit_logs(
    db: AsyncSession,
    page: int = 1,
    per_page: int = 50,
) -> tuple[list[AdminAuditLog], int]:
    """Возвращает аудит-логи админов."""
    result = await db.execute(
        select(AdminAuditLog)
        .order_by(AdminAuditLog.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
    )
    total_result = await db.execute(select(func.count(AdminAuditLog.id)))
    return list(result.scalars().all()), total_result.scalar() or 0


async def get_api_keys(
    db: AsyncSession,
    page: int = 1,
    per_page: int = 50,
    user_id: Optional[str] = None,
) -> tuple[list[ApiKey], int]:
    """Возвращает API-ключи (все или по пользователю)."""
    query = select(ApiKey).order_by(ApiKey.created_at.desc())
    count_query = select(func.count(ApiKey.id))

    if user_id:
        query = query.where(ApiKey.user_id == user_id)
        count_query = count_query.where(ApiKey.user_id == user_id)

    query = query.offset((page - 1) * per_page).limit(per_page)
    result = await db.execute(query)
    total_result = await db.execute(count_query)
    return list(result.scalars().all()), total_result.scalar() or 0


async def revoke_api_key_admin(db: AsyncSession, key: ApiKey) -> None:
    """Отзывает API-ключ админом."""
    key.is_active = False
    await db.commit()


async def get_queue_status() -> dict[str, Any]:
    """Заглушка: возвращает статус очереди ASR."""
    return {"active_tasks": 0, "pending_tasks": 0}


async def cancel_task(task_id: str) -> bool:
    """Заглушка: отмена задачи в очереди."""
    return False


async def disconnect_user_session(user_id: str, session_id: str) -> bool:
    """Заглушка: принудительное отключение WS-сессии."""
    return False


async def get_telegram_config(db: AsyncSession) -> Optional[TelegramBotConfig]:
    """Возвращает конфигурацию Telegram-бота."""
    result = await db.execute(
        select(TelegramBotConfig).where(TelegramBotConfig.is_active.is_(True))
    )
    return result.scalar_one_or_none()


async def get_telegram_stats(db: AsyncSession) -> dict[str, int]:
    """Возвращает статистику Telegram-пользователей."""
    total_result = await db.execute(
        select(func.count(User.id)).where(User.telegram_id.isnot(None))
    )
    return {
        "total_telegram_users": total_result.scalar() or 0,
        "total_webapp_sessions": 0,
    }


async def set_telegram_webhook(url: str, bot_token: str) -> dict[str, str]:
    """Заглушка: установка webhook Telegram-бота."""
    return {"detail": "Webhook установка — заглушка", "url": url}


async def broadcast_message(message: str) -> dict[str, str]:
    """Заглушка: рассылка сообщения всем пользователям бота."""
    return {"detail": "Рассылка — заглушка", "message": message}
