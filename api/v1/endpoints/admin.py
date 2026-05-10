"""FastAPI-роутер админ-панели."""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.deps import get_current_user, require_admin, require_superadmin
from db.models import ApiKey, Plan, Subscription, SystemLog, Transaction, User
from db.session import get_db_session
from models.admin import (
    AdminApiKeyResponse,
    AdminAuditLogResponse,
    AdminBroadcastRequest,
    AdminMaintenanceToggle,
    AdminMetricsResponse,
    AdminPlanCreateUpdateRequest,
    AdminPlanResponse,
    AdminSubscriptionResponse,
    AdminSystemLogResponse,
    AdminTelegramConfig,
    AdminTelegramStats,
    AdminTransactionResponse,
    AdminUserDetailResponse,
    AdminUserListItem,
    AdminUserUpdateRequest,
    PaginationParams,
)
from services import admin_service

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/metrics", response_model=AdminMetricsResponse)
async def admin_metrics(current_user: User = Depends(require_admin)):
    """Текущие метрики системы (заглушка)."""
    return AdminMetricsResponse()


@router.get("/metrics/history")
async def admin_metrics_history(
    range: str = "24h",
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """История метрик за период с группировкой по 5-минутным слотам."""
    # Парсим параметр range (поддерживаем 1h, 24h, 7d и т.д.)
    try:
        if range.endswith("h"):
            hours = int(range[:-1])
            since = datetime.utcnow() - timedelta(hours=hours)
        elif range.endswith("d"):
            days = int(range[:-1])
            since = datetime.utcnow() - timedelta(days=days)
        else:
            since = datetime.utcnow() - timedelta(hours=24)
    except ValueError:
        since = datetime.utcnow() - timedelta(hours=24)

    result = await db.execute(
        select(SystemLog)
        .where(
            SystemLog.component == "SystemMetricsCollector",
            SystemLog.created_at >= since,
        )
        .order_by(SystemLog.created_at)
    )
    logs = result.scalars().all()

    # Группировка по 5-минутным слотам
    slots = defaultdict(
        lambda: {
            "cpu_values": [],
            "gpu_values": [],
            "conn_values": [],
            "queue_values": [],
        }
    )

    for log in logs:
        ts = log.created_at
        slot_ts = ts.replace(
            minute=(ts.minute // 5) * 5, second=0, microsecond=0
        )
        key = slot_ts.isoformat()
        meta = log.meta or {}
        slots[key]["cpu_values"].append(meta.get("cpu_percent"))
        slots[key]["gpu_values"].append(meta.get("gpu_utilization_percent"))
        slots[key]["conn_values"].append(meta.get("active_connections"))
        slots[key]["queue_values"].append(meta.get("queue_depth"))

    def _avg(values):
        clean = [v for v in values if v is not None]
        return round(sum(clean) / len(clean), 2) if clean else None

    response = []
    for key in sorted(slots.keys()):
        data = slots[key]
        response.append(
            {
                "timestamp": key,
                "cpu": _avg(data["cpu_values"]),
                "gpu": _avg(data["gpu_values"]),
                "active_connections": _avg(data["conn_values"]),
                "queue_depth": _avg(data["queue_values"]),
            }
        )

    return response


@router.get("/users", response_model=list[AdminUserListItem])
async def admin_users_list(
    pagination: PaginationParams = Depends(),
    search: Optional[str] = None,
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Список пользователей с пагинацией и фильтрами."""
    users, total = await admin_service.get_users_list(
        db,
        page=pagination.page,
        per_page=pagination.per_page,
        search=search,
        role=role,
        is_active=is_active,
    )
    return [
        AdminUserListItem(
            id=u.id,
            email=u.email,
            full_name=u.full_name,
            role=u.role,
            is_active=u.is_active,
            created_at=u.created_at,
            last_login_at=u.last_login_at,
            telegram_linked=u.telegram_id is not None,
        )
        for u in users
    ]


@router.get("/users/{user_id}", response_model=AdminUserDetailResponse)
async def admin_user_detail(
    user_id: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Детали пользователя."""
    user = await admin_service.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Пользователь не найден"
        )
    return AdminUserDetailResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
        telegram_linked=user.telegram_id is not None,
        phone=user.phone,
        telegram_id=user.telegram_id,
        telegram_username=user.telegram_username,
    )


@router.put("/users/{user_id}", response_model=AdminUserDetailResponse)
async def admin_user_update(
    user_id: str,
    payload: AdminUserUpdateRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Редактирование пользователя админом."""
    user = await admin_service.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Пользователь не найден"
        )
    user = await admin_service.update_user(
        db, user, payload.model_dump(exclude_unset=True)
    )
    return AdminUserDetailResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
        telegram_linked=user.telegram_id is not None,
        phone=user.phone,
        telegram_id=user.telegram_id,
        telegram_username=user.telegram_username,
    )


@router.delete("/users/{user_id}")
async def admin_user_delete(
    user_id: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Soft-delete / блокировка пользователя."""
    user = await admin_service.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Пользователь не найден"
        )
    user.is_active = False
    await db.commit()
    return {"detail": "Пользователь деактивирован"}


@router.get("/users/{user_id}/sessions")
async def admin_user_sessions(
    user_id: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
    limit: int = 50,
):
    """История ASR-сессий пользователя."""
    sessions = await admin_service.get_user_sessions(db, user_id, limit=limit)
    return [
        {
            "id": s.id,
            "session_type": s.session_type.value,
            "status": s.status.value,
            "audio_duration_sec": s.audio_duration_sec,
            "created_at": s.created_at,
            "completed_at": s.completed_at,
        }
        for s in sessions
    ]


@router.post("/users/{user_id}/impersonate")
async def admin_user_impersonate(
    user_id: str,
    current_user: User = Depends(require_superadmin),
    db: AsyncSession = Depends(get_db_session),
):
    """Получить access token от имени пользователя (superadmin only)."""
    user = await admin_service.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Пользователь не найден"
        )
    token = await admin_service.impersonate_user(user)
    return {"access_token": token, "token_type": "bearer"}


@router.get("/tariffs", response_model=list[AdminPlanResponse])
async def admin_tariffs_list(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Список тарифных планов."""
    plans = await admin_service.get_plans(db)
    return [
        AdminPlanResponse(
            id=p.id,
            code=p.code,
            name=p.name,
            description=p.description,
            max_requests_per_minute=p.max_requests_per_minute,
            max_audio_duration_sec=p.max_audio_duration_sec,
            price_per_month=float(p.price_per_month)
            if p.price_per_month is not None
            else None,
            is_active=p.is_active,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )
        for p in plans
    ]


@router.post("/tariffs", response_model=AdminPlanResponse)
async def admin_tariff_create(
    payload: AdminPlanCreateUpdateRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Создать тарифный план."""
    plan = await admin_service.create_plan(db, payload.model_dump())
    return AdminPlanResponse(
        id=plan.id,
        code=plan.code,
        name=plan.name,
        description=plan.description,
        max_requests_per_minute=plan.max_requests_per_minute,
        max_audio_duration_sec=plan.max_audio_duration_sec,
        price_per_month=float(plan.price_per_month)
        if plan.price_per_month is not None
        else None,
        is_active=plan.is_active,
        created_at=plan.created_at,
        updated_at=plan.updated_at,
    )


@router.put("/tariffs/{plan_id}", response_model=AdminPlanResponse)
async def admin_tariff_update(
    plan_id: str,
    payload: AdminPlanCreateUpdateRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Обновить тарифный план."""
    result = await db.execute(select(Plan).where(Plan.id == plan_id))
    plan = result.scalar_one_or_none()
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Тариф не найден"
        )
    plan = await admin_service.update_plan(
        db, plan, payload.model_dump(exclude_unset=True)
    )
    return AdminPlanResponse(
        id=plan.id,
        code=plan.code,
        name=plan.name,
        description=plan.description,
        max_requests_per_minute=plan.max_requests_per_minute,
        max_audio_duration_sec=plan.max_audio_duration_sec,
        price_per_month=float(plan.price_per_month)
        if plan.price_per_month is not None
        else None,
        is_active=plan.is_active,
        created_at=plan.created_at,
        updated_at=plan.updated_at,
    )


@router.delete("/tariffs/{plan_id}")
async def admin_tariff_delete(
    plan_id: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Деактивировать тариф."""
    result = await db.execute(select(Plan).where(Plan.id == plan_id))
    plan = result.scalar_one_or_none()
    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Тариф не найден"
        )
    await admin_service.delete_plan(db, plan)
    return {"detail": "Тариф деактивирован"}


@router.get("/subscriptions", response_model=list[AdminSubscriptionResponse])
async def admin_subscriptions_list(
    pagination: PaginationParams = Depends(),
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Список подписок."""
    subs, total = await admin_service.get_subscriptions(
        db, page=pagination.page, per_page=pagination.per_page
    )
    return [
        AdminSubscriptionResponse(
            id=s.id,
            user_id=s.user_id,
            plan_id=s.plan_id,
            plan_name=None,
            status=s.status,
            started_at=s.started_at,
            expires_at=s.expires_at,
            auto_renew=s.auto_renew,
        )
        for s in subs
    ]


@router.post("/subscriptions/{sub_id}/extend")
async def admin_subscription_extend(
    sub_id: str,
    days: int = 30,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Ручное продление подписки."""
    result = await db.execute(select(Subscription).where(Subscription.id == sub_id))
    sub = result.scalar_one_or_none()
    if not sub:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Подписка не найдена"
        )
    sub = await admin_service.extend_subscription(db, sub, days=days)
    return {"detail": f"Подписка продлена на {days} дней", "expires_at": sub.expires_at}


@router.post("/subscriptions/{sub_id}/cancel")
async def admin_subscription_cancel(
    sub_id: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Ручная отмена подписки."""
    result = await db.execute(select(Subscription).where(Subscription.id == sub_id))
    sub = result.scalar_one_or_none()
    if not sub:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Подписка не найдена"
        )
    await admin_service.cancel_subscription_admin(db, sub)
    return {"detail": "Подписка отменена"}


@router.get("/transactions", response_model=list[AdminTransactionResponse])
async def admin_transactions_list(
    pagination: PaginationParams = Depends(),
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Список транзакций."""
    txs, total = await admin_service.get_transactions(
        db, page=pagination.page, per_page=pagination.per_page
    )
    return [
        AdminTransactionResponse(
            id=t.id,
            user_id=t.user_id,
            subscription_id=t.subscription_id,
            amount=float(t.amount) if t.amount is not None else None,
            currency=t.currency,
            status=t.status,
            payment_provider=t.payment_provider,
            external_payment_id=t.external_payment_id,
            created_at=t.created_at,
        )
        for t in txs
    ]


@router.get("/logs", response_model=list[AdminSystemLogResponse])
async def admin_logs_list(
    pagination: PaginationParams = Depends(),
    level: Optional[str] = None,
    component: Optional[str] = None,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Системные логи."""
    logs, total = await admin_service.get_system_logs(
        db,
        page=pagination.page,
        per_page=pagination.per_page,
        level=level,
        component=component,
    )
    return [
        AdminSystemLogResponse(
            id=l.id,
            level=l.level,
            component=l.component,
            message=l.message,
            meta=l.meta,
            created_at=l.created_at,
        )
        for l in logs
    ]


@router.get("/audit", response_model=list[AdminAuditLogResponse])
async def admin_audit_list(
    pagination: PaginationParams = Depends(),
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Аудит действий админов."""
    logs, total = await admin_service.get_audit_logs(
        db, page=pagination.page, per_page=pagination.per_page
    )
    return [
        AdminAuditLogResponse(
            id=l.id,
            admin_id=l.admin_id,
            action=l.action,
            target_type=l.target_type,
            target_id=l.target_id,
            details=l.details,
            created_at=l.created_at,
        )
        for l in logs
    ]


@router.get("/api-keys", response_model=list[AdminApiKeyResponse])
async def admin_api_keys_list(
    pagination: PaginationParams = Depends(),
    user_id: Optional[str] = None,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Все API-ключи (с фильтром по пользователю)."""
    keys, total = await admin_service.get_api_keys(
        db, page=pagination.page, per_page=pagination.per_page, user_id=user_id
    )
    return [
        AdminApiKeyResponse(
            id=k.id,
            user_id=k.user_id,
            user_email=None,
            name=k.name,
            is_active=k.is_active,
            created_at=k.created_at,
            last_used_at=k.last_used_at,
        )
        for k in keys
    ]


@router.delete("/api-keys/{key_id}")
async def admin_api_key_revoke(
    key_id: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Отозвать API-ключ админом."""
    result = await db.execute(select(ApiKey).where(ApiKey.id == key_id))
    key = result.scalar_one_or_none()
    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Ключ не найден"
        )
    await admin_service.revoke_api_key_admin(db, key)
    return {"detail": "Ключ отозван"}


@router.get("/queue")
async def admin_queue(current_user: User = Depends(require_admin)):
    """Текущая очередь задач."""
    return await admin_service.get_queue_status()


@router.post("/queue/{task_id}/cancel")
async def admin_queue_cancel(
    task_id: str,
    current_user: User = Depends(require_admin),
):
    """Отменить задачу."""
    success = await admin_service.cancel_task(task_id)
    return {
        "detail": "Задача отменена" if success else "Не удалось отменить задачу"
    }


@router.post("/users/{user_id}/sessions/{session_id}/disconnect")
async def admin_disconnect_session(
    user_id: str,
    session_id: str,
    current_user: User = Depends(require_admin),
):
    """Принудительно закрыть WS-сессию пользователя."""
    success = await admin_service.disconnect_user_session(user_id, session_id)
    return {
        "detail": "Сессия закрыта" if success else "Не удалось закрыть сессию"
    }


@router.post("/maintenance")
async def admin_maintenance(
    payload: AdminMaintenanceToggle,
    current_user: User = Depends(require_admin),
):
    """Включить/выключить режим обслуживания."""
    admin_service.set_maintenance_mode(payload.enabled)
    return {
        "detail": f"Режим обслуживания {'включён' if payload.enabled else 'выключён'}"
    }


@router.get("/telegram/config", response_model=AdminTelegramConfig)
async def admin_telegram_config(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Настройки Telegram-бота."""
    cfg = await admin_service.get_telegram_config(db)
    if not cfg:
        return AdminTelegramConfig()
    return AdminTelegramConfig(webapp_url=cfg.webapp_url, is_active=cfg.is_active)


@router.post("/telegram/webhook")
async def admin_telegram_webhook(
    url: str,
    current_user: User = Depends(require_admin),
):
    """Установить webhook бота (заглушка)."""
    import os

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    return await admin_service.set_telegram_webhook(url, bot_token)


@router.get("/telegram/stats", response_model=AdminTelegramStats)
async def admin_telegram_stats(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db_session),
):
    """Статистика Telegram Web App."""
    stats = await admin_service.get_telegram_stats(db)
    return AdminTelegramStats(**stats)


@router.post("/telegram/broadcast")
async def admin_telegram_broadcast(
    payload: AdminBroadcastRequest,
    current_user: User = Depends(require_admin),
):
    """Рассылка сообщения всем пользователям бота (заглушка)."""
    return await admin_service.broadcast_message(payload.message)
