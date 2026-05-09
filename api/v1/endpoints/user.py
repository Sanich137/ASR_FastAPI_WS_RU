"""FastAPI-роутер пользовательского кабинета."""

import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.deps import get_current_user
from db.enums import SubscriptionStatus
from db.models import ApiKey, ASRSession, Plan, Subscription, User
from db.session import get_db_session
from models.user import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyResponse,
    ASRSessionItem,
    TelegramLinkStatus,
    TelegramUnlinkRequest,
    UserProfileResponse,
    UserProfileUpdateRequest,
    UserQuotaResponse,
    UserStatsResponse,
    UserSubscriptionResponse,
)

router = APIRouter(prefix="/user", tags=["user"])


@router.get("/profile", response_model=UserProfileResponse)
async def user_profile(current_user: User = Depends(get_current_user)):
    """Получить профиль текущего пользователя."""
    return UserProfileResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        phone=current_user.phone,
        role=current_user.role,
        is_active=current_user.is_active,
        telegram_linked=current_user.telegram_id is not None,
    )


@router.put("/profile", response_model=UserProfileResponse)
async def user_profile_update(
    payload: UserProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Обновить профиль текущего пользователя."""
    if payload.full_name is not None:
        current_user.full_name = payload.full_name
    if payload.phone is not None:
        current_user.phone = payload.phone
    await db.commit()
    await db.refresh(current_user)
    return UserProfileResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        phone=current_user.phone,
        role=current_user.role,
        is_active=current_user.is_active,
        telegram_linked=current_user.telegram_id is not None,
    )


@router.delete("/profile")
async def user_profile_delete(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Soft-delete аккаунта текущего пользователя."""
    current_user.is_active = False
    await db.commit()
    return {"detail": "Аккаунт деактивирован"}


@router.get("/quota", response_model=UserQuotaResponse)
async def user_quota(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Получить текущую квоту пользователя (заглушка до полной интеграции rate limiting)."""
    result = await db.execute(
        select(Subscription, Plan)
        .join(Plan, Subscription.plan_id == Plan.id)
        .where(
            Subscription.user_id == current_user.id,
            Subscription.status == SubscriptionStatus.active,
        )
    )
    row = result.first()
    if row:
        sub, plan = row
        return UserQuotaResponse(
            plan_name=plan.name,
            max_requests_per_minute=plan.max_requests_per_minute,
            requests_used_this_minute=0,  # TODO: интегрировать rate limiting счётчик
            max_audio_duration_sec=plan.max_audio_duration_sec,
        )
    return UserQuotaResponse()


@router.get("/subscription", response_model=UserSubscriptionResponse)
async def user_subscription(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Получить текущую подписку пользователя."""
    result = await db.execute(
        select(Subscription, Plan)
        .join(Plan, Subscription.plan_id == Plan.id)
        .where(Subscription.user_id == current_user.id)
        .order_by(Subscription.created_at.desc())
    )
    row = result.first()
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Подписка не найдена",
        )
    sub, plan = row
    return UserSubscriptionResponse(
        status=sub.status,
        plan_name=plan.name,
        started_at=sub.started_at,
        expires_at=sub.expires_at,
        auto_renew=sub.auto_renew,
    )


@router.post("/subscription/upgrade")
async def user_subscription_upgrade(
    current_user: User = Depends(get_current_user),
):
    """Заглушка для запроса на смену тарифа."""
    return {"detail": "Функция смены тарифа в разработке"}


@router.post("/subscription/cancel")
async def user_subscription_cancel(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Отмена авто-продления текущей подписки."""
    result = await db.execute(
        select(Subscription).where(
            Subscription.user_id == current_user.id,
            Subscription.status == SubscriptionStatus.active,
        )
    )
    sub: Optional[Subscription] = result.scalar_one_or_none()
    if not sub:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Активная подписка не найдена",
        )
    sub.auto_renew = False
    await db.commit()
    return {"detail": "Авто-продление отменено"}


@router.get("/sessions", response_model=list[ASRSessionItem])
async def user_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
    limit: int = 20,
    offset: int = 0,
):
    """Список ASR-сессий текущего пользователя."""
    result = await db.execute(
        select(ASRSession)
        .where(ASRSession.user_id == current_user.id)
        .order_by(ASRSession.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    sessions = result.scalars().all()
    return [
        ASRSessionItem(
            id=s.id,
            session_type=s.session_type,
            status=s.status,
            audio_duration_sec=s.audio_duration_sec,
            created_at=s.created_at,
            completed_at=s.completed_at,
        )
        for s in sessions
    ]


@router.get("/sessions/{session_id}")
async def user_session_detail(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Детали конкретной ASR-сессии."""
    result = await db.execute(
        select(ASRSession).where(
            ASRSession.id == session_id,
            ASRSession.user_id == current_user.id,
        )
    )
    session: Optional[ASRSession] = result.scalar_one_or_none()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Сессия не найдена",
        )
    return {
        "id": session.id,
        "session_type": session.session_type,
        "status": session.status,
        "audio_duration_sec": session.audio_duration_sec,
        "processing_duration_sec": session.processing_duration_sec,
        "cost": float(session.cost) if session.cost is not None else None,
        "result_json": session.result_json,
        "created_at": session.created_at,
        "completed_at": session.completed_at,
        "error_message": session.error_message,
    }


@router.get("/stats", response_model=UserStatsResponse)
async def user_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Агрегированная статистика пользователя."""
    total_result = await db.execute(
        select(func.count(ASRSession.id)).where(ASRSession.user_id == current_user.id)
    )
    total_sessions = total_result.scalar() or 0

    audio_result = await db.execute(
        select(func.coalesce(func.sum(ASRSession.audio_duration_sec), 0)).where(
            ASRSession.user_id == current_user.id
        )
    )
    total_audio_sec = audio_result.scalar() or 0

    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    month_result = await db.execute(
        select(func.count(ASRSession.id)).where(
            ASRSession.user_id == current_user.id,
            ASRSession.created_at >= month_start,
        )
    )
    sessions_this_month = month_result.scalar() or 0

    return UserStatsResponse(
        total_sessions=total_sessions,
        total_audio_hours=round(total_audio_sec / 3600, 2),
        sessions_this_month=sessions_this_month,
    )


@router.get("/api-keys", response_model=list[ApiKeyResponse])
async def user_api_keys(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Список API-ключей пользователя (без plain key)."""
    result = await db.execute(
        select(ApiKey)
        .where(ApiKey.user_id == current_user.id)
        .order_by(ApiKey.created_at.desc())
    )
    keys = result.scalars().all()
    return [
        ApiKeyResponse(
            id=k.id,
            name=k.name,
            is_active=k.is_active,
            created_at=k.created_at,
            last_used_at=k.last_used_at,
        )
        for k in keys
    ]


@router.post("/api-keys", response_model=ApiKeyCreateResponse)
async def user_api_key_create(
    payload: ApiKeyCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Создать новый API-ключ. Plain key возвращается только один раз."""
    plain_key = f"asr_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(plain_key.encode()).hexdigest()

    api_key = ApiKey(
        user_id=current_user.id,
        name=payload.name,
        key_hash=key_hash,
        is_active=True,
    )
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)

    return ApiKeyCreateResponse(
        id=api_key.id,
        name=api_key.name,
        is_active=api_key.is_active,
        created_at=api_key.created_at,
        last_used_at=api_key.last_used_at,
        plain_key=plain_key,
    )


@router.delete("/api-keys/{key_id}")
async def user_api_key_delete(
    key_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Отозвать API-ключ."""
    result = await db.execute(
        select(ApiKey).where(
            ApiKey.id == key_id,
            ApiKey.user_id == current_user.id,
        )
    )
    key: Optional[ApiKey] = result.scalar_one_or_none()
    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ключ не найден",
        )
    key.is_active = False
    await db.commit()
    return {"detail": "Ключ отозван"}


@router.get("/telegram/link", response_model=TelegramLinkStatus)
async def user_telegram_link(
    current_user: User = Depends(get_current_user),
):
    """Проверить, привязан ли Telegram-аккаунт."""
    return TelegramLinkStatus(
        linked=current_user.telegram_id is not None,
        telegram_username=current_user.telegram_username,
    )


@router.post("/telegram/unlink")
async def user_telegram_unlink(
    payload: TelegramUnlinkRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Отвязать Telegram-аккаунт от пользователя."""
    if not current_user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Нельзя отвязать Telegram для аккаунта без пароля. Установите пароль.",
        )

    current_user.telegram_id = None
    current_user.telegram_username = None
    current_user.telegram_first_name = None
    current_user.telegram_last_name = None
    current_user.telegram_photo_url = None
    current_user.telegram_auth_date = None
    await db.commit()
    return {"detail": "Telegram отвязан"}
