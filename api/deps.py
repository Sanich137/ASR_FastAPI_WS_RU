from datetime import datetime, timezone
from typing import Optional, Any

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer

from config import settings
from core.exceptions import (
    CredentialsException,
    PermissionDeniedException,
    RateLimitExceededException,
)
from core.security import decode_token
from models.domain.user import User
from models.enums import Role, SubscriptionType


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> User:
    """Получение текущего пользователя из токена или гостевого доступа."""
    if not token:
        return User(
            id="guest",
            role=Role.guest,
            daily_quota=settings.GUEST_DAILY_QUOTA,
            quota_used_today=0,
            is_active=True,
        )

    try:
        payload = decode_token(token, expected_type="access")
    except Exception:
        raise CredentialsException()

    # Заглушка: в реальности роль и квота берутся из БД
    return User(
        id=payload.sub,
        role=Role.user,
        daily_quota=settings.GUEST_DAILY_QUOTA,
        quota_used_today=0,
        is_active=True,
    )


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Проверка, что пользователь активен."""
    if not current_user.is_active:
        raise CredentialsException()
    return current_user


def require_role(*roles: Role):
    """Зависимость для проверки роли пользователя."""
    async def role_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        if current_user.role not in roles:
            raise PermissionDeniedException()
        return current_user

    return Depends(role_checker)


async def check_daily_quota(
    current_user: User = Depends(get_current_user),
) -> User:
    """Проверка дневной квоты пользователя."""
    # Админы и суперадмины не ограничены
    if current_user.role in (Role.admin, Role.superadmin):
        return current_user

    # Пользователи с активной подпиской pro/enterprise не ограничены
    if current_user.subscription_type in (
        SubscriptionType.pro,
        SubscriptionType.enterprise,
    ):
        if (
            current_user.subscription_expires is None
            or current_user.subscription_expires > datetime.now(timezone.utc)
        ):
            return current_user

    # Обычные пользователи и гости проверяются по квоте
    if current_user.quota_used_today >= current_user.daily_quota:
        raise RateLimitExceededException()

    return current_user


async def require_paid_access(
    current_user: User = Depends(get_current_user),
) -> User:
    """Заглушка для проверки разовых/рекуррентных платежей."""
    # TODO: реализовать проверку оплаты при подключении платёжной системы
    return current_user
