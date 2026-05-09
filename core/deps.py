"""FastAPI-зависимости для аутентификации и авторизации (RBAC + API Key)."""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.enums import UserRole
from db.models import ApiKey, User
from db.session import get_db_session

# Предполагается, что в core/security.py есть decode_token и TokenPayload
from core.security import decode_token  # type: ignore[import-untyped]

bearer_scheme = HTTPBearer(auto_error=False)


async def _extract_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None,
) -> str | None:
    """Извлекает токен из заголовка Authorization, cookie или query-параметра."""
    if credentials:
        return credentials.credentials
    token = request.cookies.get("access_token")
    if token:
        return token
    return request.query_params.get("token")


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db_session),
) -> User:
    """Возвращает текущего пользователя по JWT access token."""
    token = await _extract_token(request, credentials)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Не предоставлен токен авторизации",
        )

    payload = decode_token(token)
    if not payload or not getattr(payload, "sub", None):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Невалидный или просроченный токен",
        )

    result = await db.execute(select(User).where(User.id == payload.sub))
    user: User | None = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь не найден или деактивирован",
        )
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Проверяет, что пользователь — админ или суперадмин."""
    if user.role not in (UserRole.admin, UserRole.superadmin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Требуются права администратора",
        )
    return user


def require_superadmin(user: User = Depends(get_current_user)) -> User:
    """Проверяет, что пользователь — суперадмин."""
    if user.role != UserRole.superadmin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Требуются права суперадминистратора",
        )
    return user


async def get_current_user_or_none(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db_session),
) -> User | None:
    """Возвращает пользователя или None (для опциональной авторизации)."""
    try:
        return await get_current_user(request, credentials, db)
    except HTTPException:
        return None


async def require_api_key(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
) -> User:
    """Аутентификация по API-ключу из заголовка X-API-Key."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Не предоставлен API-ключ",
        )

    import hashlib
    from datetime import datetime, timezone

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    result = await db.execute(
        select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active.is_(True))
    )
    key_obj: ApiKey | None = result.scalar_one_or_none()
    if not key_obj:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Невалидный или отозванный API-ключ",
        )

    # Обновляем last_used_at
    key_obj.last_used_at = datetime.now(timezone.utc)
    await db.commit()

    result_user = await db.execute(select(User).where(User.id == key_obj.user_id))
    user: User | None = result_user.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь ключа не найден или деактивирован",
        )
    return user


async def require_api_key_or_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db_session),
) -> User:
    """Пробует JWT-аутентификацию, затем API-ключ."""
    try:
        return await get_current_user(request, credentials, db)
    except HTTPException:
        pass
    return await require_api_key(request, db)
