"""Бизнес-логика аутентификации (JWT, Telegram, регистрация)."""

import hashlib
import hmac
import time
import urllib.parse
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.enums import UserRole
from db.models import User
from core.security import (  # type: ignore[import-untyped]
    create_access_token,
    create_refresh_token,
    get_password_hash,
    verify_password,
)


# In-memory blacklist для refresh-токенов (в production → Redis)
_refresh_blacklist: dict[str, float] = {}


async def is_refresh_blacklisted(token: str) -> bool:
    """Проверяет, не отозван ли refresh-токен."""
    exp = _refresh_blacklist.get(token)
    if not exp:
        return False
    if time.time() > exp:
        _refresh_blacklist.pop(token, None)
        return False
    return True


async def blacklist_refresh_token(token: str, ttl_sec: int = 7 * 24 * 3600) -> None:
    """Добавляет refresh-токен в чёрный список."""
    _refresh_blacklist[token] = time.time() + ttl_sec


async def authenticate_user(db: AsyncSession, email: str, password: str) -> User | None:
    """Проверяет email/пароль и возвращает пользователя."""
    result = await db.execute(select(User).where(User.email == email))
    user: User | None = result.scalar_one_or_none()
    if not user or not user.is_active:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def generate_tokens(user: User) -> dict[str, str]:
    """Генерирует пару access/refresh токенов."""
    role_str = user.role.value if hasattr(user.role, "value") else user.role
    access = create_access_token({"sub": user.id, "role": role_str})
    refresh = create_refresh_token({"sub": user.id})
    return {"access_token": access, "refresh_token": refresh}


async def register_user(
    db: AsyncSession, email: str, password: str, full_name: str | None = None
) -> User:
    """Регистрирует нового пользователя."""
    result = await db.execute(select(User).where(User.email == email))
    if result.scalar_one_or_none():
        raise ValueError("Пользователь с таким email уже существует")

    user = User(
        email=email,
        hashed_password=get_password_hash(password),
        full_name=full_name,
        role=UserRole.user,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


def validate_telegram_init_data(init_data: str, bot_token: str) -> dict | None:
    """Проверяет подпись initData от Telegram Web App."""
    try:
        parsed = dict(urllib.parse.parse_qsl(init_data))
        received_hash = parsed.pop("hash", None)
        if not received_hash:
            return None

        data_check_string = "\n".join(f"{k}={v}" for k, v in sorted(parsed.items()))

        secret_key = hmac.new(
            key=b"WebAppData",
            msg=bot_token.encode(),
            digestmod=hashlib.sha256,
        ).digest()

        calculated_hash = hmac.new(
            key=secret_key,
            msg=data_check_string.encode(),
            digestmod=hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(calculated_hash, received_hash):
            return None

        return parsed
    except Exception:
        return None


async def get_or_create_telegram_user(db: AsyncSession, tg_data: dict) -> User:
    """Находит или создаёт пользователя по telegram_id."""
    telegram_id = int(tg_data.get("id", 0))
    if not telegram_id:
        raise ValueError("Отсутствует telegram_id")

    result = await db.execute(select(User).where(User.telegram_id == telegram_id))
    user: User | None = result.scalar_one_or_none()

    if user:
        # Обновляем Telegram-поля
        user.telegram_username = tg_data.get("username") or user.telegram_username
        user.telegram_first_name = tg_data.get("first_name") or user.telegram_first_name
        user.telegram_last_name = tg_data.get("last_name") or user.telegram_last_name
        user.telegram_photo_url = tg_data.get("photo_url") or user.telegram_photo_url
        user.telegram_auth_date = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(user)
        return user

    # Создаём нового пользователя без пароля (только Telegram)
    user = User(
        email=f"tg_{telegram_id}@telegram.local",
        hashed_password="",
        telegram_id=telegram_id,
        telegram_username=tg_data.get("username"),
        telegram_first_name=tg_data.get("first_name"),
        telegram_last_name=tg_data.get("last_name"),
        telegram_photo_url=tg_data.get("photo_url"),
        telegram_auth_date=datetime.now(timezone.utc),
        role=UserRole.user,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user
