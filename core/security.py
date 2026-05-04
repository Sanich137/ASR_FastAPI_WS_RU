import bcrypt
import jwt

from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel

from config import settings
from core.exceptions import InvalidTokenException, TokenExpiredException


class TokenPayload(BaseModel):
    """Типизированная модель payload JWT-токена."""
    sub: str
    exp: datetime
    iat: datetime
    type: Literal["access", "refresh"]


def get_password_hash(password: str) -> str:
    """Генерация хеша пароля через bcrypt с использованием настроек из конфига."""
    salt = bcrypt.gensalt(rounds=settings.BCRYPT_ROUNDS)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверка пароля против хеша через bcrypt."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Создание access-токена."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": now, "type": "access"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Создание refresh-токена."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": now, "type": "refresh"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str, expected_type: Literal["access", "refresh"] | None = None) -> TokenPayload:
    """Декодирование и валидация JWT-токена."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise TokenExpiredException()
    except jwt.InvalidTokenError:
        raise InvalidTokenException()

    token_type = payload.get("type")
    if expected_type is not None and token_type != expected_type:
        raise InvalidTokenException()

    exp = payload.get("exp")
    iat = payload.get("iat")
    if exp is None:
        raise InvalidTokenException()
    exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc)
    iat_dt = datetime.fromtimestamp(iat, tz=timezone.utc) if iat is not None else exp_dt

    sub = payload.get("sub")
    if sub is None:
        raise InvalidTokenException()

    return TokenPayload(sub=sub, exp=exp_dt, iat=iat_dt, type=token_type)
