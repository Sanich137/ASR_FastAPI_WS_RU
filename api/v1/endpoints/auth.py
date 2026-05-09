"""FastAPI-роутер аутентификации: JWT, Telegram, logout, change-password."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings

from core.deps import get_current_user
from core.security import decode_token  # type: ignore[import-untyped]
from db.models import User
from db.session import get_db_session
from models.auth import (
    ChangePasswordRequest,
    TelegramAuthRequest,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
)
from services.auth_service import (
    authenticate_user,
    blacklist_refresh_token,
    generate_tokens,
    get_or_create_telegram_user,
    is_refresh_blacklisted,
    register_user,
    validate_telegram_init_data,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
async def auth_register(
    payload: UserRegisterRequest,
    response: Response,
    db: AsyncSession = Depends(get_db_session),
):
    """Регистрация нового пользователя. Refresh-токен в httpOnly cookie."""
    try:
        user = await register_user(db, payload.email, payload.password, payload.full_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    tokens = generate_tokens(user)
    response.set_cookie(
        key="refresh_token",
        value=tokens["refresh_token"],
        httponly=True,
        secure=settings.IS_PROD,
        samesite="strict",
        max_age=7 * 24 * 3600,
    )
    return TokenResponse(access_token=tokens["access_token"])


@router.post("/login", response_model=TokenResponse)
async def auth_login(
    payload: UserLoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db_session),
):
    """Вход по email/пароль. Refresh-токен в httpOnly cookie."""
    user = await authenticate_user(db, payload.email, payload.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный email или пароль",
        )

    tokens = generate_tokens(user)
    response.set_cookie(
        key="refresh_token",
        value=tokens["refresh_token"],
        httponly=True,
        secure=settings.IS_PROD,
        samesite="strict",
        max_age=7 * 24 * 3600,
    )
    return TokenResponse(access_token=tokens["access_token"])


@router.post("/refresh", response_model=TokenResponse)
async def auth_refresh(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db_session),
):
    """Обмен валидного refresh cookie на новую пару токенов."""
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Отсутствует refresh токен",
        )

    if await is_refresh_blacklisted(refresh_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh токен отозван",
        )

    payload = decode_token(refresh_token)
    if not payload or not getattr(payload, "sub", None):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Невалидный refresh токен",
        )

    from sqlalchemy import select

    result = await db.execute(select(User).where(User.id == payload.sub))
    user: User | None = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь не найден",
        )

    tokens = generate_tokens(user)
    response.set_cookie(
        key="refresh_token",
        value=tokens["refresh_token"],
        httponly=True,
        secure=settings.IS_PROD,
        samesite="strict",
        max_age=7 * 24 * 3600,
    )
    return TokenResponse(access_token=tokens["access_token"])


@router.post("/logout")
async def auth_logout(request: Request, response: Response):
    """Инвалидация refresh токена (blacklist) + очистка cookie."""
    refresh_token = request.cookies.get("refresh_token")
    if refresh_token:
        await blacklist_refresh_token(refresh_token, ttl_sec=7 * 24 * 3600)
    response.delete_cookie(key="refresh_token", httponly=True, secure=settings.IS_PROD, samesite="strict")
    return {"detail": "Успешный выход"}


@router.post("/change-password")
async def auth_change_password(
    payload: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Смена пароля текущего пользователя (требуется старый пароль)."""
    from core.security import get_password_hash, verify_password  # type: ignore[import-untyped]

    if not verify_password(payload.old_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Неверный текущий пароль",
        )
    current_user.hashed_password = get_password_hash(payload.new_password)
    await db.commit()
    return {"detail": "Пароль успешно изменён"}


@router.get("/me")
async def auth_me(current_user: User = Depends(get_current_user)):
    """Текущий аутентифицированный пользователь."""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "is_active": current_user.is_active,
    }


@router.post("/telegram", response_model=TokenResponse)
async def auth_telegram(
    payload: TelegramAuthRequest,
    response: Response,
    db: AsyncSession = Depends(get_db_session),
):
    """Аутентификация через Telegram Web App (initData + HMAC-проверка)."""
    import os

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not bot_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Telegram бот не настроен",
        )

    tg_data = validate_telegram_init_data(payload.init_data, bot_token)
    if not tg_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Невалидная подпись Telegram initData",
        )

    user = await get_or_create_telegram_user(db, tg_data)
    tokens = generate_tokens(user)
    response.set_cookie(
        key="refresh_token",
        value=tokens["refresh_token"],
        httponly=True,
        secure=settings.IS_PROD,
        samesite="strict",
        max_age=7 * 24 * 3600,
    )
    return TokenResponse(access_token=tokens["access_token"])
