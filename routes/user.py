"""HTML-роуты пользовательского кабинета."""

from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates

from core.deps import get_current_user_or_none

templates = Jinja2Templates(directory="templates")
router = APIRouter()


@router.get("/login")
async def login_page(request: Request):
    """Страница входа."""
    return templates.TemplateResponse("auth/login.html", {"request": request})


@router.get("/register")
async def register_page(request: Request):
    """Страница регистрации."""
    return templates.TemplateResponse("auth/register.html", {"request": request})


@router.get("/dashboard")
async def dashboard_page(request: Request, user=Depends(get_current_user_or_none)):
    """Главная панель пользователя."""
    return templates.TemplateResponse(
        "user/dashboard.html",
        {"request": request, "user": user, "access_token": ""},
    )


@router.get("/asr")
async def asr_page(request: Request, user=Depends(get_current_user_or_none)):
    """Интерфейс распознавания."""
    return templates.TemplateResponse(
        "user/asr.html",
        {"request": request, "user": user, "access_token": ""},
    )


@router.get("/history")
async def history_page(request: Request, user=Depends(get_current_user_or_none)):
    """История сессий."""
    return templates.TemplateResponse(
        "user/history.html",
        {"request": request, "user": user, "access_token": ""},
    )


@router.get("/history/{session_id}")
async def history_detail_page(request: Request, session_id: str, user=Depends(get_current_user_or_none)):
    """Детали сессии."""
    return templates.TemplateResponse(
        "user/history_detail.html",
        {"request": request, "user": user, "session_id": session_id, "access_token": ""},
    )


@router.get("/subscription")
async def subscription_page(request: Request, user=Depends(get_current_user_or_none)):
    """Управление подпиской."""
    return templates.TemplateResponse(
        "user/subscription.html",
        {"request": request, "user": user, "access_token": ""},
    )


@router.get("/profile")
async def profile_page(request: Request, user=Depends(get_current_user_or_none)):
    """Профиль пользователя."""
    return templates.TemplateResponse(
        "user/profile.html",
        {"request": request, "user": user, "access_token": ""},
    )


@router.get("/api-keys")
async def api_keys_page(request: Request, user=Depends(get_current_user_or_none)):
    """Управление API-ключами."""
    return templates.TemplateResponse(
        "user/api_keys.html",
        {"request": request, "user": user, "access_token": ""},
    )
