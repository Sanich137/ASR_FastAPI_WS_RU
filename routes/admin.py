"""HTML-роуты админ-панели (Jinja2).

TODO: подключить router в main.py через app.include_router(routes.admin.router).
"""

from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates

from core.deps import require_admin

templates = Jinja2Templates(directory="templates")
router = APIRouter(prefix="/admin")


@router.get("/dashboard")
async def admin_dashboard_page(
    request: Request,
    _admin=Depends(require_admin),
):
    """Страница dashboard админ-панели."""
    return templates.TemplateResponse("admin/dashboard.html", {"request": request})


@router.get("/login")
async def admin_login_page(request: Request):
    """Страница входа в админ-панель."""
    return templates.TemplateResponse("admin/login.html", {"request": request})


@router.get("/users")
async def admin_users_page(
    request: Request,
    _admin=Depends(require_admin),
):
    """Страница управления пользователями."""
    return templates.TemplateResponse("admin/users.html", {"request": request})


@router.get("/users/{user_id}")
async def admin_user_detail_page(
    request: Request,
    user_id: str,
    _admin=Depends(require_admin),
):
    """Страница деталей пользователя."""
    return templates.TemplateResponse("admin/user_detail.html", {"request": request, "user_id": user_id})


@router.get("/sessions")
async def admin_sessions_page(
    request: Request,
    _admin=Depends(require_admin),
):
    """Страница мониторинга сессий."""
    return templates.TemplateResponse("admin/sessions.html", {"request": request})


@router.get("/sessions/{session_id}")
async def admin_session_detail_page(
    request: Request,
    session_id: str,
    _admin=Depends(require_admin),
):
    """Страница деталей сессии."""
    return templates.TemplateResponse("admin/session_detail.html", {"request": request, "session_id": session_id})


@router.get("/subscriptions")
async def admin_subscriptions_page(
    request: Request,
    _admin=Depends(require_admin),
):
    """Страница управления подписками."""
    return templates.TemplateResponse("admin/subscriptions.html", {"request": request})


@router.get("/transactions")
async def admin_transactions_page(
    request: Request,
    _admin=Depends(require_admin),
):
    """Страница транзакций."""
    return templates.TemplateResponse("admin/transactions.html", {"request": request})


@router.get("/tariffs")
async def admin_tariffs_page(
    request: Request,
    _admin=Depends(require_admin),
):
    """Страница тарифных планов."""
    return templates.TemplateResponse("admin/tariffs.html", {"request": request})


@router.get("/api-keys")
async def admin_api_keys_page(
    request: Request,
    _admin=Depends(require_admin),
):
    """Страница управления API-ключами."""
    return templates.TemplateResponse("admin/api_keys.html", {"request": request})


@router.get("/logs")
async def admin_logs_page(
    request: Request,
    _admin=Depends(require_admin),
):
    """Страница системных логов."""
    return templates.TemplateResponse("admin/logs.html", {"request": request})


@router.get("/settings")
async def admin_settings_page(
    request: Request,
    _admin=Depends(require_admin),
):
    """Страница настроек и maintenance mode."""
    return templates.TemplateResponse("admin/settings.html", {"request": request})


@router.get("/telegram")
async def admin_telegram_page(
    request: Request,
    _admin=Depends(require_admin),
):
    """Страница настроек Telegram-интеграции."""
    return templates.TemplateResponse("admin/telegram.html", {"request": request})
