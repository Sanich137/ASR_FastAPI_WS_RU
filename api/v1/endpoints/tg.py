"""Точка входа для Telegram Web App."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["telegram"])


@router.get("/tg", response_class=HTMLResponse)
async def telegram_webapp_entry(request: Request):
    """Возвращает HTML-страницу входа для Telegram Web App."""
    # В будущем заменить на TemplateResponse("tg/index.html", {"request": request})
    html = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>ASR Telegram</title>
    <script src="https://telegram.org/js/telegram-web-app.js"></script>
    <style>
        body { margin: 0; padding: 16px; font-family: sans-serif; background: var(--tg-theme-bg-color, #fff); color: var(--tg-theme-text-color, #000); }
        button { width: 100%; padding: 12px; margin-top: 12px; font-size: 16px; }
    </style>
</head>
<body>
    <h2>ASR Сервис</h2>
    <p id="status">Авторизация...</p>
    <script>
        const tg = window.Telegram.WebApp;
        tg.ready();
        tg.expand();
        async function auth() {
            const status = document.getElementById('status');
            try {
                const resp = await fetch('/api/v1/auth/telegram', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ init_data: tg.initData })
                });
                if (!resp.ok) throw new Error('Auth failed');
                const data = await resp.json();
                window.__ACCESS_TOKEN__ = data.access_token;
                status.textContent = 'Авторизация успешна';
                tg.MainButton.setText('Распознать');
                tg.MainButton.show();
            } catch (e) {
                status.textContent = 'Ошибка авторизации: ' + e.message;
            }
        }
        auth();
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)
