"""Дополнительные middleware: rate limiting, maintenance mode."""

import time
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from services.admin_service import is_maintenance_mode


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting по количеству запросов в минуту (in-memory)."""

    def __init__(
        self,
        app,
        max_requests: int = 60,
        window_seconds: float = 60.0,
        exempt_paths: Optional[set[str]] = None,
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.exempt_paths = exempt_paths or {
            "/docs",
            "/openapi.json",
            "/static",
            "/admin/login",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/refresh",
            "/api/v1/auth/telegram",
            "/tg",
        }
        self._storage: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Maintenance mode для изменяющих запросов
        if is_maintenance_mode() and request.method not in ("GET", "HEAD", "OPTIONS"):
            return Response(
                content='{"detail":"Сервис на обслуживании"}',
                status_code=503,
                media_type="application/json",
            )

        # Пропускаем exempt пути
        if any(path.startswith(ep) for ep in self.exempt_paths):
            return await call_next(request)

        # Определяем ключ лимита
        api_key = request.headers.get("X-API-Key")
        auth_header = request.headers.get("Authorization", "")
        client_ip = request.client.host if request.client else "unknown"

        if api_key:
            import hashlib

            key = f"rate_limit:apikey:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        elif auth_header.startswith("Bearer "):
            token = auth_header[7:]
            key = f"rate_limit:jwt:{token[:16]}"
        else:
            key = f"rate_limit:ip:{client_ip}"

        now = time.time()
        timestamps = self._storage.get(key, [])
        timestamps = [t for t in timestamps if now - t < self.window_seconds]

        if len(timestamps) >= self.max_requests:
            retry_after = int(self.window_seconds - (now - timestamps[0]))
            return Response(
                content=f'{{"detail":"Превышен лимит запросов","retry_after":{retry_after}}}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(retry_after)},
            )

        timestamps.append(now)
        self._storage[key] = timestamps

        return await call_next(request)
