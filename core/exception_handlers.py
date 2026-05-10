import logging
import traceback

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from models.fast_api_models import ErrorResponse

logger = logging.getLogger(__name__)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            success=False,
            error_description="Validation error",
            details=str(exc),
        ).model_dump()
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    accept = request.headers.get("accept", "")
    is_html = "text/html" in accept

    if is_html and exc.status_code in (401, 403):
        login_url = "/admin/login" if request.url.path.startswith("/admin") else "/login"
        if request.url.path not in (login_url, "/login", "/admin/login"):
            return RedirectResponse(url=login_url, status_code=303)

    return JSONResponse(
        status_code=exc.status_code,
        headers=exc.headers,
        content=ErrorResponse(
            success=False,
            error_description=exc.detail,
        ).model_dump()
    )


async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error_description="Internal server error",
            details=str(exc),
        ).model_dump()
    )


def register_exception_handlers(app):
    """Регистрация обработчиков исключений на экземпляре FastAPI."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
