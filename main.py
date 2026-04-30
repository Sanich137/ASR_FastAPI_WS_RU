from Recognizer import Recognizer
import logging
import uvicorn
from config import settings
import os
import gc
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
import uuid
from core.logging_config import setup_logging, request_id_var
from utils.files_whatcher import start_file_watcher
from utils.pre_start_init import paths
import threading

from routes.ws_audio_transkrib import router as ws_audio_transkrib_router
from routes.v1 import router as v1_router
from routes.legacy import router as legacy_router
from models.fast_api_models import ErrorResponse
import models
from config import WS_DESCRIPTION

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_var.reset(token)


@asynccontextmanager
async def lifespan(app):
    # Настройка логирования до любых других операций
    setup_logging()

    # Установка HF_HOME для HuggingFace Hub
    os.environ["HF_HOME"] = settings.HF_HOME

    # on_start
    logger.debug("Приложение FastAPI запущено")

    # Настройка сборщика мусора.
    gc.set_threshold(500, 5, 5)

    app.state.recognizer = Recognizer()

    if settings.DO_LOCAL_FILE_RECOGNITIONS:
        observer_thread = threading.Thread(
            target=lambda: start_file_watcher(file_path=str(paths.get("local_recognition_folder"))),
            daemon=True
        )
        observer_thread.start()
        logger.info("File watcher started")

    yield  # Здесь приложение работает
    # cleanup (если нужно)
    del app.state.recognize


app = FastAPI(
    lifespan=lifespan,
    version="1.0",
    docs_url='/docs',
    root_path='/root',
    title='ASR',
    description=WS_DESCRIPTION
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            success=False,
            error_description="Validation error",
            details=str(exc),
        ).model_dump()
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error_description=exc.detail,
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    import traceback
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error_description="Internal server error",
            details=str(exc),
        ).model_dump()
    )


# RequestID middleware
app.add_middleware(RequestIDMiddleware)

# ProxyHeaders middleware
app.add_middleware(
    ProxyHeadersMiddleware,
    trusted_hosts=settings.TRUSTED_PROXIES,
)

# TrustedHost middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip middleware
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Routers
app.include_router(ws_audio_transkrib_router, tags=["legacy"])
app.include_router(legacy_router, tags=["legacy"])
app.include_router(v1_router, tags=["v1"])

try:
    if __name__ == '__main__':
        # app.openapi = app.openapi_schema
        uvicorn.run(app, host=settings.HOST, port=settings.PORT)
except KeyboardInterrupt:
    logger.info('\nDone')
except Exception as e:
    logger.error(f'\nDone with error {e}')
