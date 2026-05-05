import logging
import uvicorn
from config import settings
import os
import gc
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import MutableHeaders
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
import uuid
from core.logging_config import setup_logging, request_id_var
from core.exception_handlers import register_exception_handlers
from utils.files_whatcher import start_file_watcher
from utils.pre_start_init import paths
import threading
from VoiceActivityDetector import vad

from routes.ws_audio_transkrib import router as ws_audio_transkrib_router
from api.legacy import router as legacy_router
from api.v1.api import router as api_v1_router
import models
from config import WS_DESCRIPTION

logger = logging.getLogger(__name__)


class RequestIDMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        token = request_id_var.set(request_id)

        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=message["headers"])
                headers["X-Request-ID"] = request_id
                message["headers"] = headers.raw
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            request_id_var.reset(token)


class DeprecationHeaderMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_deprecation(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=message["headers"])
                path = scope.get("path", "")
                if path in {"/root/", "/root/demo", "/root/is_alive", "/root/post_file", "/root/post_one_step_req", "/root/ws"}:
                    headers["Deprecation"] = "true"
                    headers["Warning"] = f'299 - "Legacy API is deprecated. Use /api/v1/ instead."'
                message["headers"] = headers.raw
            await send(message)

        await self.app(scope, receive, send_with_deprecation)


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

    # Инициируем recognizer
    from Recognizer import Recognizer
    app.state.recognizer = Recognizer()

    # Инициируем punctuator
    from Punctuation import SbertPuncCaseOnnx
    app.state.punctuator = SbertPuncCaseOnnx(paths.get("punctuation_model_path"),use_gpu=settings.PUNCTUATE_WITH_GPU)

    # Инициируем diarizer
    if settings.CAN_DIAR:
        from Diarisation import ensure_diar_model
        if ensure_diar_model():
            from Diarisation.do_diarize import Diarizer
            app.state.diarizer = Diarizer(
                embedding_model_path=paths.get("diar_speaker_model_path"),
                vad=vad,  # todo- моежт быть использовать разные VAD для диаризации и разделения на чанки?
                max_phrase_gap=1,
                batch_size=settings.DIAR_GPU_BATCH_SIZE,
                cpu_workers=settings.CPU_WORKERS,
                use_gpu=settings.DIAR_WITH_GPU
            )
            logger.info("Модель диаризации загружена")
        else:
            settings.CAN_DIAR = False
            logger.warning("Диаризация недоступна: модель не найдена и не удалось скачать")

    if settings.DO_LOCAL_FILE_RECOGNITIONS:
        observer_thread = threading.Thread(
            target=lambda: start_file_watcher(file_path=str(paths.get("local_recognition_folder"))),
            daemon=True
        )
        observer_thread.start()
        logger.info("File watcher started")

    yield  # Здесь приложение работает

    # cleanup (если нужно)
    if hasattr(app.state, "recognizer"):
        del app.state.recognizer
    if hasattr(app.state, "punctuator"):
        del app.state.punctuator
    if hasattr(app.state, "diarizer"):
        del app.state.diarizer


app = FastAPI(
    lifespan=lifespan,
    version="1.0",
    docs_url='/docs',
    root_path='/root',
    title='ASR',
    description=WS_DESCRIPTION
    )

# Deprecation warning for legacy endpoints at startup
logger.warning(
    "Legacy endpoints (/ws, /post_file, /post_one_step_req, /is_alive, /demo, /) are deprecated. "
    "Use /api/v1/ instead.",
)

# RequestID middleware
app.add_middleware(RequestIDMiddleware)

# Deprecation header middleware for legacy endpoints
app.add_middleware(DeprecationHeaderMiddleware)

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
    minimum_size=500
)

# Exception handlers
register_exception_handlers(app)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Routers
app.include_router(ws_audio_transkrib_router, tags=["legacy"])
app.include_router(legacy_router, tags=["legacy"])
app.include_router(api_v1_router, tags=["api/v1"])

try:
    if __name__ == '__main__':
        # app.openapi = app.openapi_schema
        uvicorn.run(app, host=settings.HOST, port=settings.PORT)
except KeyboardInterrupt:
    logger.info('\nDone')
except Exception as e:
    logger.error(f'\nDone with error {e}')
