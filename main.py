from utils.do_logging import logger
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
from fastapi.openapi.utils import get_openapi
from utils.files_whatcher import start_file_watcher
from utils.pre_start_init import paths
import threading

from routes.ws_audio_transkrib import router as ws_audio_transkrib_router
from routes.v1 import router as v1_router
from routes.legacy import router as legacy_router
from models.fast_api_models import ErrorResponse
import models
from config import WS_DESCRIPTION

@asynccontextmanager
async def lifespan(app):
    # on_start
    logger.debug("Приложение FastAPI запущено")
    
    # Настройка сборщика мусора.
    gc.set_threshold(500, 5, 5)
    
    # Установка HF_HOME для HuggingFace Hub
    os.environ["HF_HOME"] = settings.HF_HOME
    
    if settings.DO_LOCAL_FILE_RECOGNITIONS:
        observer_thread = threading.Thread(
            target=lambda: start_file_watcher(file_path=str(paths.get("local_recognition_folder"))),
            daemon=True
        )
        observer_thread.start()
        logger.info("File watcher started")

    yield  # Здесь приложение работает


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


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Routers
app.include_router(ws_audio_transkrib_router, tags=["legacy"])
app.include_router(legacy_router, tags=["legacy"])
app.include_router(v1_router, tags=["v1"])

# def custom_openapi():
#     if app.openapi_schema:
#         return app.openapi_schema
#     openapi_schema = get_openapi(
#         title="ASR Speech Recognition API",
#         version="1.0.0",
#         description="Real-time Russian ASR via WebSocket. Send raw audio chunks (16 kHz, mono).",
#         routes=app.routes,
#         contact={"email": "Kojevnikov@amulex.ru"},
#     )
#     # Добавляем описание для WebSocket endpoint
#     # FastAPI автоматически генерирует WebSocket документацию, но мы можем улучшить её описание
#     if "/ws" in openapi_schema.get("paths", {}):
#         # Удаляем стандартный путь, так как он не подходит для WebSocket
#         del openapi_schema["paths"]["/ws"]
#
#     # Добавляем кастомное описание WebSocket
#     openapi_schema["paths"]["/ws"] = {
#         "summary": "WebSocket Stream for Audio Transcription",
#         "description": "Подключитесь по WebSocket для потоковой передачи аудио. Отправляйте raw audio bytes (PCM, 16-bit, mono, 16kHz). Получайте результаты транскрипции в JSON.",
#         "servers": [
#             {"url": "ws://{host}/ws", "description": "WebSocket server"}
#         ],
#         "x-postman-collection-name": "ASR WebSocket"
#     }
#
#     app.openapi_schema = openapi_schema
#     return openapi_schema




try:
    if __name__ == '__main__':
        # app.openapi = app.openapi_schema
        uvicorn.run(app, host=settings.HOST, port=settings.PORT)
except KeyboardInterrupt:
    logger.info('\nDone')
except Exception as e:
    logger.error(f'\nDone with error {e}')

