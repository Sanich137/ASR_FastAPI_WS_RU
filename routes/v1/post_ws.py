from fastapi import APIRouter, WebSocket
from fastapi.responses import JSONResponse
from models.fast_api_models import V1BaseResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# TODO: Реализовать WebSocket-роут для потокового ASR
# Заглушка для будущей доработки в рамках Этапа 6
