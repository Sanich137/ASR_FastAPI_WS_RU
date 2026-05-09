"""
WebSocket-роут /api/v1/asr/ws
Использует ConnectionManager, AudioSession, MessageRouter, asr_pipeline.
Сохраняет обратную совместимость протокола (config, audio, eof/eos).
"""

import asyncio
import base64
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from config import settings
from models.ws_models import (
    WSConfigMessage,
    WSAudioMessage,
    WSEosMessage,
    WSErrorMessage,
    WSMessageType,
    parse_ws_message,
)
from services.ws_manager import ConnectionManager
from services.ws_session import AudioSession, SessionState
from services.ws_handler import MessageRouter, handle_config, handle_ping, handle_status_request
from services.ws_metrics import SystemMetricsCollector
from services.asr_pipeline import process_audio_stream_chunk, process_final_audio
from Recognizer import get_recognizer, Recognizer
from Punctuation import get_punctuator, SbertPuncCaseOnnx
from db.session import get_db_session
from db.models import ASRSession
from db.enums import ASRSessionStatus, ASRSessionType
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone

router = APIRouter(prefix="/asr", tags=["ASR"])
logger = logging.getLogger(__name__)


@asynccontextmanager
async def audio_session_lifecycle(client_id: str):
    """
    Контекстный менеджер жизненного цикла AudioSession.

    Гарантирует очистку AudioSegment-буферов.
    """
    session = AudioSession(client_id=client_id)
    try:
        yield session
    finally:
        await session.reset()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    recognizer: Recognizer = Depends(get_recognizer),
    punctuator: SbertPuncCaseOnnx = Depends(get_punctuator),
    db: AsyncSession = Depends(get_db_session),
):
    """
    WebSocket endpoint для потокового распознавания речи (ASR).

    Протокол:
      1. Клиент отправляет config (WSConfigMessage).
      2. Клиент отправляет audio_chunk (WSAudioMessage или binary frame).
      3. По завершении — eos/eof (WSEosMessage или текст "eof").
    """
    manager: ConnectionManager = websocket.app.state.ws_manager
    metrics: SystemMetricsCollector = websocket.app.state.metrics_collector
    state_store = websocket.app.state.state_store

    client_id = str(uuid.uuid4())
    logger.info("New WS connection: %s", client_id)

    # 1. Подключение (с проверкой лимита соединений)
    if not await manager.connect(websocket, client_id):
        logger.warning("Connection rejected for %s (max connections reached)", client_id)
        return

    # 1a. Ожидание первого фрейма (auth или config) — таймаут 5 сек
    user_id = None
    pending_message = None
    try:
        auth_msg = await asyncio.wait_for(websocket.receive(), timeout=5.0)
        if auth_msg.get("type") == "websocket.disconnect":
            logger.info("Client %s disconnected before auth", client_id)
            return
        if auth_msg.get("text"):
            import json
            auth_data = json.loads(auth_msg["text"])
            if auth_data.get("type") == "auth":
                token = auth_data.get("access_token", "")
                from core.security import decode_token
                payload = decode_token(token)
                if payload and getattr(payload, "sub", None):
                    user_id = payload.sub
            else:
                # Первое сообщение не auth — сохраняем для обработки в цикле (config и т.д.)
                pending_message = auth_msg
    except asyncio.TimeoutError:
        # Гостевой доступ: не закрываем соединение, просто логируем
        logger.info("No auth for client %s, continuing as guest", client_id)
    except Exception as exc:
        logger.warning("Auth error for client %s: %s", client_id, exc)

    # 2. Жизненный цикл сессии (гарантированная очистка в finally)
    async with audio_session_lifecycle(client_id) as session:
        asr_db_session = None
        if user_id:
            session.user_id = user_id  # привязка к пользователю (Этап 5)
            # Создаём запись ASRSession в БД
            asr_db_session = ASRSession(
                user_id=user_id,
                session_type=ASRSessionType.websocket,
                status=ASRSessionStatus.processing,
                request_ip=websocket.client.host if websocket.client else None,
            )
            db.add(asr_db_session)
            await db.commit()
            await db.refresh(asr_db_session)

        # 3. Регистрация хендлеров сообщений
        msg_router = MessageRouter()
        msg_router.register_handler(WSMessageType.config, handle_config)
        msg_router.register_handler(WSMessageType.ping, handle_ping)
        msg_router.register_handler(WSMessageType.status_request, handle_status_request)

        try:
            while True:
                # 4. Получение сообщения с idle timeout
                try:
                    if pending_message is not None:
                        message = pending_message
                        pending_message = None
                    else:
                        message = await asyncio.wait_for(
                            websocket.receive(),
                            timeout=settings.WS_IDLE_TIMEOUT_SEC,
                        )
                except asyncio.TimeoutError:
                    logger.info("Idle timeout for client %s", client_id)
                    await manager.send_message(
                        client_id,
                        WSErrorMessage(
                            code="idle_timeout",
                            message=f"No messages for {settings.WS_IDLE_TIMEOUT_SEC}s",
                            is_fatal=False,
                        ),
                    )
                    break

                # Обработка disconnect от клиента
                if message.get("type") == "websocket.disconnect":
                    logger.info("Client %s disconnected (code=%s)", client_id, message.get("code"))
                    break

                # Определяем тип содержимого: bytes (binary) или text (JSON)
                if message.get("bytes"):
                    if session.config is None:
                        logger.warning("Audio chunk received before config from %s", client_id)
                        await manager.send_message(
                            client_id,
                            WSErrorMessage(
                                code="missing_config",
                                message="Send config before audio chunks",
                                is_fatal=False,
                            ),
                        )
                        continue
                    # Binary frame: отправляем сырые байты напрямую в pipeline, без base64-обёртки
                    await process_audio_stream_chunk(
                        session, message["bytes"], recognizer, punctuator, manager, metrics
                    )
                    continue
                elif message.get("text"):
                    msg = parse_ws_message(message["text"])
                else:
                    logger.warning("Unknown WS message format for %s: %s", client_id, message)
                    continue

                # 5. Маршрутизация служебных сообщений
                if msg.type in (
                    WSMessageType.config,
                    WSMessageType.ping,
                    WSMessageType.status_request,
                ):
                    await msg_router.route(msg, session, manager, metrics_collector=metrics)

                # Подписка на периодический статус при запросе status
                if msg.type == WSMessageType.status_request:
                    manager.set_subscribe_status(session.client_id, True)

                # Копирование флагов из конфига в сессию (для ASR pipeline)
                if isinstance(msg, WSConfigMessage):
                    session.config = msg
                    session.wait_null_answers = msg.wait_null_answers
                    session.do_dialogue = msg.do_dialogue
                    session.do_punctuation = msg.do_punctuation
                    session.channel_name = msg.channel_name or "Null"
                    logger.debug(
                        "Config set for %s: sample_rate=%d, format=%s, transport=%s",
                        client_id,
                        msg.sample_rate,
                        msg.audio_format,
                        msg.audio_transport,
                    )

                # 6. Обработка аудио-чанка
                if isinstance(msg, WSAudioMessage):
                    if session.config is None:
                        logger.warning("Audio chunk received before config from %s", client_id)
                        await manager.send_message(
                            client_id,
                            WSErrorMessage(
                                code="missing_config",
                                message="Send config before audio chunks",
                                is_fatal=False,
                            ),
                        )
                        continue
                    chunk_bytes = b""
                    if msg.audio_base64:
                        chunk_bytes = base64.b64decode(msg.audio_base64)
                    if chunk_bytes:
                        await process_audio_stream_chunk(
                            session, chunk_bytes, recognizer, punctuator, manager, metrics
                        )

                # 7. Обработка конца потока (eos/eof)
                if isinstance(msg, WSEosMessage):
                    await process_final_audio(session, recognizer, punctuator, manager, metrics)
                    break

        except WebSocketDisconnect:
            logger.info("Client %s disconnected normally", client_id)
        except Exception as exc:
            logger.exception("WS error for %s: %s", client_id, exc)
            if asr_db_session:
                try:
                    asr_db_session.status = ASRSessionStatus.failed
                    asr_db_session.error_message = str(exc)
                    asr_db_session.completed_at = datetime.now(timezone.utc)
                    await db.commit()
                except Exception:
                    pass
            try:
                await manager.send_message(
                    client_id,
                    WSErrorMessage(
                        code="internal_error",
                        message=str(exc),
                        is_fatal=True,
                    ),
                )
            except Exception:
                pass
        finally:
            logger.info("Closing WS connection %s", client_id)
            # Сохранение результата в БД
            if asr_db_session:
                try:
                    asr_db_session.status = ASRSessionStatus.completed
                    asr_db_session.completed_at = datetime.now(timezone.utc)
                    asr_db_session.result_json = session.ws_collected_asr_res
                    await db.commit()
                except Exception as exc:
                    logger.debug("Failed to save ASRSession result: %s", exc)
            # Сохранение мета-информации в StateStore (аудит / восстановление)
            try:
                await state_store.set(f"session:{client_id}", session.to_dict())
            except Exception as exc:
                logger.debug("Failed to save session state: %s", exc)
            await manager.disconnect(client_id)
