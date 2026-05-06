"""
Модуль services/ws_handler.py
Содержит класс MessageRouter и набор стандартных хендлеров
для обработки входящих WebSocket-сообщений в рамках ASR-сессии.
"""

import base64
import logging
from typing import Callable, Awaitable, Protocol

import numpy as np

from models.ws_models import (
    WSBaseMessage,
    WSConfigMessage,
    WSAudioMessage,
    WSStatusRequest,
    WSPingMessage,
    WSPongMessage,
    WSErrorMessage,
    WSStatusResponse,
    WSMessageType,
)
from services.ws_session import AudioSession, SessionState
from services.ws_metrics import SystemMetricsCollector

logger = logging.getLogger(__name__)


class WSManagerProtocol(Protocol):
    """
    Протокол для менеджера соединений, используемого хендлерами.

    Позволяет MessageRouter и хендлерам работать с любой реализацией
    менеджера (ConnectionManager, FakeManager в тестах и т.д.).
    """

    async def send_message(self, client_id: str, message: WSBaseMessage | dict) -> None:
        """
        Отправляет сообщение указанному клиенту.

        Args:
            client_id: Идентификатор клиента.
            message: Сообщение для отправки (Pydantic-модель или dict).
        """
        ...

    @property
    def active_connections_count(self) -> int:
        """Текущее количество активных соединений."""
        ...

    @property
    def max_connections(self) -> int:
        """Максимально допустимое количество соединений."""
        ...


async def handle_config(
    message: WSConfigMessage,
    session: AudioSession,
    manager: WSManagerProtocol,
) -> None:
    """
    Обрабатывает сообщение конфигурации от клиента.

    Устанавливает конфигурацию в AudioSession и переводит сессию
    в состояние receiving.

    Args:
        message: Сообщение конфигурации (WSConfigMessage).
        session: Текущая аудио-сессия.
        manager: Менеджер соединений (для отправки ответов при необходимости).
    """
    session.config = message
    session.state = SessionState.receiving
    logger.debug(
        "Config set for client %s: sample_rate=%d, transport=%s",
        session.client_id,
        message.sample_rate,
        message.audio_transport,
    )


async def handle_audio(
    message: WSAudioMessage,
    session: AudioSession,
    manager: WSManagerProtocol,
) -> None:
    """
    Обрабатывает аудио-чанк от клиента.

    Декодирует base64 в numpy-массив (int16 -> float32 нормализованный [-1, 1])
    и добавляет во внутренний буфер сессии.
    Если буфер переполнен — отправляет клиенту WSErrorMessage.

    Args:
        message: Сообщение с аудио (WSAudioMessage).
        session: Текущая аудио-сессия.
        manager: Менеджер соединений для отправки ошибок клиенту.
    """
    if message.audio_base64 is None:
        return

    try:
        raw_bytes = base64.b64decode(message.audio_base64)
        # Преобразуем int16 (PCM16) в float32 нормализованный [-1, 1]
        audio_array = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception as exc:
        logger.warning(
            "Failed to decode audio base64 for client %s: %s",
            session.client_id,
            exc,
        )
        error = WSErrorMessage(
            code="decode_error",
            message=f"Invalid audio_base64 data: {exc}",
            is_fatal=False,
        )
        await manager.send_message(session.client_id, error)
        return

    added = await session.add_audio(audio_array)
    if not added:
        logger.warning("Audio buffer overflow for client %s", session.client_id)
        error = WSErrorMessage(
            code="buffer_overflow",
            message="Audio buffer limit exceeded",
            is_fatal=False,
        )
        await manager.send_message(session.client_id, error)


async def handle_status_request(
    message: WSStatusRequest,
    session: AudioSession,
    manager: WSManagerProtocol,
    metrics_collector: SystemMetricsCollector,
) -> None:
    """
    Обрабатывает запрос статуса от клиента.

    Собирает метрики через SystemMetricsCollector и отправляет
    WSStatusResponse запросившему клиенту.

    Args:
        message: Запрос статуса (WSStatusRequest).
        session: Текущая аудио-сессия.
        manager: Менеджер соединений.
        metrics_collector: Коллектор системных метрик.
    """
    status = metrics_collector.collect(
        active_connections=manager.active_connections_count,
        max_connections=manager.max_connections,
    )
    await manager.send_message(session.client_id, status)


async def handle_ping(
    message: WSPingMessage,
    session: AudioSession,
    manager: WSManagerProtocol,
) -> None:
    """
    Обрабатывает ping-сообщение.

    Отправляет клиенту pong в ответ.

    Args:
        message: Ping-сообщение.
        session: Текущая аудио-сессия.
        manager: Менеджер соединений.
    """
    pong = WSPongMessage()
    await manager.send_message(session.client_id, pong)


class MessageRouter:
    """
    Маршрутизатор входящих WebSocket-сообщений.

    Регистрирует хендлеры для каждого WSMessageType и направляет
    сообщения в соответствующий обработчик.
    """

    def __init__(self) -> None:
        """
        Инициализирует пустой реестр хендлеров.
        """
        self._handlers: dict[WSMessageType, Callable[..., Awaitable[None]]] = {}

    def register_handler(
        self,
        msg_type: WSMessageType,
        handler: Callable[..., Awaitable[None]],
    ) -> None:
        """
        Регистрирует хендлер для указанного типа сообщения.

        Args:
            msg_type: Тип WebSocket-сообщения.
            handler: Асинхронная функция-обработчик.
        """
        self._handlers[msg_type] = handler
        logger.debug("Registered handler for %s", msg_type)

    async def route(
        self,
        message: WSBaseMessage,
        session: AudioSession,
        manager: WSManagerProtocol,
        metrics_collector: SystemMetricsCollector | None = None,
    ) -> None:
        """
        Направляет сообщение в зарегистрированный хендлер.

        Если хендлер не найден — отправляет клиенту WSErrorMessage.
        Для status_request требуется metrics_collector.

        Args:
            message: Входящее сообщение (любая модель WSBaseMessage).
            session: Текущая аудио-сессия.
            manager: Менеджер соединений.
            metrics_collector: Коллектор метрик (опционально, нужен для status_request).
        """
        handler = self._handlers.get(message.type)
        if handler is None:
            logger.warning("No handler registered for message type %s", message.type)
            error = WSErrorMessage(
                code="unsupported_type",
                message=f"Unsupported message type: {message.type}",
                is_fatal=False,
            )
            await manager.send_message(session.client_id, error)
            return

        try:
            if message.type == WSMessageType.status_request:
                if metrics_collector is None:
                    raise RuntimeError("metrics_collector required for status_request")
                await handler(message, session, manager, metrics_collector)
            else:
                await handler(message, session, manager)
        except Exception as exc:
            logger.exception("Handler error for %s: %s", message.type, exc)
            error = WSErrorMessage(
                code="handler_error",
                message=f"Internal handler error: {exc}",
                is_fatal=False,
            )
            await manager.send_message(session.client_id, error)
