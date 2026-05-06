"""
Модуль services/ws_manager.py
Содержит класс ConnectionManager для централизованного управления
WebSocket-соединениями: подключение, отключение, отправка сообщений,
broadcast статуса и graceful disconnect_all.
"""

import logging
import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect

from models.ws_models import WSBaseMessage, WSStatusResponse

logger = logging.getLogger(__name__)


@dataclass
class ConnectionMeta:
    """
    Мета-информация о WebSocket-соединении.

    Attributes:
        connected_at: Unix-timestamp установки соединения.
        last_activity_at: Unix-timestamp последней активности.
        bytes_received: Количество полученных байт.
        messages_received: Количество полученных сообщений.
        client_ip: IP-адрес клиента (или None).
        user_agent: User-Agent клиента (или None).
        subscribe_status: Флаг подписки на периодические status_response.
    """
    connected_at: float = field(default_factory=lambda: __import__("time").time())
    last_activity_at: float = field(default_factory=lambda: __import__("time").time())
    bytes_received: int = 0
    messages_received: int = 0
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    subscribe_status: bool = False


class ConnectionManager:
    """
    Централизованный менеджер активных WebSocket-соединений.

    Хранит только объекты WebSocket и мета-информацию в памяти текущего процесса.
    В будущем мета может выноситься в StateStore (Redis) для кластеризации.

    Attributes:
        max_connections: Максимальное количество одновременных соединений.
        active_connections: Словарь client_id -> WebSocket.
        connection_meta: Словарь client_id -> ConnectionMeta.
    """

    def __init__(self, max_connections: int = 100) -> None:
        """
        Инициализирует менеджер с заданным лимитом соединений.

        Args:
            max_connections: Максимально допустимое число активных соединений.
        """
        self.max_connections: int = max_connections
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_meta: Dict[str, ConnectionMeta] = {}

    @property
    def active_connections_count(self) -> int:
        """Текущее количество активных соединений."""
        return len(self.active_connections)

    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """
        Принимает новое WebSocket-соединение.

        Args:
            websocket: Объект WebSocket из FastAPI.
            client_id: Уникальный идентификатор клиента.

        Returns:
            True — соединение установлено и добавлено в реестр.
            False — превышен лимит соединений, websocket.close(1008) вызван.
        """
        if self.active_connections_count >= self.max_connections:
            logger.warning("Max connections (%d) reached, rejecting %s", self.max_connections, client_id)
            await websocket.close(code=1008, reason="Server overloaded")
            return False

        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_meta[client_id] = ConnectionMeta()
        logger.debug("Client %s connected. Total: %d", client_id, self.active_connections_count)
        return True

    async def disconnect(self, client_id: str) -> None:
        """
        Закрывает соединение и удаляет клиента из реестров.

        Args:
            client_id: Идентификатор клиента для отключения.
        """
        websocket = self.active_connections.pop(client_id, None)
        self.connection_meta.pop(client_id, None)
        if websocket is not None:
            try:
                await websocket.close()
            except Exception as exc:
                logger.debug("Error closing websocket for %s: %s", client_id, exc)
        logger.debug("Client %s disconnected. Total: %d", client_id, self.active_connections_count)

    async def send_message(self, client_id: str, message: WSBaseMessage | dict | str) -> None:
        """
        Отправляет сообщение указанному клиенту.

        Args:
            client_id: Идентификатор клиента.
            message: Сообщение (Pydantic-модель, dict или строка).
        """
        websocket = self.active_connections.get(client_id)
        if websocket is None:
            logger.warning("Cannot send message: client %s not found", client_id)
            return

        if isinstance(message, WSBaseMessage):
            payload = message.model_dump_json()
        elif isinstance(message, dict):
            import json
            payload = json.dumps(message, separators=(',', ':'))
        else:
            payload = str(message)

        try:
            await websocket.send_text(payload)
            meta = self.connection_meta.get(client_id)
            if meta is not None:
                meta.last_activity_at = __import__("time").time()
        except Exception as exc:
            logger.warning("Failed to send message to %s: %s", client_id, exc)

    async def broadcast_status(self, status: WSStatusResponse) -> None:
        """
        Отправляет статус всем активным соединениям, подписанным на статус.

        Args:
            status: Сообщение со статусом адаптера.
        """
        payload = status.model_dump_json()
        tasks = []
        for client_id, meta in list(self.connection_meta.items()):
            if meta.subscribe_status:
                ws = self.active_connections.get(client_id)
                if ws is not None:
                    tasks.append(self._send_text_safe(ws, payload, client_id))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_text_safe(self, websocket: WebSocket, payload: str, client_id: str) -> None:
        """Внутренний хелпер для безопасной отправки текста."""
        try:
            await websocket.send_text(payload)
        except Exception as exc:
            logger.debug("broadcast_status failed for %s: %s", client_id, exc)

    async def disconnect_all(self, code: int = 1001, reason: str = "Server shutdown") -> None:
        """
        Принудительно закрывает все активные соединения.

        Используется при graceful shutdown.

        Args:
            code: Код закрытия WebSocket (по умолчанию 1001 — going away).
            reason: Причина закрытия.
        """
        logger.info("Disconnecting all %d connections", self.active_connections_count)
        tasks = []
        for client_id, websocket in list(self.active_connections.items()):
            tasks.append(self._close_safe(websocket, code, reason, client_id))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self.active_connections.clear()
        self.connection_meta.clear()

    async def _close_safe(self, websocket: WebSocket, code: int, reason: str, client_id: str) -> None:
        """Внутренний хелпер для безопасного закрытия websocket."""
        try:
            await websocket.close(code=code, reason=reason)
        except Exception as exc:
            logger.debug("disconnect_all close error for %s: %s", client_id, exc)
