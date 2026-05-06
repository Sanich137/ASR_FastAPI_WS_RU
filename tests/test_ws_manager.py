"""
Тесты для services/ws_manager.py (ConnectionManager).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from services.ws_manager import ConnectionManager, ConnectionMeta
from models.ws_models import WSStatusResponse, WSPingMessage


class FakeWebSocket:
    """
    Фейковый WebSocket для тестирования ConnectionManager.
    Имитирует минимальный интерфейс FastAPI WebSocket.
    """

    def __init__(self):
        self.accepted = False
        self.closed = False
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self.sent_texts: list[str] = []

    async def accept(self):
        self.accepted = True

    async def close(self, code: int = 1000, reason: str = ""):
        self.closed = True
        self.close_code = code
        self.close_reason = reason

    async def send_text(self, data: str):
        self.sent_texts.append(data)


@pytest.fixture
def manager() -> ConnectionManager:
    """Фикстура: ConnectionManager с лимитом 3 соединения."""
    return ConnectionManager(max_connections=3)


class TestConnectionManagerConnect:
    @pytest.mark.asyncio
    async def test_connect_success(self, manager: ConnectionManager) -> None:
        """Успешное подключение добавляет клиента в реестр."""
        ws = FakeWebSocket()
        result = await manager.connect(ws, "client-1")
        assert result is True
        assert ws.accepted is True
        assert manager.active_connections_count == 1
        assert "client-1" in manager.active_connections

    @pytest.mark.asyncio
    async def test_connect_rejects_when_full(self, manager: ConnectionManager) -> None:
        """При превышении лимита соединение отклоняется с кодом 1008."""
        for i in range(3):
            ws = FakeWebSocket()
            assert await manager.connect(ws, f"client-{i}") is True

        ws_rejected = FakeWebSocket()
        result = await manager.connect(ws_rejected, "client-overflow")
        assert result is False
        assert ws_rejected.closed is True
        assert ws_rejected.close_code == 1008


class TestConnectionManagerDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_removes_client(self, manager: ConnectionManager) -> None:
        """Отключение удаляет клиента из реестров."""
        ws = FakeWebSocket()
        await manager.connect(ws, "client-a")
        await manager.disconnect("client-a")
        assert manager.active_connections_count == 0
        assert "client-a" not in manager.active_connections
        assert ws.closed is True

    @pytest.mark.asyncio
    async def test_disconnect_unknown_client(self, manager: ConnectionManager) -> None:
        """Отключение несуществующего клиента не вызывает ошибок."""
        await manager.disconnect("ghost")
        assert manager.active_connections_count == 0


class TestConnectionManagerSendMessage:
    @pytest.mark.asyncio
    async def test_send_pydantic_model(self, manager: ConnectionManager) -> None:
        """Отправка Pydantic-модели сериализуется в JSON."""
        ws = FakeWebSocket()
        await manager.connect(ws, "client-b")
        msg = WSPingMessage()
        await manager.send_message("client-b", msg)
        assert len(ws.sent_texts) == 1
        assert '"type":"ping"' in ws.sent_texts[0]

    @pytest.mark.asyncio
    async def test_send_dict(self, manager: ConnectionManager) -> None:
        """Отправка dict сериализуется в JSON."""
        ws = FakeWebSocket()
        await manager.connect(ws, "client-c")
        await manager.send_message("client-c", {"foo": "bar"})
        assert len(ws.sent_texts) == 1
        assert '"foo":"bar"' in ws.sent_texts[0]

    @pytest.mark.asyncio
    async def test_send_to_unknown_client(self, manager: ConnectionManager) -> None:
        """Отправка несуществующему клиенту не вызывает ошибок."""
        await manager.send_message("unknown", WSPingMessage())


class TestConnectionManagerBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_status_only_subscribed(self, manager: ConnectionManager) -> None:
        """Broadcast отправляет статус только подписанным клиентам."""
        ws1 = FakeWebSocket()
        ws2 = FakeWebSocket()
        await manager.connect(ws1, "sub-1")
        await manager.connect(ws2, "sub-2")
        manager.connection_meta["sub-1"].subscribe_status = True
        manager.connection_meta["sub-2"].subscribe_status = False

        status = WSStatusResponse(adapter_status="idle")
        await manager.broadcast_status(status)
        assert len(ws1.sent_texts) == 1
        assert len(ws2.sent_texts) == 0


class TestConnectionManagerDisconnectAll:
    @pytest.mark.asyncio
    async def test_disconnect_all_closes_everyone(self, manager: ConnectionManager) -> None:
        """disconnect_all закрывает все соединения и очищает реестры."""
        ws1 = FakeWebSocket()
        ws2 = FakeWebSocket()
        await manager.connect(ws1, "all-1")
        await manager.connect(ws2, "all-2")

        await manager.disconnect_all(code=1001, reason="shutdown")
        assert ws1.closed is True
        assert ws2.closed is True
        assert ws1.close_code == 1001
        assert manager.active_connections_count == 0
        assert len(manager.connection_meta) == 0
