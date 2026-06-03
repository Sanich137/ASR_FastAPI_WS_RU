"""
Тесты для services/ws_handler.py (MessageRouter и хендлеры).
"""

import base64

import numpy as np
import pytest

from models.ws_models import (
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
from services.ws_handler import (
    MessageRouter,
    handle_config,
    handle_audio,
    handle_status_request,
    handle_ping,
)


class FakeManager:
    """
    Фейковый менеджер соединений для тестирования хендлеров.

    Имитирует WSManagerProtocol без реальных WebSocket-объектов.
    """

    def __init__(self, max_connections: int = 100) -> None:
        self.sent_messages: list[tuple[str, object]] = []
        self._active_connections = 0
        self._max_connections = max_connections

    async def send_message(self, client_id: str, message: object) -> None:
        """Сохраняет сообщение во внутренний список для проверки в тестах."""
        self.sent_messages.append((client_id, message))

    @property
    def active_connections_count(self) -> int:
        return self._active_connections

    @property
    def max_connections(self) -> int:
        return self._max_connections


@pytest.fixture
def session() -> AudioSession:
    """Фикстура: свежая AudioSession с лимитом буфера 5 секунд."""
    return AudioSession(client_id="test-client", max_buffer_duration_sec=5.0)


@pytest.fixture
def manager() -> FakeManager:
    """Фикстура: фейковый менеджер соединений."""
    return FakeManager()


@pytest.fixture
def metrics() -> SystemMetricsCollector:
    """Фикстура: коллектор метрик."""
    return SystemMetricsCollector()


class TestHandleConfig:
    @pytest.mark.asyncio
    async def test_sets_config_and_state(self, session: AudioSession, manager: FakeManager) -> None:
        """Проверяет, что handle_config устанавливает конфиг и переводит сессию в receiving."""
        msg = WSConfigMessage(sample_rate=8000, audio_transport="binary")
        await handle_config(msg, session, manager)
        assert session.config == msg
        assert session.state == SessionState.receiving


class TestHandleAudio:
    @pytest.mark.asyncio
    async def test_decodes_and_adds_audio(self, session: AudioSession, manager: FakeManager) -> None:
        """Проверяет успешное декодирование base64 и добавление в буфер."""
        raw = np.zeros(16000, dtype=np.int16).tobytes()
        b64 = base64.b64encode(raw).decode()
        msg = WSAudioMessage(audio_base64=b64, seq_num=0)
        await handle_audio(msg, session, manager)
        assert len(session.buffer) == 1
        assert manager.sent_messages == []

    @pytest.mark.asyncio
    async def test_buffer_overflow_sends_error(self, session: AudioSession, manager: FakeManager) -> None:
        """Проверяет, что при переполнении буфера отправляется WSErrorMessage."""
        raw = np.zeros(96000, dtype=np.int16).tobytes()  # 6 сек при 16 кГц, лимит 5
        b64 = base64.b64encode(raw).decode()
        msg = WSAudioMessage(audio_base64=b64, seq_num=0)
        await handle_audio(msg, session, manager)
        assert len(manager.sent_messages) == 1
        client_id, error = manager.sent_messages[0]
        assert client_id == session.client_id
        assert isinstance(error, WSErrorMessage)
        assert error.code == "buffer_overflow"

    @pytest.mark.asyncio
    async def test_invalid_base64_sends_error(self, session: AudioSession, manager: FakeManager) -> None:
        """Проверяет обработку некорректного base64."""
        msg = WSAudioMessage(audio_base64="!!!invalid!!!", seq_num=0)
        await handle_audio(msg, session, manager)
        assert len(manager.sent_messages) == 1
        _, error = manager.sent_messages[0]
        assert isinstance(error, WSErrorMessage)
        assert error.code == "decode_error"


class TestHandleStatusRequest:
    @pytest.mark.asyncio
    async def test_sends_status_response(
        self,
        session: AudioSession,
        manager: FakeManager,
        metrics: SystemMetricsCollector,
    ) -> None:
        """Проверяет, что handle_status_request отправляет WSStatusResponse."""
        msg = WSStatusRequest()
        await handle_status_request(msg, session, manager, metrics)
        assert len(manager.sent_messages) == 1
        _, response = manager.sent_messages[0]
        assert isinstance(response, WSStatusResponse)


class TestHandlePing:
    @pytest.mark.asyncio
    async def test_sends_pong(self, session: AudioSession, manager: FakeManager) -> None:
        """Проверяет, что handle_ping отправляет WSPongMessage."""
        msg = WSPingMessage()
        await handle_ping(msg, session, manager)
        assert len(manager.sent_messages) == 1
        _, response = manager.sent_messages[0]
        assert isinstance(response, WSPongMessage)


class TestMessageRouter:
    @pytest.mark.asyncio
    async def test_register_and_route_config(self, session: AudioSession, manager: FakeManager) -> None:
        """Проверяет регистрацию хендлера и маршрутизацию сообщения config."""
        router = MessageRouter()
        router.register_handler(WSMessageType.config, handle_config)
        msg = WSConfigMessage()
        await router.route(msg, session, manager)
        assert session.config is not None

    @pytest.mark.asyncio
    async def test_route_unknown_type_sends_error(self, session: AudioSession, manager: FakeManager) -> None:
        """Проверяет, что незарегистрированный тип сообщения вызывает ошибку."""
        router = MessageRouter()
        msg = WSPingMessage()  # ping не зарегистрирован
        await router.route(msg, session, manager)
        assert len(manager.sent_messages) == 1
        _, error = manager.sent_messages[0]
        assert isinstance(error, WSErrorMessage)
        assert error.code == "unsupported_type"

    @pytest.mark.asyncio
    async def test_route_status_request_without_collector(
        self,
        session: AudioSession,
        manager: FakeManager,
    ) -> None:
        """Проверяет, что status_request без metrics_collector вызывает handler_error."""
        router = MessageRouter()
        router.register_handler(WSMessageType.status_request, handle_status_request)
        msg = WSStatusRequest()
        await router.route(msg, session, manager, metrics_collector=None)
        assert len(manager.sent_messages) == 1
        _, error = manager.sent_messages[0]
        assert isinstance(error, WSErrorMessage)
        assert error.code == "handler_error"

    @pytest.mark.asyncio
    async def test_route_with_collector(
        self,
        session: AudioSession,
        manager: FakeManager,
        metrics: SystemMetricsCollector,
    ) -> None:
        """Проверяет корректную маршрутизацию status_request с metrics_collector."""
        router = MessageRouter()
        router.register_handler(WSMessageType.status_request, handle_status_request)
        msg = WSStatusRequest()
        await router.route(msg, session, manager, metrics_collector=metrics)
        assert len(manager.sent_messages) == 1
        _, response = manager.sent_messages[0]
        assert isinstance(response, WSStatusResponse)

    @pytest.mark.asyncio
    async def test_handler_exception_caught(self, session: AudioSession, manager: FakeManager) -> None:
        """Проверяет, что исключение в хендлере перехватывается и отправляется ошибка."""
        async def bad_handler(msg, sess, mgr):
            raise ValueError("boom")

        router = MessageRouter()
        router.register_handler(WSMessageType.ping, bad_handler)
        msg = WSPingMessage()
        await router.route(msg, session, manager)
        assert len(manager.sent_messages) == 1
        _, error = manager.sent_messages[0]
        assert isinstance(error, WSErrorMessage)
        assert error.code == "handler_error"
