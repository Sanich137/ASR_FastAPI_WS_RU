"""
Тесты для services/ws_session.py (AudioSession).
"""

import time

import numpy as np
import pytest

from services.ws_session import AudioSession, SessionState
from models.ws_models import WSConfigMessage


@pytest.fixture
def session():
    """Фикстура: свежая сессия с малым лимитом буфера для тестов."""
    return AudioSession(client_id="test-client-001", max_buffer_duration_sec=2.0)


class TestAudioSessionLifecycle:
    def test_initial_state(self, session):
        """Проверяет начальное состояние сессии после создания."""
        assert session.state == SessionState.connecting
        assert session.config is None
        assert session.client_id == "test-client-001"
        assert len(session.buffer) == 0

    @pytest.mark.asyncio
    async def test_add_audio_success(self, session):
        """Успешное добавление аудио и переход в состояние receiving."""
        # 1 секунда аудио при 16 кГц = 16000 сэмплов
        frame = np.zeros(16000, dtype=np.float32)
        result = await session.add_audio(frame)
        assert result is True
        assert session.state == SessionState.receiving
        assert len(session.buffer) == 1

    @pytest.mark.asyncio
    async def test_add_audio_overflow(self, session):
        """Отказ при превышении max_buffer_duration_sec."""
        # Лимит 2 секунды. Первый фрейм 1.2 сек — ок. Второй 1.2 сек — переполнение.
        frame = np.zeros(int(1.2 * 16000), dtype=np.float32)
        assert await session.add_audio(frame) is True
        assert await session.add_audio(frame) is False
        assert len(session.buffer) == 1

    @pytest.mark.asyncio
    async def test_add_audio_with_config(self, session):
        """Проверка учёта sample_rate из конфига при расчёте длительности."""
        session.config = WSConfigMessage(sample_rate=8000)
        frame = np.zeros(8000, dtype=np.float32)  # 1 сек при 8 кГц
        assert await session.add_audio(frame) is True
        assert session.current_buffer_duration_sec == pytest.approx(1.0, 0.01)

    @pytest.mark.asyncio
    async def test_get_full_audio(self, session):
        """Конкатенация нескольких фреймов в единый массив."""
        frame1 = np.ones(8000, dtype=np.float32)
        frame2 = np.ones(8000, dtype=np.float32) * 2
        await session.add_audio(frame1)
        await session.add_audio(frame2)
        full = await session.get_full_audio()
        assert len(full) == 16000
        assert full[0] == 1.0
        assert full[-1] == 2.0

    @pytest.mark.asyncio
    async def test_get_full_audio_empty_buffer(self, session):
        """Пустой буфер возвращает пустой numpy-массив."""
        full = await session.get_full_audio()
        assert len(full) == 0
        assert full.dtype == np.float32

    @pytest.mark.asyncio
    async def test_reset(self, session):
        """Сброс сессии в начальное состояние."""
        session.config = WSConfigMessage(sample_rate=16000)
        await session.add_audio(np.zeros(16000, dtype=np.float32))
        session.state = SessionState.processing

        await session.reset()
        assert session.state == SessionState.connecting
        assert session.config is None
        assert len(session.buffer) == 0
        assert session.current_buffer_duration_sec == 0.0

    def test_is_expired_true(self, session):
        """Таймаут истёк."""
        session.last_activity = time.time() - 100
        assert session.is_expired(timeout_sec=10) is True

    def test_is_expired_false(self, session):
        """Таймаут не истёк."""
        session.last_activity = time.time()
        assert session.is_expired(timeout_sec=10) is False


class TestAudioSessionBufferDuration:
    @pytest.mark.asyncio
    async def test_duration_calculation_with_different_sample_rates(self):
        """Суммарная длительность при стандартном sample_rate."""
        sess = AudioSession(client_id="sr-test", max_buffer_duration_sec=5.0)
        sess.config = WSConfigMessage(sample_rate=16000)
        await sess.add_audio(np.zeros(8000, dtype=np.float32))  # 0.5 сек
        await sess.add_audio(np.zeros(8000, dtype=np.float32))  # ещё 0.5 сек
        assert sess.current_buffer_duration_sec == pytest.approx(1.0, 0.01)

    @pytest.mark.asyncio
    async def test_duration_respects_config_sample_rate(self):
        """Суммарная длительность пересчитывается при sample_rate=8000."""
        sess = AudioSession(client_id="sr-test-8k", max_buffer_duration_sec=5.0)
        sess.config = WSConfigMessage(sample_rate=8000)
        await sess.add_audio(np.zeros(8000, dtype=np.float32))  # 1 сек при 8кГц
        assert sess.current_buffer_duration_sec == pytest.approx(1.0, 0.01)


class TestAudioSessionSerialization:
    def test_to_dict_and_from_dict(self):
        """Сериализация и восстановление мета-полей сессии."""
        sess = AudioSession(client_id="ser-test")
        sess.config = WSConfigMessage(sample_rate=8000, do_dialogue=True, do_punctuation=True)
        sess.channel_name = "ch-1"
        sess.audio_duration = 5.0
        sess.ws_collected_asr_res = {"channel_1": [{"text": "привет"}]}
        sess.do_dialogue = True
        sess.do_punctuation = True

        d = sess.to_dict()
        assert d["client_id"] == "ser-test"
        assert d["channel_name"] == "ch-1"
        assert d["audio_duration"] == 5.0
        assert d["do_dialogue"] is True
        assert d["do_punctuation"] is True

        restored = AudioSession.from_dict(d)
        assert restored.client_id == "ser-test"
        assert restored.channel_name == "ch-1"
        assert restored.audio_duration == 5.0
        assert restored.do_dialogue is True
        assert restored.config.sample_rate == 8000
        assert restored.ws_collected_asr_res == {"channel_1": [{"text": "привет"}]}
