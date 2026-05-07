"""
Тесты для services/asr_pipeline.py (потоковое распознавание с overlap).
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pydub import AudioSegment

from services.asr_pipeline import process_audio_stream_chunk, process_final_audio
from services.ws_session import AudioSession, SessionState
from services.ws_manager import ConnectionManager
from models.ws_models import WSConfigMessage, WSResultMessage, WSRecognitionData
from config import settings


class FakeManager(ConnectionManager):
    """Фейковый менеджер для тестов ASR pipeline."""

    def __init__(self):
        super().__init__(max_connections=10)
        self.sent_messages: list[tuple[str, object]] = []

    async def send_message(self, client_id: str, message: object) -> None:
        self.sent_messages.append((client_id, message))


def _make_pcm16_silence(duration_sec: float, sample_rate: int = 16000) -> bytes:
    """Генерирует PCM16 тишину заданной длительности."""
    samples = int(duration_sec * sample_rate)
    return b"\x00\x00" * samples


@pytest.fixture
def session() -> AudioSession:
    """Фикстура: AudioSession с типичной конфигурацией ASR."""
    sess = AudioSession(client_id="test-asr-42", max_buffer_duration_sec=300.0)
    sess.config = WSConfigMessage(
        sample_rate=16000,
        audio_format="pcm16",
        wait_null_answers=True,
        do_dialogue=False,
        do_punctuation=False,
        channel_name="ch-test",
    )
    sess.state = SessionState.receiving
    return sess


@pytest.fixture
def manager() -> FakeManager:
    return FakeManager()


@pytest.fixture
def recognizer():
    return MagicMock()


@pytest.fixture
def punctuator():
    return MagicMock()


class TestProcessAudioStreamChunk:
    @pytest.mark.asyncio
    async def test_small_chunk_accumulates_no_vad(self, session, manager, recognizer, punctuator):
        """Малый чанк (< MAX_OVERLAP_DURATION) накапливается, VAD не вызывается."""
        chunk = _make_pcm16_silence(1.0)  # 1 сек < MAX_OVERLAP_DURATION (обычно 30)
        with patch("services.asr_pipeline.find_last_speech_position_v2", new_callable=AsyncMock) as mock_vad:
            with patch("services.asr_pipeline.simple_recognise", new_callable=AsyncMock) as mock_asr:
                await process_audio_stream_chunk(session, chunk, recognizer, punctuator, manager)
                mock_vad.assert_not_awaited()
                mock_asr.assert_not_awaited()
                assert session.audio_buffer.duration_seconds >= 1.0

    @pytest.mark.asyncio
    async def test_large_chunk_triggers_vad_and_sends_result(self, session, manager, recognizer, punctuator):
        """Большой чанк вызывает VAD, распознавание и отправку результата."""
        # Устанавливаем буфер так, чтобы combined_duration >= MAX_OVERLAP_DURATION
        session.audio_buffer = AudioSegment.silent(
            int((settings.MAX_OVERLAP_DURATION - 0.2) * 1000), frame_rate=16000
        )
        chunk = _make_pcm16_silence(0.5)

        async def mock_vad(session, is_last_chunk):
            session.audio_to_asr.append(session.audio_buffer)
            session.audio_overlap = AudioSegment.silent(1, frame_rate=16000)
            session.audio_buffer = AudioSegment.silent(1, frame_rate=16000)

        with patch("services.asr_pipeline.find_last_speech_position_v2", mock_vad):
            with patch("services.asr_pipeline.simple_recognise", new_callable=AsyncMock) as mock_asr:
                mock_asr.return_value = {
                    "tokens": ["п", "р", "и", "в", "е", "т"],
                    "timestamps": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    "text": "привет",
                }
                await process_audio_stream_chunk(session, chunk, recognizer, punctuator, manager)

        assert len(manager.sent_messages) == 1
        client_id, msg = manager.sent_messages[0]
        assert client_id == session.client_id
        assert isinstance(msg, WSResultMessage)
        assert msg.silence is False
        assert msg.data.text == "привет"
        assert msg.last_message is False

    @pytest.mark.asyncio
    async def test_silence_partial_when_empty_text(self, session, manager, recognizer, punctuator):
        """При пустом тексте и wait_null_answers=True отправляется silence partial."""
        session.audio_buffer = AudioSegment.silent(
            int((settings.MAX_OVERLAP_DURATION - 0.05) * 1000), frame_rate=16000
        )
        chunk = _make_pcm16_silence(0.1)

        async def mock_vad(session, is_last_chunk):
            session.audio_to_asr.append(session.audio_buffer)
            session.audio_overlap = AudioSegment.silent(1, frame_rate=16000)
            session.audio_buffer = AudioSegment.silent(1, frame_rate=16000)

        with patch("services.asr_pipeline.find_last_speech_position_v2", mock_vad):
            with patch("services.asr_pipeline.simple_recognise", new_callable=AsyncMock) as mock_asr:
                mock_asr.return_value = {
                    "tokens": [],
                    "timestamps": [],
                    "text": "",
                }
                await process_audio_stream_chunk(session, chunk, recognizer, punctuator, manager)

        assert len(manager.sent_messages) == 1
        _, msg = manager.sent_messages[0]
        assert isinstance(msg, WSResultMessage)
        assert msg.silence is True
        assert msg.data.text == ""

    @pytest.mark.asyncio
    async def test_odd_bytes_padded(self, session, manager, recognizer, punctuator):
        """Нечётное количество байтов дополняется до чётного."""
        chunk = b"\x01\x02\x03"  # 3 байта
        with patch("services.asr_pipeline.find_last_speech_position_v2", new_callable=AsyncMock):
            with patch("services.asr_pipeline.simple_recognise", new_callable=AsyncMock):
                await process_audio_stream_chunk(session, chunk, recognizer, punctuator, manager)
                assert session.audio_buffer.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_error_sends_ws_error_message(self, session, manager, recognizer, punctuator):
        """При исключении внутри pipeline отправляется WSErrorMessage."""
        chunk = _make_pcm16_silence(1.0)
        session.audio_buffer = AudioSegment.silent(
            int((settings.MAX_OVERLAP_DURATION - 0.2) * 1000), frame_rate=16000
        )
        with patch("services.asr_pipeline.find_last_speech_position_v2", side_effect=ValueError("boom")):
            await process_audio_stream_chunk(session, chunk, recognizer, punctuator, manager)

        assert len(manager.sent_messages) == 1
        _, msg = manager.sent_messages[0]
        assert msg.type == "error"
        assert msg.code == "vad_error"


class TestProcessFinalAudio:
    @pytest.mark.asyncio
    async def test_final_under_2sec_padded(self, session, manager, recognizer, punctuator):
        """Финальный аудио < 2 сек дополняется тишиной и распознаётся."""
        session.audio_overlap = AudioSegment.silent(500, frame_rate=16000)  # 0.5 сек
        session.audio_buffer = AudioSegment.silent(500, frame_rate=16000)  # 0.5 сек

        with patch("services.asr_pipeline.simple_recognise", new_callable=AsyncMock) as mock_asr:
            mock_asr.return_value = {
                "tokens": ["т", "е", "с", "т"],
                "timestamps": [0.0, 0.1, 0.2, 0.3],
                "text": "тест",
            }
            await process_final_audio(session, recognizer, punctuator, manager)

        assert len(manager.sent_messages) == 1
        _, msg = manager.sent_messages[0]
        assert isinstance(msg, WSResultMessage)
        assert msg.last_message is True
        assert msg.silence is False
        assert msg.data.text == "тест"

    @pytest.mark.asyncio
    async def test_final_empty_silence(self, session, manager, recognizer, punctuator):
        """Финальный результат пустой — отправляется silence=True."""
        session.audio_overlap = AudioSegment.silent(1000, frame_rate=16000)
        session.audio_buffer = AudioSegment.silent(1000, frame_rate=16000)

        with patch("services.asr_pipeline.simple_recognise", new_callable=AsyncMock) as mock_asr:
            mock_asr.return_value = {
                "tokens": [],
                "timestamps": [],
                "text": "",
            }
            await process_final_audio(session, recognizer, punctuator, manager)

        assert len(manager.sent_messages) == 1
        _, msg = manager.sent_messages[0]
        assert msg.silence is True
        assert msg.last_message is True

    @pytest.mark.asyncio
    async def test_final_with_dialogue(self, session, manager, recognizer, punctuator):
        """При do_dialogue=True вызывается do_sensitizing и sentenced_data заполняется."""
        session.do_dialogue = True
        session.do_punctuation = True
        session.audio_overlap = AudioSegment.silent(2000, frame_rate=16000)
        session.audio_buffer = AudioSegment.silent(1000, frame_rate=16000)

        with patch("services.asr_pipeline.simple_recognise", new_callable=AsyncMock) as mock_asr:
            mock_asr.return_value = {
                "tokens": ["п", "р", "и", "в", "е", "т"],
                "timestamps": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "text": "привет",
            }
            with patch("services.asr_pipeline.do_sensitizing", new_callable=AsyncMock) as mock_sent:
                mock_sent.return_value = {"raw_text_sentenced_recognition": "Ch1: Привет."}
                await process_final_audio(session, recognizer, punctuator, manager)

        assert len(manager.sent_messages) == 1
        _, msg = manager.sent_messages[0]
        assert msg.sentenced_data == {"raw_text_sentenced_recognition": "Ch1: Привет."}

    @pytest.mark.asyncio
    async def test_final_error_sends_ws_error(self, session, manager, recognizer, punctuator):
        """При исключении на финальном этапе отправляется WSErrorMessage."""
        with patch("services.asr_pipeline.simple_recognise", side_effect=RuntimeError("boom")):
            await process_final_audio(session, recognizer, punctuator, manager)

        assert len(manager.sent_messages) == 1
        _, msg = manager.sent_messages[0]
        assert msg.type == "error"
        assert msg.code == "asr_pipeline_final_error"
        assert session.state == SessionState.error
