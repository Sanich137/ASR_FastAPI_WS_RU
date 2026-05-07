"""
Тесты для find_last_speech_position_v2 (utils/chunk_doing.py).
Проверяет корректность VAD-разделения с фиктивными AudioSegment.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pydub import AudioSegment

from utils.chunk_doing import find_last_speech_position_v2
from services.ws_session import AudioSession


@pytest.fixture
def session() -> AudioSession:
    sess = AudioSession(client_id="vad-test", max_buffer_duration_sec=300.0)
    sess.audio_buffer = AudioSegment.silent(5000, frame_rate=16000)  # 5 сек
    sess.audio_overlap = AudioSegment.silent(1, frame_rate=16000)
    sess.audio_to_asr = []
    return sess


class TestFindLastSpeechPositionV2:
    @pytest.mark.asyncio
    async def test_last_chunk_splits_to_asr(self, session: AudioSession) -> None:
        """При is_last_chunk=True весь буфер уходит в audio_to_asr по частям."""
        with patch("utils.chunk_doing.vad", MagicMock()):
            await find_last_speech_position_v2(session, is_last_chunk=True)
            assert len(session.audio_to_asr) > 0
            # Проверяем, что суммарная длительность равна исходной
            total = sum(seg.duration_seconds for seg in session.audio_to_asr)
            assert total == pytest.approx(5.0, 0.1)

    @pytest.mark.asyncio
    async def test_non_last_chunk_clears_buffer(self, session: AudioSession) -> None:
        """При is_last_chunk=False буфер очищается, overlap получает хвост."""
        with patch("utils.chunk_doing.vad", MagicMock()) as mock_vad:
            mock_vad.reset_state = AsyncMock()
            mock_vad.state = None
            mock_vad.is_speech = AsyncMock(return_value=(0.99, None))  # всегда речь
            mock_vad.prob_level = 0.5

            await find_last_speech_position_v2(session, is_last_chunk=False)
            # Если весь сегмент — речь, всё уходит в audio_to_asr
            assert len(session.audio_to_asr) == 1
            assert session.audio_buffer.duration_seconds < 0.1  # silent(1ms)
            assert session.audio_overlap.duration_seconds == pytest.approx(0.0, 0.1)
