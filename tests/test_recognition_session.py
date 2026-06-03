"""
Тесты для services/recognition_session.py.
"""

import io
import os

import pytest
from unittest.mock import MagicMock, AsyncMock

from services.recognition_session import RecognitionSession, FileRecognitionSession, SessionState
from models.ws_models import WSConfigMessage


class TestRecognitionSession:
    def test_init_generates_uuid(self):
        """Проверяет автоматическую генерацию session_id."""
        sess = RecognitionSession()
        assert len(sess.session_id) == 36
        assert sess.state == SessionState.created

    def test_audio_buffers_are_silent_segment(self):
        """Проверяет, что буферы инициализируются как AudioSegment.silent."""
        sess = RecognitionSession()
        assert sess.audio_buffer.duration_seconds < 0.1
        assert sess.audio_overlap.duration_seconds < 0.1
        assert sess.audio_to_asr == []

    def test_reset_clears_buffers(self):
        """Проверяет очистку буферов и сброс состояния."""
        sess = RecognitionSession()
        sess.audio_to_asr = [MagicMock()]
        sess.audio_duration = 10.0
        sess.reset()
        assert sess.audio_to_asr == []
        assert sess.audio_duration == 0.0
        assert sess.state == SessionState.created

    def test_to_dict_excludes_audio_segment(self):
        """Проверяет, что to_dict сериализует только мета-поля."""
        sess = RecognitionSession()
        sess.config = WSConfigMessage(sample_rate=16000, do_dialogue=True)
        sess.do_dialogue = True
        sess.audio_duration = 5.0
        d = sess.to_dict()
        assert d["session_id"] == sess.session_id
        assert d["audio_duration"] == 5.0
        assert d["do_dialogue"] is True
        assert "audio_buffer" not in d

    def test_from_dict_restores_meta(self):
        """Проверяет восстановление сессии из dict."""
        sess = RecognitionSession()
        sess.config = WSConfigMessage(sample_rate=8000, do_punctuation=True)
        sess.channel_name = "ch-1"
        d = sess.to_dict()
        restored = RecognitionSession.from_dict(d)
        assert restored.channel_name == "ch-1"
        assert restored.config.sample_rate == 8000
        assert restored.state == SessionState.created


class TestFileRecognitionSession:
    def test_init_with_post_id(self):
        """Проверяет инициализацию с кастомным post_id и params."""
        sess = FileRecognitionSession(post_id="post-123", params={"foo": "bar"})
        assert sess.post_id == "post-123"
        assert sess.params == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_save_upload(self):
        """Проверяет сохранение UploadFile в BytesIO."""
        sess = FileRecognitionSession()
        mock_file = MagicMock()
        mock_file.read = AsyncMock(return_value=b"audio_data")
        await sess.save_upload(mock_file)
        assert sess.file_buffer is not None
        assert sess.file_buffer.read() == b"audio_data"

    def test_cleanup_closes_buffer(self):
        """Проверяет закрытие BytesIO при cleanup."""
        sess = FileRecognitionSession()
        sess.file_buffer = io.BytesIO(b"data")
        sess.tmp_path = None
        sess.cleanup()
        assert sess.file_buffer is None

    def test_cleanup_removes_tmp_path(self, tmp_path):
        """Проверяет удаление временного файла с диска."""
        fake_file = tmp_path / "test.wav"
        fake_file.write_text("fake audio")
        sess = FileRecognitionSession()
        sess.tmp_path = str(fake_file)
        sess.cleanup()
        assert not os.path.exists(str(fake_file))
        assert sess.tmp_path is None
