"""
Модуль services/recognition_session.py
Базовый класс RecognitionSession и специализации для WebSocket и HTTP-файлового распознавания.
"""

import os
import uuid
from enum import Enum
from typing import Optional, Any
from io import BytesIO

from pydub import AudioSegment
from fastapi import UploadFile

from config import settings
from models.ws_models import WSConfigMessage


class SessionState(str, Enum):
    """Состояния жизненного цикла сессии распознавания."""
    created = "created"
    connecting = "connecting"
    receiving = "receiving"
    processing = "processing"
    completed = "completed"
    error = "error"


class RecognitionSession:
    """
    Базовая сессия распознавания речи.
    Содержит общие поля для потокового (WS) и файлового (HTTP) распознавания:
    AudioSegment-буферы, накопление результатов, конфигурация.
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        """
        Инициализирует базовую сессию.

        Args:
            session_id: UUID сессии. Если None — генерируется автоматически.
        """
        self.session_id: str = session_id or str(uuid.uuid4())
        self.state: SessionState = SessionState.created

        # AudioSegment-буферы (общие для WS и файлового режима)
        self.audio_buffer: AudioSegment = AudioSegment.silent(
            1, frame_rate=settings.BASE_SAMPLE_RATE
        )
        self.audio_overlap: AudioSegment = AudioSegment.silent(
            1, frame_rate=settings.BASE_SAMPLE_RATE
        )
        self.audio_to_asr: list[AudioSegment] = []
        self.audio_duration: float = 0.0

        # Результаты и мета
        self.collected_asr_res: dict = {f"channel_{1}": []}
        self.channel_name: str = "Null"
        self.do_dialogue: bool = False
        self.do_punctuation: bool = False
        self.config: Optional[WSConfigMessage] = None

    @property
    def client_id(self) -> str:
        """Возвращает session_id как client_id для совместимости с WS-обработчиками и VAD."""
        return self.session_id

    async def reset(self) -> None:
        """
        Очищает AudioSegment-буферы, результаты и сбрасывает состояние.
        """
        self.audio_buffer = AudioSegment.silent(1, frame_rate=settings.BASE_SAMPLE_RATE)
        self.audio_overlap = AudioSegment.silent(1, frame_rate=settings.BASE_SAMPLE_RATE)
        self.audio_to_asr = []
        self.audio_duration = 0.0
        self.collected_asr_res = {f"channel_{1}": []}
        self.state = SessionState.created

    def to_dict(self) -> dict:
        """
        Сериализует мета-поля в dict для StateStore.
        AudioSegment-объекты не сериализуются.
        """
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "audio_duration": self.audio_duration,
            "channel_name": self.channel_name,
            "do_dialogue": self.do_dialogue,
            "do_punctuation": self.do_punctuation,
            "collected_asr_res": self.collected_asr_res,
            "config": self.config.model_dump() if self.config else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RecognitionSession":
        """
        Восстанавливает сессию из dict (только мета-поля, без AudioSegment).
        """
        session = cls(session_id=data.get("session_id"))
        session.state = SessionState(data.get("state", "created"))
        session.audio_duration = data.get("audio_duration", 0.0)
        session.channel_name = data.get("channel_name", "Null")
        session.do_dialogue = data.get("do_dialogue", False)
        session.do_punctuation = data.get("do_punctuation", False)
        session.collected_asr_res = data.get("collected_asr_res", {f"channel_{1}": []})
        if data.get("config"):
            session.config = WSConfigMessage(**data["config"])
        return session


class FileRecognitionSession(RecognitionSession):
    """
    Сессия для файлового распознавания (HTTP Upload / URL).
    Расширяет RecognitionSession полями для работы с временными файлами.
    """

    def __init__(
        self,
        post_id: Optional[str] = None,
        params: Optional[Any] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Инициализирует файловую сессию.

        Args:
            post_id: Идентификатор поста/задачи.
            params: Параметры запроса (PostFileRequest или SyncASRRequest).
            session_id: UUID сессии.
        """
        super().__init__(session_id=session_id)
        self.post_id: str = post_id or str(uuid.uuid4())
        self.params: Optional[Any] = params
        self.tmp_path: Optional[str] = None
        self.file_buffer: Optional[BytesIO] = None

    async def save_upload(self, file: UploadFile) -> None:
        """
        Сохраняет загруженный файл во внутренний BytesIO.

        Args:
            file: Объект UploadFile из FastAPI.
        """
        self.file_buffer = BytesIO(await file.read())
        self.file_buffer.seek(0)

    def cleanup(self) -> None:
        """
        Очищает ресурсы: закрывает BytesIO, удаляет временный файл с диска.
        """
        if self.file_buffer is not None:
            self.file_buffer.close()
            self.file_buffer = None
        if self.tmp_path and isinstance(self.tmp_path, (str, os.PathLike)):
            try:
                os.remove(self.tmp_path)
            except OSError:
                pass
            self.tmp_path = None
        elif self.tmp_path is not None:
            # tmp_path содержит не строку/путь (например, BytesIO) — просто сбрасываем
            self.tmp_path = None
