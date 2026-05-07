"""
Модуль services/ws_session.py
Содержит класс AudioSession для управления состоянием и буфером аудио
в рамках одной WebSocket-сессии распознавания речи (ASR).
"""

import time
from collections import deque
from enum import Enum
from typing import Optional

import numpy as np
from pydub import AudioSegment

from models.ws_models import WSConfigMessage
from config import settings


class SessionState(str, Enum):
    """Состояния жизненного цикла аудио-сессии."""
    connecting = "connecting"
    receiving = "receiving"
    processing = "processing"
    completed = "completed"
    error = "error"


class AudioSession:
    """
    Управляет буфером аудио и состоянием для одного WebSocket-клиента.

    Attributes:
        client_id: Уникальный идентификатор сессии.
        state: Текущее состояние сессии (SessionState).
        buffer: Очередь (deque) numpy-массивов с фрагментами аудио.
        config: Конфигурация клиента (WSConfigMessage) или None.
        max_buffer_duration_sec: Максимальная суммарная длительность буфера в секундах.
        last_activity: Unix-timestamp последней активности (добавления аудио или конфига).
    """

    def __init__(
        self,
        client_id: str,
        max_buffer_duration_sec: Optional[float] = None,
    ) -> None:
        """
        Инициализирует новую аудио-сессию.

        Args:
            client_id: UUID или строковый идентификатор соединения.
            max_buffer_duration_sec: Максимальная длительность буфера (сек).
                По умолчанию берётся из settings.WS_MAX_BUFFER_DURATION_SEC.
        """
        self.client_id: str = client_id
        self.state: SessionState = SessionState.connecting
        self.buffer: deque[np.ndarray] = deque()
        self.config: Optional[WSConfigMessage] = None
        self.max_buffer_duration_sec: float = (
            max_buffer_duration_sec
            if max_buffer_duration_sec is not None
            else getattr(settings, "WS_MAX_BUFFER_DURATION_SEC", 300.0)
        )
        self.last_activity: float = time.time()

        # --- Поля для ASR pipeline (Задача 6.3) ---
        self.audio_buffer: AudioSegment = AudioSegment.silent(
            1, frame_rate=settings.BASE_SAMPLE_RATE
        )
        self.audio_overlap: AudioSegment = AudioSegment.silent(
            1, frame_rate=settings.BASE_SAMPLE_RATE
        )
        self.audio_to_asr: list[AudioSegment] = []
        self.audio_duration: float = 0.0
        self.ws_collected_asr_res: dict = {f"channel_{1}": []}
        self.wait_null_answers: bool = True
        self.do_dialogue: bool = False
        self.do_punctuation: bool = False
        self.channel_name: str = "Null"

    @property
    def current_buffer_duration_sec(self) -> float:
        """
        Вычисляет суммарную длительность аудио в буфере (в секундах).

        Использует sample_rate из config (по умолчанию 16000 Гц),
        считая, что каждый элемент буфера — одномерный массив сэмплов.
        """
        sample_rate = (
            self.config.sample_rate
            if self.config is not None
            else 16000
        )
        if sample_rate <= 0:
            sample_rate = 16000
        total_samples = sum(len(chunk) for chunk in self.buffer)
        return total_samples / sample_rate

    async def add_audio(self, frame: np.ndarray) -> bool:
        """
        Добавляет фрагмент аудио в буфер, если не превышен лимит длительности.

        Args:
            frame: Одномерный numpy-массив аудио-сэмплов (float32 или int16).

        Returns:
            True — фрагмент успешно добавлен.
            False — буфер переполнен, фрагмент отклонён.
        """
        self.last_activity = time.time()

        if self.state == SessionState.connecting:
            self.state = SessionState.receiving

        # Проверка на переполнение: оцениваем длительность после добавления
        sample_rate = (
            self.config.sample_rate
            if self.config is not None
            else 16000
        )
        if sample_rate <= 0:
            sample_rate = 16000

        incoming_duration = len(frame) / sample_rate
        if self.current_buffer_duration_sec + incoming_duration > self.max_buffer_duration_sec:
            return False

        self.buffer.append(frame)
        return True

    async def get_full_audio(self) -> np.ndarray:
        """
        Конкатенирует все фрагменты буфера в единый numpy-массив.

        Returns:
            Объединённый массив сэмплов. Если буфер пуст — возвращается пустой массив.
        """
        if not self.buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(list(self.buffer))

    async def reset(self) -> None:
        """
        Очищает буфер, сбрасывает конфигурацию и переводит сессию
        в начальное состояние (connecting).
        """
        self.buffer.clear()
        self.config = None
        self.state = SessionState.connecting
        self.last_activity = time.time()

        # Сброс ASR pipeline state
        self.audio_buffer = AudioSegment.silent(1, frame_rate=settings.BASE_SAMPLE_RATE)
        self.audio_overlap = AudioSegment.silent(1, frame_rate=settings.BASE_SAMPLE_RATE)
        self.audio_to_asr = []
        self.audio_duration = 0.0
        self.ws_collected_asr_res = {f"channel_{1}": []}
        self.wait_null_answers = True
        self.do_dialogue = False
        self.do_punctuation = False
        self.channel_name = "Null"

    def is_expired(self, timeout_sec: float) -> bool:
        """
        Проверяет, истёк ли таймаут неактивности сессии.

        Args:
            timeout_sec: Допустимое время простоя в секундах.

        Returns:
            True, если с момента last_activity прошло больше timeout_sec.
        """
        return (time.time() - self.last_activity) > timeout_sec

    def to_dict(self) -> dict:
        """
        Сериализует лёгкие (не AudioSegment) поля сессии в dict для StateStore.

        Returns:
            dict с client_id, config, channel_name, audio_duration, флагами
            и накопленными результатами распознавания.
        """
        return {
            "client_id": self.client_id,
            "config": self.config.model_dump() if self.config else None,
            "channel_name": self.channel_name,
            "audio_duration": self.audio_duration,
            "do_dialogue": self.do_dialogue,
            "do_punctuation": self.do_punctuation,
            "wait_null_answers": self.wait_null_answers,
            "ws_collected_asr_res": self.ws_collected_asr_res,
            "state": self.state.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AudioSession":
        """
        Восстанавливает сессию из dict (только мета-поля, без AudioSegment-буферов).

        Args:
            data: dict, полученный из to_dict().

        Returns:
            AudioSession с восстановленной конфигурацией и флагами.
        """
        session = cls(client_id=data["client_id"])
        if data.get("config"):
            session.config = WSConfigMessage(**data["config"])
        session.channel_name = data.get("channel_name", "Null")
        session.audio_duration = data.get("audio_duration", 0.0)
        session.do_dialogue = data.get("do_dialogue", False)
        session.do_punctuation = data.get("do_punctuation", False)
        session.wait_null_answers = data.get("wait_null_answers", True)
        session.ws_collected_asr_res = data.get("ws_collected_asr_res", {f"channel_{1}": []})
        session.state = SessionState(data.get("state", "connecting"))
        return session
