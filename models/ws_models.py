import time
import base64
from enum import Enum
from typing import Literal, Union, Annotated

from pydantic import BaseModel, Field, TypeAdapter


class WSMessageType(str, Enum):
    config = "config"
    audio_chunk = "audio_chunk"
    ping = "ping"
    pong = "pong"
    status_request = "status_request"
    status_response = "status_response"
    error = "error"
    eos = "eos"
    partial_result = "partial_result"
    final_result = "final_result"


class WSBaseMessage(BaseModel):
    type: WSMessageType
    timestamp: float | None = Field(default_factory=time.time)


class WSConfigMessage(WSBaseMessage):
    type: Literal[WSMessageType.config] = WSMessageType.config
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    language: str = "ru"
    wait_null_answers: bool = True
    enable_diarization: bool = False
    num_speakers: int = Field(default=-1, ge=-1, le=10)
    enable_punctuation: bool = False
    do_dialogue: bool = False
    do_punctuation: bool = False
    audio_format: str = "pcm16"
    audio_transport: Literal["json_base64", "binary"] = Field(
        default="json_base64",
        description='Транспорт аудио: "json_base64" — аудио внутри JSON как base64-строка; '
                    '"binary" — сырые байты через WebSocket binary frames (receive_bytes()).'
    )
    channel_name: str | None = None


class WSAudioMessage(WSBaseMessage):
    type: Literal[WSMessageType.audio_chunk] = WSMessageType.audio_chunk
    audio_base64: str | None = None
    seq_num: int = Field(default=0, ge=0)


class WSPingMessage(WSBaseMessage):
    type: Literal[WSMessageType.ping] = WSMessageType.ping


class WSPongMessage(WSBaseMessage):
    type: Literal[WSMessageType.pong] = WSMessageType.pong


class WSStatusRequest(WSBaseMessage):
    type: Literal[WSMessageType.status_request] = WSMessageType.status_request
    command: str = "get_status"


class WSStatusResponse(WSBaseMessage):
    type: Literal[WSMessageType.status_response] = WSMessageType.status_response
    adapter_status: Literal["idle", "busy", "overloaded"] = "idle"
    gpu_memory_free_mb: int | None = None
    gpu_memory_total_mb: int | None = None
    gpu_utilization_percent: float | None = None
    cpu_memory_free_mb: int | None = None
    cpu_memory_total_mb: int | None = None
    cpu_utilization_percent: float | None = None
    active_tasks_count: int = 0
    active_connections_count: int = 0
    queue_depth: int = 0
    uptime_sec: float = 0.0
    uptime_formatted: str = "0s"
    temperature_celsius: float | None = None


class WSWordItem(BaseModel):
    conf: float
    start: float
    end: float
    word: str


class WSRecognitionData(BaseModel):
    result: list[WSWordItem] = Field(default_factory=list)
    text: str = ""


class WSResultMessage(WSBaseMessage):
    type: Literal[WSMessageType.partial_result, WSMessageType.final_result] = WSMessageType.partial_result
    channel_name: str = "Null"
    silence: bool = False
    data: WSRecognitionData
    error: str | None = None
    last_message: bool = False
    sentenced_data: dict | None = None


class WSPhraseResult(WSBaseMessage):
    type: Literal[WSMessageType.partial_result, WSMessageType.final_result] = WSMessageType.partial_result
    text: str
    speaker_id: str | None = None
    start_time: float
    end_time: float
    is_final: bool = False


class WSErrorMessage(WSBaseMessage):
    type: Literal[WSMessageType.error] = WSMessageType.error
    code: str = "unknown_error"
    message: str
    is_fatal: bool = False


class WSEosMessage(WSBaseMessage):
    type: Literal[WSMessageType.eos] = WSMessageType.eos


WSMessage = Annotated[
    Union[
        WSConfigMessage,
        WSAudioMessage,
        WSPingMessage,
        WSPongMessage,
        WSStatusRequest,
        WSStatusResponse,
        WSResultMessage,
        WSErrorMessage,
        WSEosMessage,
    ],
    Field(discriminator="type"),
]


def wrap_binary_audio(audio_bytes: bytes, seq_num: int = 0) -> WSAudioMessage:
    """Оборачивает raw bytes в WSAudioMessage (base64)."""
    return WSAudioMessage(
        audio_base64=base64.b64encode(audio_bytes).decode("utf-8"),
        seq_num=seq_num,
    )


# TypeAdapter для дискриминированного union — используется в роутах и тестах
ws_message_adapter = TypeAdapter(WSMessage)


def parse_ws_message(raw: str | bytes | dict) -> WSBaseMessage:
    """Валидирует входящее WS-сообщение из JSON-строки, bytes или dict."""
    if isinstance(raw, (str, bytes)):
        return ws_message_adapter.validate_json(raw)
    return ws_message_adapter.validate_python(raw)
