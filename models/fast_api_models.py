from pydantic import BaseModel, HttpUrl, Field, ConfigDict, RootModel
from typing import Union, Annotated, Optional, Any, List, Dict
from fastapi import UploadFile

from config import settings


class BaseResponse(BaseModel):
    """
    Базовая модель ответа API.
    """
    success: bool = True
    error_description: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    sentenced_data: Optional[Dict[str, Any]] = None
    diarized_data: Optional[Dict[str, Any]] = None


class V1BaseResponse(BaseModel):
    """
    Единый формат ответа для API v1.
    Все данные помещаются в поле data.
    """
    success: bool = True
    error_description: Optional[str] = None
    data: Any = {}


class RawData(RootModel[Dict[str, Any]]):
    """Структура сырых данных ASR. Словарь каналов, где каждый канал — список результатов."""


class SentencedData(BaseModel):
    """Структура разбитого на предложения ответа."""
    raw_text_sentenced_recognition: Optional[str] = None
    list_of_sentenced_recognitions: Optional[List[Dict[str, Any]]] = None
    full_text_only: Optional[List[str]] = None
    err_state: Optional[Any] = None


class DiarizedData(BaseModel):
    """Структура данных диаризации."""
    speakers: Optional[List[str]] = None
    segments: Optional[List[Dict[str, Any]]] = None


class ASRData(BaseModel):
    """Данные ответа ASR роутеров (post_by_url, post_by_file)."""
    raw_data: Optional[RawData] = None
    sentenced_data: Optional[SentencedData] = None
    diarized_data: Optional[DiarizedData] = None


class V1ASRResponse(V1BaseResponse):
    """Модель ответа для ASR роутов (post_by_url, post_by_file)."""
    data: ASRData = ASRData()


class IsAliveData(BaseModel):
    """Структура данных ответа is_alive."""
    state: str
    tasks_in_work: int
    free_memory_mb: Optional[float] = None
    gpu_load_percent: Optional[float] = None
    temperature_celsius: Optional[float] = None


class V1IsAliveResponse(V1BaseResponse):
    """Модель ответа для is_alive."""
    data: Optional[IsAliveData] = None


class ErrorResponse(BaseModel):
    """
    Модель ошибки API.
    """
    success: bool = False
    error_description: str
    details: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    sentenced_data: Optional[Dict[str, Any]] = None
    diarized_data: Optional[Dict[str, Any]] = None


class UserBase(BaseModel):
    """
    Базовая информация о пользователе (заготовка для JWT).
    """
    username: Optional[str] = None
    email: Optional[str] = None
    is_active: bool = True


class SyncASRRequest(BaseModel):
    """
    :parameter keep_raw: - Если False, то запрос вернёт только пост-обработанные данные do_punctuation и do_dialogue.
    :parameter do_echo_clearing - Проверяет наличие повторений между каналами
    :parameter do_dialogue - Собирает из распознанного текста фразы, разделённые более длинным молчанием,
                             чем некое среднее значение.
    :parameter do_punctuation - расставляет пунктуацию. Пока в разработке
    """

    AudioFileUrl: HttpUrl
    keep_raw: Union[bool, None] = True
    do_echo_clearing: Union[bool, None] = True
    do_dialogue: Union[bool, None] = False
    do_punctuation: Union[bool, None] = False
    do_diarization: Union[bool, None] = False
    make_mono: Union[bool, None] = False
    diar_vad_sensity: int = 3
    do_auto_speech_speed_correction: Union[bool, None] = settings.DO_SPEED_SPEECH_CORRECTION
    speech_speed_correction_multiplier: Union[float, None] = settings.SPEED_SPEECH_CORRECTION_MULTIPLIER
    use_batch: Union[bool, None] = settings.USE_BATCH
    batch_size: Union[int, None] = settings.ASR_BATCH_SIZE

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "AudioFileUrl": "https://example.com/audio.wav",
                "keep_raw": True,
                "do_echo_clearing": True,
                "do_dialogue": False,
                "do_punctuation": False,
                "do_diarization": False,
                "make_mono": False,
                "diar_vad_sensity": 3,
                "do_auto_speech_speed_correction": True,
                "speech_speed_correction_multiplier": 1.0,
                "use_batch": True,
                "batch_size": 8
            }
        }
    )


class PostFileRequest(BaseModel):
    """
    Модель для проверки запроса пользователя.

    :param keep_raw: Если False, то запрос вернёт только пост-обработанные данные do_punctuation и do_dialogue.
    :param do_echo_clearing: Проверяет наличие повторений между каналами.
    :param do_dialogue: Собирает из распознанного текста фразы, разделённые более длинным молчанием,
    чем некое среднее значение.
    :param do_punctuation: Расставляет пунктуацию.
    """
    keep_raw: Union[bool, None] = True
    do_echo_clearing: Union[bool, None] = False
    do_dialogue: Union[bool, None] = False
    do_punctuation: Union[bool, None] = False
    do_diarization: Union[bool, None] = False
    use_batch: Union[bool, None] = settings.USE_BATCH
    batch_size: Union[int, None] = settings.ASR_BATCH_SIZE
    diar_vad_sensity: int = 3
    do_auto_speech_speed_correction: Union[bool, None] = settings.DO_SPEED_SPEECH_CORRECTION
    speech_speed_correction_multiplier: Union[float, None] = settings.SPEED_SPEECH_CORRECTION_MULTIPLIER
    make_mono: Union[bool, None] = False

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "keep_raw": True,
                "do_echo_clearing": False,
                "do_dialogue": False,
                "do_punctuation": False,
                "do_diarization": False,
                "use_batch": True,
                "batch_size": 8,
                "diar_vad_sensity": 3,
                "do_auto_speech_speed_correction": True,
                "speech_speed_correction_multiplier": 1.0,
                "make_mono": False
            }
        }
    )
