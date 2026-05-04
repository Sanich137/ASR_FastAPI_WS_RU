from pydantic import BaseModel, HttpUrl, Field, ConfigDict
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
    data: dict = {}


class RawData(BaseModel):
    """Структура сырых данных ASR."""
    result: Optional[List[Dict[str, Any]]] = None
    text: Optional[str] = None


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

class WebSocketModel(BaseModel):
    """OpenAPI не хочет описывать WS, а я не хочу изучать OPEN API. По этому описание тут.

    Подключение на порт: 49153
    На вход жду поток binary, buffer_size +- 6400, mono, wav.

    Протокол обмена сообщениями:
      1. Начальное сообщение (конфигурация):
         {'text': { "config" : { "sample_rate" : any(int/float), "wait_null_answers": Bool,
         "do_dialogue": Bool, "do_punctuation": Bool}}}
         do_punctuation отработает только если do_dialogue = True

      2. Последующие сообщения с аудио-данными:
         {"bytes": binary}

      3. Последнее сообщение (сигнал окончания передачи):
         {'text': '{ "eof" : 1}'}

    Формат ответа от сервера:
      {"silence": Bool, "data": str, "error": None/str, "last_message": Bool,
       "sentenced_data": {}}

    Пример ответа "data": {
      "result" : [{
          "conf" : 1.000000,
          "end" : 3.120000,
          "start" : 2.340000,
          "word" : "здравствуйте"
        }, {
          "conf" : 1.000000,
          "end" : 3.870000,
          "start" : 3.600000,
          "word" : "вы"
        },
         ...
         {
          "conf" : 0.994019,
          "end" : 11.790000,
          "start" : 10.890000,
          "word" : "записываются"
        }],
      "text" : "здравствуйте вы ... записываются"
    }

    Пример ответа "sentenced_data": {
        "raw_text_sentenced_recognition": "channel_1: Татьяна, добрый день. Меня зовут Ульяна.'/n'channel_1: Звоню уточнить по поводу документов.",
        "list_of_sentenced_recognitions": [
          {
            "start": 2.28,
            "text": "Татьяна, добрый день. Меня зовут Ульяна.",
            "speaker": "channel_1"
          },
          {
            "start": 8.24,
            "text": "Звоню уточнить по поводу документов.",
            "speaker": "channel_1"
          },
        ],
        "full_text_only": [
          "Татьяна, добрый день. Меня зовут Ульяна. Звоню уточнить по поводу документов."
        ],
        "err_state": null
      }
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": {
                    "config": {
                        "sample_rate": 8000,
                        "wait_null_answers": False,
                        "do_dialogue": True,
                        "do_punctuation": True
                    }
                }
            }
        }
    )

    pass
