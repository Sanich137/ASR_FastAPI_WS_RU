from pydantic import BaseModel, HttpUrl, Field
from typing import Union, Annotated
from fastapi import UploadFile

import config


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
    do_auto_speech_speed_correction: Union[bool, None] = config.DO_SPEED_SPEECH_CORRECTION
    speech_speed_correction_multiplier: Union[float, None] = config.SPEED_SPEECH_CORRECTION_MULTIPLIER
    use_batch: Union[bool, None] = config.USE_BATCH
    batch_size: Union[int, None] = config.ASR_BATCH_SIZE




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
    use_batch: Union[bool, None] = config.USE_BATCH
    batch_size: Union[int, None] = config.ASR_BATCH_SIZE
    diar_vad_sensity: int = 3
    do_auto_speech_speed_correction: Union[bool, None] = config.DO_SPEED_SPEECH_CORRECTION
    speech_speed_correction_multiplier: Union[float, None] = config.SPEED_SPEECH_CORRECTION_MULTIPLIER
    make_mono: Union[bool, None] = False

class PostFileRequestDiarize(BaseModel):
    """
    Модель для проверки запроса пользователя.
    :param keep_raw: Если False, то запрос вернёт только пост-обработанные данные do_punctuation и do_dialogue.
    :param do_echo_clearing: Проверяет наличие повторений между каналами.
    :param num_speakers: Предполагаемое количество спикеров в разговоре. -1 - значит мы не знаем сколько спикеров и определяем их параметром cluster_threshold.
    :param cluster_threshold: Значение от 0 до 1. Чем меньше, тем более чувствительное выделение спикеров (тем их больше)
    :param do_punctuation: Расставляет пунктуацию.
    """
    keep_raw: bool = True
    do_echo_clearing: bool = False
    do_punctuation: bool = False
    num_speakers: int = -1,
    cluster_threshold: float = 0.2

class WebSocketModel(BaseModel):
    """OpenAPI не хочет описывать WS, а я не хочу изучать OPEN API. По этому описание тут.
    \n
    \n Подключение на порт: 49153
    \n На вход жду поток binary, buffer_size +- 6400, mono, wav.
    \n На вход я должен получить словарь {'text': { "config" : { "sample_rate" : any(int/float), "wait_null_answers": Bool,
    "do_dialogue": Bool, "do_punctuation": Bool}}}
    \n do_punctuation отработает только если do_dialogue = True
    \n Далее сообщения с данными {"bytes": binary}
    \n По окончании передачи {'text': '{ "eof" : 1}'}
    \n Ответ получать в формате:     {"silence": Bool,"data": str, "error": None/str, "last_message": Bool,
  "sentenced_data": {}}


    \n Пример ответа "data": {
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
    "raw_text_sentenced_recognition": "channel_1: Татьяна, добрый день. Меня зовут Ульяна.\nchannel_1: Звоню уточнить по поводу документов.\n",
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
    pass