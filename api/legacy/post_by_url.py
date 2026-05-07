import uuid
import asyncio
from fastapi import APIRouter, Depends
from utils.get_audio_file import getting_audiofile, open_default_audiofile
from models.fast_api_models import SyncASRRequest, BaseResponse

from Recognizer import get_recognizer, Recognizer
from Punctuation import get_punctuator, SbertPuncCaseOnnx
from Recognizer.engine.file_recognition import process_file

from Diarisation import get_diarizer
from Diarisation.do_diarize import Diarizer

import logging
logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/post_one_step_req", response_model=BaseResponse)
async def post(params: SyncASRRequest,
                recognizer: Recognizer = Depends(get_recognizer),
                punctuator: SbertPuncCaseOnnx = Depends(get_punctuator),
                diarizer: Diarizer = Depends(get_diarizer)
) -> BaseResponse:
    """
    На вход ждёт str(HttpUrl) - прямую ссылку на скачивание файла 'mp3', 'wav' или 'ogg'.\n
    Если на вход передаётся не моно, то ответ будет в несколько элементов списка для каждого канала.\n

    :param: do_dialogue: - true, если нужно разбить речь на диалог\n
    :param: do_punctuation - true, если нужно расставить пунктуацию. Применяется к диалогу, и отдельно к общему тексту.\n
    :param:keep_raw: Сохранять ли в выводе "сырые данные" - распознавание по словам. \n
    :param:do_echo_clearing: Очищать от межканального эха \n
    :param:do_diarization: Разделать речь на спикеров. Работает только с моно файлами. \n
    :param:make_mono: Объединить каналы аудио в моно файл \n
    :param:diar_vad_sensity: Чувствительность детектора голоса. \n
    :param:do_auto_speech_speed_correction: Корректировать скорость речи (для очень быстрой речи). \n
    :param:speech_speed_correction_multiplier: Задать коэффициент корректировки скорости речи \n
    :param:use_batch: Union[bool, None] = Использовать пакетную обработку. Полезно при невозможности использовать Tensorrt \n
    :param:batch_size: Union[int, None] = Размер пакета для обработки. \n
    """

    logger.warning(
        "Legacy endpoint /post_one_step_req is deprecated. Use /api/v1/asr/url instead.",
    )

    # Получаем файл
    post_id = uuid.uuid4()
    if params.AudioFileUrl:
        res, error_description, buffer = await getting_audiofile(params.AudioFileUrl, post_id)
    else:
        res, error_description, buffer = await open_default_audiofile(post_id)

    if not res:
        logger.error(f'Ошибка получения файла - {error_description}, ссылка на файл - {params.AudioFileUrl}')
        return BaseResponse(
            success=False,
            error_description=error_description,
            raw_data={},
            sentenced_data={},
            diarized_data={},
        )

    try:
        # Запускаем обработку в потоке
        result_dict = await asyncio.to_thread(process_file,
                                              tmp_path=buffer,
                                              params=params,
                                              recognizer=recognizer,
                                              punctuator=punctuator,
                                              diarizer=diarizer)

        result = BaseResponse(**result_dict)
    except Exception as e:
        error_description = f"Ошибка обработки в process_file - {e}"
        logger.error(error_description)
        return BaseResponse(
            success=False,
            error_description=str(error_description),
            raw_data={},
            sentenced_data={},
            diarized_data={},
        )

    return result
