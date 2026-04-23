import uuid
import asyncio
import os
from fastapi import APIRouter
from utils.pre_start_init import posted_and_downloaded_audio
from utils.do_logging import logger
from utils.get_audio_file import getting_audiofile, open_default_audiofile
from models.fast_api_models import SyncASRRequest, BaseResponse
from Recognizer.engine.file_recognition import process_file
from threading import Lock
from io import BytesIO


router = APIRouter()

# Глобальный лок для потокобезопасности
audio_lock = Lock()

@router.post("/post_one_step_req")
async def post(params: SyncASRRequest):
    """
    На вход принимает HttpUrl - прямую ссылку на скачивание файла 'mp3', 'wav' или 'ogg'.\n
    Если на вход передаётся не моно, то ответ будет в несколько элементов списка для каждого канала.\n
    По умолчанию отдаёт сырой результат распознавания с разбивкой на части продолжительностью около 15 секунд\n

    :param: do_dialogue: - true, если нужно разбить речь на диалог\n
    :param: do_punctuation - true, если нужно расставить пунктуацию. Применяется к диалогу, общему тексту. В проекте.\n
    При проектировании таймаутов учитывайте скорость распознавания (около 100 секунд аудио распознаётся за 2-5 секунд
    распознавания одного канала)
    """
    res = True
    error_description = str()

    result = {
        "success": res,
        "error_description": error_description,
        "raw_data": dict(),
        "sentenced_data": dict(),
    }

    # Получаем файл
    post_id = uuid.uuid4()
    if params.AudioFileUrl:
        res, error_description = await getting_audiofile(params.AudioFileUrl, post_id)
    else:
        res, error_description = await open_default_audiofile(post_id)

    if not res:
        logger.error(f'Ошибка получения файла - {error_description}, ссылка на файл - {params.AudioFileUrl}')
        return {
            "success": False,
            "error_description": error_description,
            "raw_data": dict(),
            "sentenced_data": dict(),
        }

    try:
        # Запускаем обработку в потоке
        result = await asyncio.to_thread(process_file, posted_and_downloaded_audio[post_id], params)
    except Exception as e:
        error_description = f"Ошибка обработки в process_file - {e}"
        logger.error(error_description)
        result["success"] = False
        result['error_description'] = str(error_description)

    return result
