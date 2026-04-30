from io import BytesIO
import asyncio

from config import settings
from fastapi import APIRouter, Depends, File, Form, UploadFile
from utils.do_logging import logger
from models.fast_api_models import PostFileRequest, BaseResponse
from Recognizer import get_recognizer, Recognizer
from Recognizer.engine.file_recognition import process_file
from threading import Lock


router = APIRouter()

# Функция для извлечения параметров из FormData
def get_file_request(
    keep_raw: bool = Form(default=True, description="Сохранять сырые данные."),
    do_echo_clearing: bool = Form(default=False, description="Убирать межканальное эхо."),
    do_dialogue: bool = Form(default=False, description="Строить диалог."),
    do_punctuation: bool = Form(default=False, description="Восстанавливать пунктуацию."),
    do_diarization: bool = Form(default=False, description="Разделять по спикерам."),
    diar_vad_sensity: int = Form(default=3, description="Чувствительность VAD."),
    use_batch: bool = Form(default=settings.USE_BATCH, description="Использовать батчинг для ASR."),
    batch_size: int = Form(default=settings.ASR_BATCH_SIZE, description="Размер батча для ASR."),
    do_auto_speech_speed_correction: bool = Form(default=True, description="Корректировать скорость речи при распознавании."),
    speech_speed_correction_multiplier: float = Form(default=1, description="Базовый коэффициент скорости речи."),
    make_mono: bool = Form(default=False, description="Соединить несколько каналов в mono"),
) -> PostFileRequest:
    return PostFileRequest(
        keep_raw=keep_raw,
        do_echo_clearing=do_echo_clearing,
        do_dialogue=do_dialogue,
        do_punctuation=do_punctuation,
        do_diarization=do_diarization,
        use_batch=use_batch,
        diar_vad_sensity=diar_vad_sensity,
        batch_size=batch_size,
        make_mono=make_mono,
        do_auto_speech_speed_correction = do_auto_speech_speed_correction,
        speech_speed_correction_multiplier = speech_speed_correction_multiplier
    )


@router.post("/post_file", response_model=BaseResponse)
async def async_receive_file_legacy(
    file: UploadFile = File(description="Аудиофайл для обработки"),
    params: PostFileRequest = Depends(get_file_request),
    recognizer: Recognizer = Depends(get_recognizer)
) -> BaseResponse:
    # Сохраняем файл на диск асинхронно
    try:
        buffer = BytesIO(await file.read())
        buffer.seek(0)
    except Exception as e:
        error_description = f"Не удалось сохранить файл для распознавания: {file.filename}, размер файла: {file.size}, по причине: {e}"
        logger.error(error_description)
        return BaseResponse(
            success=False,
            error_description=error_description,
            raw_data={},
            sentenced_data={},
            diarized_data={},
        )
    else:
        logger.info(f"Получен и сохранён файл {file.filename}")
        try:
            # Запускаем обработку в потоке
            result_dict = await asyncio.to_thread(process_file, buffer, params, recognizer)
            result = BaseResponse(**result_dict)
        except Exception as e:
            error_description = f"Ошибка обработки в process_file - {e}"
            logger.error(error_description)
            result = BaseResponse(
                success=False,
                error_description=str(error_description),
                raw_data={},
                sentenced_data={},
                diarized_data={},
            )
        finally:
            # Удаляем временный файл
            await file.close()
            del file
    finally:
        logger.info((f"{result}"))
        return result
