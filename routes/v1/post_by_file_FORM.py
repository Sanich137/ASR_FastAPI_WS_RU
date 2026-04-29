from io import BytesIO
import asyncio
from utils.do_logging import logger
from config import settings
from fastapi import APIRouter, Depends, File, Form, UploadFile
from models.fast_api_models import PostFileRequest, V1ASRResponse, ASRData, RawData, SentencedData, DiarizedData
from Recognizer.engine.file_recognition import process_file

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


@router.post("/post_file", response_model=V1ASRResponse)
async def async_receive_file(
    file: UploadFile = File(description="Аудиофайл для обработки"),
    params: PostFileRequest = Depends(get_file_request),
) -> V1ASRResponse:
    try:
        buffer = BytesIO(await file.read())
        buffer.seek(0)
    except Exception as e:
        error_description = f"Не удалось сохранить файл для распознавания: {file.filename}, размер файла: {file.size}, по причине: {e}"
        logger.error(error_description)
        return V1ASRResponse(
            success=False,
            error_description=error_description,
            data=ASRData()
        )
    else:
        logger.info(f"Получен и сохранён файл {file.filename}")
        try:
            result_dict = await asyncio.to_thread(process_file, buffer, params)
            return V1ASRResponse(
                success=result_dict.get('success', True),
                error_description=result_dict.get('error_description'),
                data=ASRData(
                    raw_data=RawData(**result_dict.get('raw_data', {})) if result_dict.get('raw_data') else None,
                    sentenced_data=SentencedData(**result_dict.get('sentenced_data', {})) if result_dict.get('sentenced_data') else None,
                    diarized_data=DiarizedData(**result_dict.get('diarized_data', {})) if result_dict.get('diarized_data') else None
                )
            )
        except Exception as e:
            error_description = f"Ошибка обработки в process_file - {e}"
            logger.error(error_description)
            return V1ASRResponse(
                success=False,
                error_description=str(error_description),
                data=ASRData()
            )
        finally:
            await file.close()
            del file
