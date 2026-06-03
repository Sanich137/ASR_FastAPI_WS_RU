import asyncio
import logging
from config import settings
from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from core.deps import get_current_user_or_none
from db.session import get_db_session
from db.models import ASRSession
from db.enums import ASRSessionStatus, ASRSessionType
from models.fast_api_models import PostFileRequest, V1BaseResponse, ASRData, RawData, SentencedData, DiarizedData
from services.recognition_session import FileRecognitionSession
from Recognizer.engine.file_recognition import process_file
from Recognizer import get_recognizer, Recognizer
from Punctuation import get_punctuator, SbertPuncCaseOnnx

from Diarisation import get_diarizer
from Diarisation.do_diarize import Diarizer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/asr", tags=["ASR"])


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
        do_auto_speech_speed_correction=do_auto_speech_speed_correction,
        speech_speed_correction_multiplier=speech_speed_correction_multiplier
    )


@router.post("/file", response_model=V1BaseResponse)
async def async_receive_file(
    file: UploadFile = File(description="Аудиофайл для обработки"),
    params: PostFileRequest = Depends(get_file_request),
    recognizer: Recognizer = Depends(get_recognizer),
    punctuator: SbertPuncCaseOnnx = Depends(get_punctuator),
    diarizer: Diarizer = Depends(get_diarizer),
    current_user = Depends(get_current_user_or_none),
    db: AsyncSession = Depends(get_db_session),
) -> V1BaseResponse:
    from datetime import datetime, timezone

    session = FileRecognitionSession(params=params)
    asr_db_session = None
    if current_user:
        asr_db_session = ASRSession(
            user_id=current_user.id,
            session_type=ASRSessionType.file,
            status=ASRSessionStatus.processing,
        )
        db.add(asr_db_session)
        await db.commit()
        await db.refresh(asr_db_session)

    try:
        await session.save_upload(file)
        logger.info(f"Получен и сохранён файл {file.filename}")
        result_dict = await asyncio.to_thread(
            process_file,
            session=session,
            recognizer=recognizer,
            punctuator=punctuator,
            diarizer=diarizer
        )
        if asr_db_session:
            try:
                asr_db_session.status = ASRSessionStatus.completed if result_dict.get('success') else ASRSessionStatus.failed
                asr_db_session.completed_at = datetime.now(timezone.utc)
                asr_db_session.result_json = result_dict
                await db.commit()
            except Exception as exc:
                logger.debug("Failed to save ASRSession result: %s", exc)

        return V1BaseResponse(
            success=result_dict.get('success', True),
            error_description=result_dict.get('error_description'),
            data=ASRData(
                raw_data=RawData.model_validate(result_dict.get('raw_data', {})) if result_dict.get('raw_data') else None,
                sentenced_data=SentencedData(**result_dict.get('sentenced_data', {})) if result_dict.get('sentenced_data') else None,
                diarized_data=DiarizedData(**result_dict.get('diarized_data', {})) if result_dict.get('diarized_data') else None
            )
        )
    except Exception as e:
        error_description = f"Ошибка обработки в process_file - {e}"
        logger.error(error_description)
        return V1BaseResponse(
            success=False,
            error_description=str(error_description),
            data=ASRData()
        )
    finally:
        await file.close()
        session.cleanup()
        await session.reset()
