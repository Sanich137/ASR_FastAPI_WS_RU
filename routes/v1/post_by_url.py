import uuid
import asyncio
from utils.pre_start_init import posted_and_downloaded_audio
from utils.get_audio_file import getting_audiofile, open_default_audiofile
from models.fast_api_models import SyncASRRequest, V1ASRResponse, ASRData, RawData, SentencedData, DiarizedData

from fastapi import APIRouter, Depends
from Recognizer import get_recognizer, Recognizer
from Recognizer.engine.file_recognition import process_file
from Punctuation import get_punctuator, SbertPuncCaseOnnx
from Diarisation import get_diarizer
from Diarisation.do_diarize import Diarizer


import logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/post_one_step_req", response_model=V1ASRResponse)
async def post_v1(params: SyncASRRequest,
                  recognizer: Recognizer = Depends(get_recognizer),
                  punctuator: SbertPuncCaseOnnx = Depends(get_punctuator),
                  diarizer: Diarizer = Depends(get_diarizer)
                  ) -> V1ASRResponse:
    post_id = uuid.uuid4()
    if params.AudioFileUrl:
        res, error_description = await getting_audiofile(params.AudioFileUrl, post_id)
    else:
        res, error_description = await open_default_audiofile(post_id)

    if not res:
        logger.error(f'Ошибка получения файла - {error_description}, ссылка на файл - {params.AudioFileUrl}')
        return V1ASRResponse(
            success=False,
            error_description=error_description,
            data=ASRData()
        )

    try:
        result_dict = await asyncio.to_thread(process_file,
                                              tmp_path=posted_and_downloaded_audio[post_id],
                                              params=params,
                                              recognizer=recognizer,
                                              punctuator=punctuator,
                                              diarizer=diarizer)
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
