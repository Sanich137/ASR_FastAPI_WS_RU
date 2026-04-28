import uuid
import asyncio
from fastapi import APIRouter
from utils.pre_start_init import posted_and_downloaded_audio
from utils.do_logging import logger
from utils.get_audio_file import getting_audiofile, open_default_audiofile
from models.fast_api_models import SyncASRRequest, V1BaseResponse
from Recognizer.engine.file_recognition import process_file

router = APIRouter()

@router.post("/post_one_step_req", response_model=V1BaseResponse)
async def post_v1(params: SyncASRRequest) -> V1BaseResponse:
    post_id = uuid.uuid4()
    if params.AudioFileUrl:
        res, error_description = await getting_audiofile(params.AudioFileUrl, post_id)
    else:
        res, error_description = await open_default_audiofile(post_id)

    if not res:
        logger.error(f'Ошибка получения файла - {error_description}, ссылка на файл - {params.AudioFileUrl}')
        return V1BaseResponse(
            success=False,
            error_description=error_description,
            data={}
        )

    try:
        result_dict = await asyncio.to_thread(process_file, posted_and_downloaded_audio[post_id], params)
        return V1BaseResponse(
            success=result_dict.get('success', True),
            error_description=result_dict.get('error_description'),
            data={
                "raw_data": result_dict.get('raw_data'),
                "sentenced_data": result_dict.get('sentenced_data'),
                "diarized_data": result_dict.get('diarized_data')
            }
        )
    except Exception as e:
        error_description = f"Ошибка обработки в process_file - {e}"
        logger.error(error_description)
        return V1BaseResponse(
            success=False,
            error_description=str(error_description),
            data={}
        )
