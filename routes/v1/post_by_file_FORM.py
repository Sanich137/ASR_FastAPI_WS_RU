from io import BytesIO
import asyncio
from fastapi import APIRouter, Depends, File, Form, UploadFile
from config import settings
from utils.do_logging import logger
from models.fast_api_models import PostFileRequest, V1BaseResponse
from Recognizer.engine.file_recognition import process_file
from routes.post_by_file_FORM import get_file_request

router = APIRouter()

@router.post("/post_file", response_model=V1BaseResponse)
async def async_receive_file_v1(
    file: UploadFile = File(description="Аудиофайл для обработки"),
    params: PostFileRequest = Depends(get_file_request),
) -> V1BaseResponse:
    try:
        buffer = BytesIO(await file.read())
        buffer.seek(0)
    except Exception as e:
        error_description = f"Не удалось сохранить файл для распознавания: {file.filename}, размер файла: {file.size}, по причине: {e}"
        logger.error(error_description)
        return V1BaseResponse(
            success=False,
            error_description=error_description,
            data={}
        )
    else:
        logger.info(f"Получен и сохранён файл {file.filename}")
        try:
            result_dict = await asyncio.to_thread(process_file, buffer, params)
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
        finally:
            await file.close()
            del file
