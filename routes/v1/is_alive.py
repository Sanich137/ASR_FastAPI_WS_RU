from fastapi import APIRouter
import logging
from utils.pre_start_init import audio_to_asr
from routes.is_alive import get_gpu_free_memory
from models.fast_api_models import V1BaseResponse

router = APIRouter()

@router.get("/is_alive", response_model=V1BaseResponse)
async def check_if_service_is_alive_v1():
    logging.info('GET /v1/is_alive')
    error_description = None
    tasks_in_work = len(audio_to_asr)
    error, free_mb, gpu_load, temperature = get_gpu_free_memory()
    state = "idle" if tasks_in_work == 0 else "in_work"

    if error:
        error_description = error.get("error",None)
        data = None
    else:
        data = {
            "state": state,
            "tasks_in_work": tasks_in_work,
            "free_memory_mb": free_mb,
            "gpu_load_percent": gpu_load,
            "temperature_celsius": temperature
        }

    return V1BaseResponse(
        success=True,
        error_description=error_description,
        data=data
    )
