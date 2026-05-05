from fastapi import APIRouter
import logging
import pynvml
from utils.pre_start_init import audio_to_asr

logger = logging.getLogger(__name__)
router = APIRouter()

def get_gpu_free_memory():
    try:
        # todo Доработать, выдавать ответ в зависимости от провайдера.
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Первая видеокарта
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mb = mem_info.free / 1024**2
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_load = utilization.gpu
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except pynvml.NVMLError as e:
        return {"error": str(e)}, None, None, None
    finally:
        pynvml.nvmlShutdown()
    return None, free_mb, gpu_load,temperature


@router.get("/is_alive")
async def check_if_service_is_alive():
    logger.warning(
        "Legacy endpoint /is_alive is deprecated. Use /api/v1/health/is_alive instead.",
    )
    error_description = None
    logging.info('GET_is_alive')
    tasks_in_work = len(audio_to_asr)

    error, free_mb, gpu_load,temperature = get_gpu_free_memory()
    if error:
        error_description = error.get("error",None)

    if tasks_in_work == 0:
        state = "idle"
    else:
        state = "in_work"

    return {"error": False,
            "error_description": error_description,
            "state": state,
            "tasks_in_work": tasks_in_work,
            "free_memory_mb": free_mb,
            "gpu_load_percent": gpu_load,
            "temperature_celsius": temperature
            }
