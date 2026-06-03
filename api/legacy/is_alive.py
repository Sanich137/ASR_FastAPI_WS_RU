from fastapi import APIRouter
import logging
import pynvml

logger = logging.getLogger(__name__)
router = APIRouter()

# Кэшированный handle pynvml для избежания Init/Shutdown на каждый запрос
_nvml_handle = None

def _get_nvml_handle():
    global _nvml_handle
    if _nvml_handle is None:
        try:
            pynvml.nvmlInit()
            _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return None
    return _nvml_handle

def get_gpu_free_memory():
    handle = _get_nvml_handle()
    if handle is None:
        return {"error": "GPU not available or pynvml not initialized"}, None, None, None
    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mb = mem_info.free / 1024**2
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_load = utilization.gpu
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except pynvml.NVMLError as e:
        return {"error": str(e)}, None, None, None
    return None, free_mb, gpu_load, temperature


@router.get("/is_alive")
async def check_if_service_is_alive():
    logger.warning(
        "Legacy endpoint /is_alive is deprecated. Use /api/v1/health/is_alive instead.",
    )
    error_description = None
    logging.info('GET_is_alive')
    # Legacy: глобальный audio_to_asr больше не используется в новой архитектуре
    tasks_in_work = 0

    error, free_mb, gpu_load, temperature = get_gpu_free_memory()
    if error:
        error_description = error.get("error", None)

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
