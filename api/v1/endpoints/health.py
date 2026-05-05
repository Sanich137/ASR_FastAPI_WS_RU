import logging
from fastapi import APIRouter
from models.fast_api_models import V1BaseResponse, IsAliveData
from api.legacy.is_alive import get_gpu_free_memory
from utils.pre_start_init import audio_to_asr

router = APIRouter(prefix="/health", tags=["Health"])
logger = logging.getLogger(__name__)


@router.get("/live", response_model=V1BaseResponse)
async def health_live():
    """Liveness probe для Kubernetes."""
    return V1BaseResponse(
        success=True,
        error_description=None,
        data={"status": "ok"}
    )


@router.get("/ready", response_model=V1BaseResponse)
async def health_ready():
    """Readiness probe — базовая заглушка до реализации Задачи 4.2."""
    # TODO: проверить загрузку моделей, доступность GPU, свободную память
    return V1BaseResponse(
        success=True,
        error_description=None,
        data={"status": "ok", "checks": {"models_loaded": True, "gpu_available": None}}
    )


@router.get("/is_alive", response_model=V1BaseResponse)
async def health_is_alive():
    """Перенос текущего is_alive для консистентности API v1."""
    logger.info('GET /api/v1/health/is_alive')
    error_description = None
    tasks_in_work = len(audio_to_asr)
    error, free_mb, gpu_load, temperature = get_gpu_free_memory()
    state = "idle" if tasks_in_work == 0 else "in_work"

    if error:
        error_description = error.get("error", None)
        return V1BaseResponse(
            success=True,
            error_description=error_description,
            data=None
        )
    else:
        return V1BaseResponse(
            success=True,
            error_description=None,
            data=IsAliveData(
                state=state,
                tasks_in_work=tasks_in_work,
                free_memory_mb=free_mb,
                gpu_load_percent=gpu_load,
                temperature_celsius=temperature
            )
        )
