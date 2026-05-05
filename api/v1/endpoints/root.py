from fastapi import APIRouter
from models.fast_api_models import V1BaseResponse
from config import settings

router = APIRouter(prefix="", tags=["System"])


@router.get("/", response_model=V1BaseResponse)
async def root():
    """
    Корневой эндпоинт API v1.

    Returns:
        V1BaseResponse: базовый ответ с приветственным сообщением.
    """
    return V1BaseResponse(
        success=True,
        error_description=None,
        data={
            "message": "No_service_selected",
            "available_endpoints": {
                "POST /api/v1/asr/url": "ASR by URL",
                "POST /api/v1/asr/file": "ASR by file upload",
                "GET /api/v1/health/is_alive": "Service health check",
                "WS /api/v1/asr/ws": "WebSocket streaming ASR",
                "GET /docs": "API documentation",
                "/demo": "DEMO UI page"
            },
            "try_addr": f"http://{settings.HOST}:{settings.PORT}/docs"
        }
    )
