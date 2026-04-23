from fastapi import APIRouter
from models.fast_api_models import BaseResponse
import config

router = APIRouter()

@router.get("/", response_model=BaseResponse)
async def root():
    return BaseResponse(
        success=True,
        error_description=None,
        data={"message": "No_service_selected",
              "available_endpoints": {
                  "POST /post_one_step_req": "ASR by URL",
                  "POST /post_file": "ASR by file upload",
                  "WS /ws": "WebSocket streaming ASR",
                  "GET /is_alive": "Service health check",
                  "GET /docs": "API documentation",
                  "/demo": "DEMO UI page"
              },
              "try_addr": f"http://{config.settings.HOST}:{config.settings.PORT}/docs"}
    )
