from api.v1.endpoints.root import router as root_router
from api.v1.endpoints.asr_url import router as asr_url_router
from api.v1.endpoints.asr_file import router as asr_file_router
from api.v1.endpoints.asr_ws import router as asr_ws_router
from api.v1.endpoints.health import router as health_router

__all__ = [
    "root_router",
    "asr_url_router",
    "asr_file_router",
    "asr_ws_router",
    "health_router",
]
