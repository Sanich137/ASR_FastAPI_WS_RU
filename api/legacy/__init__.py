import logging
from fastapi import APIRouter
from .is_alive import router as is_alive_router
from .post_by_url import router as post_by_url_router
from .post_by_file_FORM import router as post_by_file_router
from .root import router as root_router
from .demo_page import router as demo_page_router

logger = logging.getLogger(__name__)
logger.warning(
    "Legacy API routes are deprecated. Use /api/v1/ instead.",
)

router = APIRouter()
router.include_router(root_router)
router.include_router(demo_page_router)
router.include_router(is_alive_router)
router.include_router(post_by_url_router)
router.include_router(post_by_file_router)
