from fastapi import APIRouter
from .is_alive import router as is_alive_router
from .post_by_url import router as post_by_url_router
from .post_by_file_FORM import router as post_by_file_router
from .root import router as root_router

router = APIRouter(prefix="/v1")
router.include_router(root_router)
router.include_router(is_alive_router)
router.include_router(post_by_url_router)
router.include_router(post_by_file_router)
