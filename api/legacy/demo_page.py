
import logging
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
logger = logging.getLogger(__name__)

router = APIRouter()

# Setup templates
templates = Jinja2Templates(directory="templates")

@router.get("/demo", response_class=HTMLResponse)
async def demo_page(request: Request):
    logger.warning(
        "Legacy endpoint /demo is deprecated. Use /api/v1/ instead.",
    )
    return templates.TemplateResponse(
        "index.html",
        {"request": request}  # Контекст для Jinja2 (если нужно)
    )
