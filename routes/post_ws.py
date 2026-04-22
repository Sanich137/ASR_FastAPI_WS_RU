from fastapi import APIRouter
from models.fast_api_models import WebSocketModel

router = APIRouter()

@router.post("/ws")
async def post_not_websocket(ws:WebSocketModel):
    """Описание для вебсокета ниже в описании WebSocketModel """
    return f"Прочти инструкцию в Schemas - 'WebSocketModel'"
