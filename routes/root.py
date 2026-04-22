from fastapi import APIRouter
import config

router = APIRouter()

@router.get("/")
async def root():
    print("Зашли в root")

    return {"error": True,
            "data": "No_service_selected",
            # "available_services": ["Vosk_Recognizer"],
            "comment": f"try_addr: http://{config.HOST}:{config.PORT}/docs"}
