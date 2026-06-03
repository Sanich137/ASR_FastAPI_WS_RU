from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field


class ASRUsageLog(BaseModel):
    """Лог использования ASR для учёта квот и биллинга."""
    id: str
    user_id: str
    endpoint: str = Field(description="Вызываемый эндпоинт")
    audio_duration_sec: Optional[float] = Field(default=None, ge=0)
    request_size_bytes: int = Field(default=0, ge=0)
    timestamp: Optional[datetime] = None
    cost: Optional[Decimal] = Field(default=None, decimal_places=2)
    quota_consumed: int = Field(default=0, ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "log_12345",
                "user_id": "usr_12345",
                "endpoint": "/api/v1/asr/url",
                "audio_duration_sec": 120.5,
                "request_size_bytes": 1048576,
                "timestamp": "2024-01-01T12:00:00Z",
                "cost": "15.50",
                "quota_consumed": 1,
            }
        }
    }


class AdminActionLog(BaseModel):
    """Лог административных действий для аудита."""
    id: str
    admin_id: str
    action: str = Field(description="Выполненное действие")
    target_user_id: Optional[str] = Field(default=None, description="ID целевого пользователя")
    details: dict = Field(default_factory=dict, description="Дополнительные параметры")
    timestamp: Optional[datetime] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "adm_12345",
                "admin_id": "usr_admin",
                "action": "block_user",
                "target_user_id": "usr_12345",
                "details": {"reason": "violation"},
                "timestamp": "2024-01-01T12:00:00Z",
            }
        }
    }
