from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field

from models.enums import Role, SubscriptionType
from models.domain.billing import Subscription


class User(BaseModel):
    """Доменная модель пользователя."""
    id: str
    email: Optional[str] = None
    hashed_password: Optional[str] = None
    role: Role = Role.user
    is_active: bool = True
    daily_quota: int = Field(default=10, ge=0, description="Дневная квота запросов")
    quota_used_today: int = Field(default=0, ge=0, description="Использовано сегодня")
    subscription_type: Optional[SubscriptionType] = None
    subscription_expires: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "usr_12345",
                "email": "user@example.com",
                "role": "user",
                "is_active": True,
                "daily_quota": 10,
                "quota_used_today": 0,
                "subscription_type": None,
                "subscription_expires": None,
            }
        }
    }


class UserAccount(BaseModel):
    """Модель аккаунта пользователя с балансом и подпиской."""
    user_id: str
    balance: Decimal = Field(default=Decimal("0.00"), decimal_places=2)
    currency: str = Field(default="RUB", max_length=3)
    tariff: str = Field(default="free")
    subscription: Optional[Subscription] = None
    auto_renew: bool = False

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "usr_12345",
                "balance": "0.00",
                "currency": "RUB",
                "tariff": "free",
                "subscription": None,
                "auto_renew": False,
            }
        }
    }


class QuotaInfo(BaseModel):
    """Информация о текущей квоте пользователя."""
    used: int = Field(ge=0, description="Использовано запросов")
    limit: int = Field(ge=0, description="Лимит запросов")
    reset_at: datetime = Field(description="Время сброса квоты")

    model_config = {
        "json_schema_extra": {
            "example": {
                "used": 5,
                "limit": 10,
                "reset_at": "2024-01-02T00:00:00Z",
            }
        }
    }


class UserProfileResponse(BaseModel):
    """Унифицированный ответ с профилем пользователя."""
    user: User
    account: UserAccount
    quota: QuotaInfo
