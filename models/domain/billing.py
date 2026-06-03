from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field

from models.enums import SubscriptionType, SubscriptionStatus, PaymentMethod, TransactionStatus


class Subscription(BaseModel):
    """Модель подписки пользователя."""
    id: str
    user_id: str
    type: SubscriptionType
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    status: SubscriptionStatus = SubscriptionStatus.active
    payment_method: Optional[PaymentMethod] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "sub_12345",
                "user_id": "usr_12345",
                "type": "pro",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-02-01T00:00:00Z",
                "status": "active",
                "payment_method": "card",
            }
        }
    }


class Transaction(BaseModel):
    """Модель разового платежа."""
    id: str
    user_id: str
    amount: Decimal = Field(decimal_places=2, gt=0)
    currency: str = Field(default="RUB", max_length=3)
    status: TransactionStatus
    external_id: Optional[str] = Field(default=None, description="ID платежа во внешней системе")
    payment_method: PaymentMethod
    is_recurring: bool = False
    created_at: Optional[datetime] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "txn_12345",
                "user_id": "usr_12345",
                "amount": "999.00",
                "currency": "RUB",
                "status": "completed",
                "external_id": "ext_12345",
                "payment_method": "card",
                "is_recurring": False,
                "created_at": "2024-01-01T12:00:00Z",
            }
        }
    }


class RecurringPayment(BaseModel):
    """Модель рекуррентного (автоматического) платежа."""
    id: str
    user_id: str
    subscription_id: str
    amount: Decimal = Field(decimal_places=2, gt=0)
    currency: str = Field(default="RUB", max_length=3)
    interval_days: int = Field(default=30, ge=1, description="Периодичность списания в днях")
    next_payment_date: Optional[datetime] = None
    status: SubscriptionStatus = SubscriptionStatus.active
    payment_method: PaymentMethod
    external_subscription_id: Optional[str] = Field(
        default=None, description="ID подписки в платёжном шлюзе"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "rec_12345",
                "user_id": "usr_12345",
                "subscription_id": "sub_12345",
                "amount": "999.00",
                "currency": "RUB",
                "interval_days": 30,
                "next_payment_date": "2024-02-01T00:00:00Z",
                "status": "active",
                "payment_method": "card",
                "external_subscription_id": "ext_sub_12345",
            }
        }
    }
