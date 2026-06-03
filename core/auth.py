from typing import Optional

from models.domain.user import User
from models.enums import Role
from core.security import get_password_hash


async def authenticate_user(email: str, password: str) -> Optional[User]:
    """Аутентификация пользователя по email и паролю (заглушка)."""
    # TODO: реализовать проверку через БД
    return None


async def register_user(email: str, password: str) -> User:
    """Регистрация нового пользователя (заглушка)."""
    # TODO: реализовать сохранение в БД
    hashed_password = get_password_hash(password)
    return User(
        id="usr_new_12345",
        email=email,
        hashed_password=hashed_password,
        role=Role.user,
        is_active=True,
    )
