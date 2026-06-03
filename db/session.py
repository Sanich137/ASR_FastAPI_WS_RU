"""Асинхронная сессия SQLAlchemy для FastAPI."""

import os

from sqlalchemy import NullPool
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# Placeholder: при запуске приложения URL должен переопределяться
# через config.py (Settings.DATABASE_URL).
# Для локальной разработки используем async SQLite (aiosqlite).
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./asr_local.db",
)

# Для SQLite рекомендуется NullPool, чтобы избежать проблем с пулом
# в однопоточном режиме.
_pool_cls = NullPool if DATABASE_URL.startswith("sqlite") else None

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    poolclass=_pool_cls,
)
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db_session():
    """Генератор сессии для FastAPI Depends."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
