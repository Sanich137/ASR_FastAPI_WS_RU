"""
Модуль core/state_store.py
Абстракция хранилища состояния для WebSocket-сессий и мета-информации.
Подготовка к кластеризации (Redis) без изменения бизнес-логики.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional


class StateStore(ABC):
    """
    Протокол (ABC) для хранилища состояния.

    Реализации должны поддерживать асинхронные операции get/set/delete
    для сериализуемых мета-данных сессий (не AudioSegment).
    """

    @abstractmethod
    async def get(self, key: str) -> Any:
        """Возвращает значение по ключу или None."""

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Сохраняет значение по ключу с опциональным TTL (секунды)."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Удаляет ключ."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Проверяет существование ключа."""

    @abstractmethod
    async def hgetall(self, key: str) -> dict:
        """Возвращает dict, если значение — dict, иначе пустой dict."""


class InMemoryStateStore(StateStore):
    """
    Реализация StateStore в памяти текущего процесса (dict + asyncio.Lock).

    Хранит только сериализуемые мета-данные; AudioSegment-объекты
    должны оставаться в AudioSession в памяти.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any:
        async with self._lock:
            return self._data.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        # TTL игнорируется в InMemory-реализации (нет фоновой очистки)
        async with self._lock:
            self._data[key] = value

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)

    async def exists(self, key: str) -> bool:
        async with self._lock:
            return key in self._data

    async def hgetall(self, key: str) -> dict:
        val = await self.get(key)
        if isinstance(val, dict):
            return val
        return {}


class RedisStateStore(StateStore):
    """
    Заглушка для Redis-реализации StateStore.

    Полный набор сигнатур готов; реализация — в будущих этапах.
    """

    async def get(self, key: str) -> Any:
        raise NotImplementedError("RedisStateStore.get is not implemented")

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        raise NotImplementedError("RedisStateStore.set is not implemented")

    async def delete(self, key: str) -> None:
        raise NotImplementedError("RedisStateStore.delete is not implemented")

    async def exists(self, key: str) -> bool:
        raise NotImplementedError("RedisStateStore.exists is not implemented")

    async def hgetall(self, key: str) -> dict:
        raise NotImplementedError("RedisStateStore.hgetall is not implemented")
