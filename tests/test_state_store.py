"""
Тесты для core/state_store.py (InMemoryStateStore и RedisStateStore-заглушка).
"""

import asyncio

import pytest

from core.state_store import InMemoryStateStore, RedisStateStore


class TestInMemoryStateStore:
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        store = InMemoryStateStore()
        await store.set("key1", "value1")
        assert await store.get("key1") == "value1"

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self):
        store = InMemoryStateStore()
        assert await store.get("missing") is None

    @pytest.mark.asyncio
    async def test_delete(self):
        store = InMemoryStateStore()
        await store.set("key1", "value1")
        await store.delete("key1")
        assert await store.get("key1") is None

    @pytest.mark.asyncio
    async def test_exists(self):
        store = InMemoryStateStore()
        await store.set("key1", "value1")
        assert await store.exists("key1") is True
        assert await store.exists("key2") is False

    @pytest.mark.asyncio
    async def test_hgetall_dict(self):
        store = InMemoryStateStore()
        await store.set("hash1", {"a": 1, "b": 2})
        assert await store.hgetall("hash1") == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_hgetall_non_dict(self):
        store = InMemoryStateStore()
        await store.set("key1", "string")
        assert await store.hgetall("key1") == {}

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        store = InMemoryStateStore()
        await store.set("counter", 0)
        # Инкремент в 10 корутинах
        async def inc():
            val = await store.get("counter")
            await store.set("counter", val + 1)

        await asyncio.gather(*[inc() for _ in range(10)])
        assert await store.get("counter") == 10


class TestRedisStateStore:
    @pytest.mark.asyncio
    async def test_get_raises_not_implemented(self):
        store = RedisStateStore()
        with pytest.raises(NotImplementedError):
            await store.get("key")

    @pytest.mark.asyncio
    async def test_set_raises_not_implemented(self):
        store = RedisStateStore()
        with pytest.raises(NotImplementedError):
            await store.set("key", "val")

    @pytest.mark.asyncio
    async def test_delete_raises_not_implemented(self):
        store = RedisStateStore()
        with pytest.raises(NotImplementedError):
            await store.delete("key")

    @pytest.mark.asyncio
    async def test_exists_raises_not_implemented(self):
        store = RedisStateStore()
        with pytest.raises(NotImplementedError):
            await store.exists("key")

    @pytest.mark.asyncio
    async def test_hgetall_raises_not_implemented(self):
        store = RedisStateStore()
        with pytest.raises(NotImplementedError):
            await store.hgetall("key")
