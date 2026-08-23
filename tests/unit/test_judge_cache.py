"""Tests for the offline judge-verdict cache (fail-open)."""

import hashlib

from data_engineering_copilot.evaluation.judge_cache import JudgeCache, judge_cache_key


class _MemClient:
    def __init__(self):
        self.store = {}

    async def get(self, k):
        return self.store.get(k)

    async def setex(self, k, ttl, v):
        self.store[k] = v


class _BrokenClient:
    async def get(self, k):
        raise ConnectionError("down")

    async def setex(self, k, ttl, v):
        raise ConnectionError("down")


def test_key_is_stable_sha256_prefixed():
    k = judge_cache_key("m", "p1", "q", "a", "c")
    raw = hashlib.sha256(b"m\x00p1\x00q\x00a\x00c").hexdigest()
    assert k == f"dec:evaljudge:{raw}"


def test_key_changes_when_any_field_changes():
    base = judge_cache_key("m", "p1", "q", "a", "c")
    assert base != judge_cache_key("m2", "p1", "q", "a", "c")
    assert base != judge_cache_key("m", "p2", "q", "a", "c")
    assert base != judge_cache_key("m", "p1", "q2", "a", "c")
    assert base != judge_cache_key("m", "p1", "q", "a2", "c")
    assert base != judge_cache_key("m", "p1", "q", "a", "c2")


async def test_put_then_get_roundtrip():
    c = JudgeCache(enabled=True, ttl_days=30, client=_MemClient())
    k = judge_cache_key("m", "p", "q", "a", "ctx")
    await c.put(k, {"score": 0.87})
    assert await c.get(k) == {"score": 0.87}


async def test_miss_returns_none():
    c = JudgeCache(enabled=True, ttl_days=30, client=_MemClient())
    assert await c.get(judge_cache_key("m", "p", "q", "a", "nope")) is None


async def test_disabled_short_circuits():
    mem = _MemClient()
    c = JudgeCache(enabled=False, ttl_days=30, client=mem)
    k = judge_cache_key("m", "p", "q", "a", "x")
    await c.put(k, {"score": 1.0})
    assert await c.get(k) is None
    assert mem.store == {}


async def test_fail_open_on_redis_error():
    c = JudgeCache(enabled=True, ttl_days=30, client=_BrokenClient())
    k = judge_cache_key("m", "p", "q", "a", "y")
    assert await c.get(k) is None  # no raise
    await c.put(k, {"score": 1.0})  # no raise
