"""Redis-backed verdict cache for OFFLINE judge calls (fail-open).

Posture: auxiliary/offline — any storage error degrades to a cache miss and a
single logged warning; evaluation always proceeds by computing fresh judgments.
Only used by eval harnesses, never by the live answer path. The shared Redis
client is async (aioredis), so get/put are coroutines.
"""

from __future__ import annotations

import hashlib
import json
import logging

logger = logging.getLogger(__name__)

_PREFIX = "dec:evaljudge:"
_WARNED = {"flag": False}


def judge_cache_key(model_id: str, prompt_version: str, question: str, answer: str, context: str) -> str:
    blob = "\x00".join((model_id, prompt_version, question, answer, context))
    return _PREFIX + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _warn_once(msg: str) -> None:
    if not _WARNED["flag"]:
        _WARNED["flag"] = True
        logger.warning("%s", msg)


class JudgeCache:
    """Async Redis verdict cache. Fail-open: errors => miss + no store."""

    def __init__(self, *, enabled: bool, ttl_days: int, client=None) -> None:
        self._enabled = enabled and client is not None
        self._ttl_seconds = max(1, int(ttl_days)) * 86400
        self._client = client

    async def get(self, key: str) -> dict | None:
        if not self._enabled or self._client is None:
            return None
        try:
            raw = await self._client.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:  # noqa: BLE001 - fail-open contract
            _warn_once(f"judge cache unavailable, continuing without: {exc}")
            return None

    async def put(self, key: str, value: dict) -> None:
        if not self._enabled or self._client is None:
            return
        try:
            await self._client.setex(key, self._ttl_seconds, json.dumps(value))
        except Exception as exc:  # noqa: BLE001 - fail-open contract
            _warn_once(f"judge cache unavailable, continuing without: {exc}")
