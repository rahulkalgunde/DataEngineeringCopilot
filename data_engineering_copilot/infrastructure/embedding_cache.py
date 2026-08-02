from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_engineering_copilot.domain.protocols import EmbedderProtocol

logger = logging.getLogger(__name__)

_EMBED_CACHE_TTL = 30 * 24 * 3600  # 30 days


class EmbeddingCache:
    """LRU cache mapping normalized text to embedding vectors.

    Avoids redundant embedder API calls when the same text is
    embedded multiple times (e.g. query rewriting, cache lookups).
    """

    def __init__(self, max_size: int = 1024) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.lower().strip().encode("utf-8")).hexdigest()

    def get(self, text: str) -> list[float] | None:
        key = self._key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, text: str, embedding: list[float]) -> None:
        key = self._key(text)
        self._cache[key] = embedding
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


class CachedEmbedder:
    """Wraps an ``EmbedderProtocol`` with L1 (memory) + L2 (Redis) embedding cache.

    Checks L1 first, then Redis, before delegating to the underlying embedder.
    Cache TTL is 30 days. Supports both ``embed_query`` and ``embed_texts``.
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        max_size: int = 1024,
        redis_client=None,
    ) -> None:
        self._inner = embedder
        self._cache = EmbeddingCache(max_size=max_size)
        self._redis = redis_client

    def _redis_key(self, text: str) -> str:
        key = self._cache._key(text)
        return f"embed:cache:{key}"

    async def _get_from_redis(self, text: str) -> list[float] | None:
        if self._redis is None:
            return None
        try:
            val = await self._redis.get(self._redis_key(text))
            if val is not None:
                raw = val if isinstance(val, str) else val.decode()
                return json.loads(raw)
        except Exception:
            logger.debug("Redis embed cache get failed", exc_info=True)
        return None

    async def _set_to_redis(self, text: str, embedding: list[float]) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.set(
                self._redis_key(text),
                json.dumps(embedding),
                ex=_EMBED_CACHE_TTL,
            )
        except Exception:
            logger.debug("Redis embed cache set failed", exc_info=True)

    async def embed_query(self, text: str) -> list[float]:
        # L1 check
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        # L2 check (Redis)
        cached = await self._get_from_redis(text)
        if cached is not None:
            self._cache.set(text, cached)
            return cached
        # Cache miss — call embedder
        emb = await self._inner.embed_query(text)
        self._cache.set(text, emb)
        await self._set_to_redis(text, emb)
        return emb

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Check cache for each text; collect uncached indices
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        for i, text in enumerate(texts):
            # L1
            cached = self._cache.get(text)
            if cached is not None:
                results[i] = cached
                continue
            # L2
            cached = await self._get_from_redis(text)
            if cached is not None:
                self._cache.set(text, cached)
                results[i] = cached
                continue
            uncached_indices.append(i)

        # Embed only uncached texts
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            fresh_embeddings = await self._inner.embed_texts(uncached_texts)
            for idx, emb in zip(uncached_indices, fresh_embeddings, strict=False):
                results[idx] = emb
                self._cache.set(texts[idx], emb)
                await self._set_to_redis(texts[idx], emb)

        return [r if r is not None else [] for r in results]

    async def close(self) -> None:
        if hasattr(self._inner, "close"):
            await self._inner.close()
