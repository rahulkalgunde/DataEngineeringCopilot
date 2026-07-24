"""Redis-backed two-tier query cache for production use.

Tier 1: Exact match (SHA-256 of normalized query -> answer)
Tier 2: Semantic similarity (embedding cosine -> answer)

Unlike in-memory caches, this survives process restarts and can be
shared across multiple API instances.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time

import numpy as np

logger = logging.getLogger(__name__)


class RedisQueryCache:
    """Production-grade query cache backed by Redis.

    Provides exact-match and semantic-similarity caching with TTL expiry.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        exact_ttl: int = 3600,
        semantic_ttl: int = 7200,
        similarity_threshold: float = 0.92,
        max_semantic_entries: int = 10000,
    ) -> None:
        import redis.asyncio as aioredis

        self._redis = aioredis.from_url(redis_url, decode_responses=True)
        self._exact_ttl = exact_ttl
        self._semantic_ttl = semantic_ttl
        self._similarity_threshold = similarity_threshold
        self._max_semantic = max_semantic_entries
        self._prefix = "rag:cache"

    def _normalize(self, query: str) -> str:
        q = query.lower().strip()
        q = re.sub(r"[^\w\s]", "", q)
        q = re.sub(r"\s+", " ", q)
        return q

    def _exact_key(self, query: str) -> str:
        normalized = self._normalize(query)
        h = hashlib.sha256(normalized.encode()).hexdigest()
        return f"{self._prefix}:exact:{h}"

    def _semantic_key(self, idx: int) -> str:
        return f"{self._prefix}:semantic:{idx}"

    async def get_exact(self, query: str) -> str | None:
        key = self._exact_key(query)
        cached = await self._redis.get(key)
        return cached if cached else None

    async def set_exact(self, query: str, answer: str) -> None:
        key = self._exact_key(query)
        await self._redis.setex(key, self._exact_ttl, answer)

    async def get_semantic(self, query_embedding: list[float]) -> str | None:
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return None

        best_score = -1.0
        best_answer = None

        cursor = 0
        while True:
            cursor, keys = await self._redis.scan(
                cursor=cursor, match=f"{self._prefix}:semantic:*", count=100
            )
            for key in keys:
                data = await self._redis.hgetall(key)
                if "embedding" not in data or "answer" not in data:
                    continue
                stored_vec = np.array(json.loads(data["embedding"]), dtype=np.float32)
                stored_norm = np.linalg.norm(stored_vec)
                if stored_norm == 0:
                    continue
                score = float(np.dot(query_vec, stored_vec) / (query_norm * stored_norm))
                if score > best_score:
                    best_score = score
                    best_answer = data["answer"]
            if cursor == 0:
                break

        if best_score >= self._similarity_threshold and best_answer:
            return best_answer
        return None

    async def set_semantic(self, query: str, query_embedding: list[float], answer: str) -> None:
        idx = int(await self._redis.incr(f"{self._prefix}:semantic:counter"))
        if idx > self._max_semantic:
            oldest = idx - self._max_semantic
            await self._redis.delete(self._semantic_key(oldest))
        key = self._semantic_key(idx)
        await self._redis.hset(
            key,
            mapping={
                "query": query,
                "embedding": json.dumps(query_embedding),
                "answer": answer,
                "created_at": str(time.time()),
            },
        )
        await self._redis.expire(key, self._semantic_ttl)

    async def get(self, query: str, query_embedding: list[float] | None = None) -> str | None:
        """Get cached answer: try exact first, then semantic."""
        exact = await self.get_exact(query)
        if exact:
            return exact
        if query_embedding is not None:
            return await self.get_semantic(query_embedding)
        return None

    async def set(self, query: str, answer: str, query_embedding: list[float] | None = None) -> None:
        """Cache answer in both tiers."""
        await self.set_exact(query, answer)
        if query_embedding is not None:
            await self.set_semantic(query, query_embedding, answer)

    async def close(self) -> None:
        await self._redis.close()
