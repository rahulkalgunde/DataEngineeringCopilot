"""Two-tier query cache: exact-match (SHA-256) + semantic similarity.

Hybrid: in-memory L1 (fast) + optional Redis L2 (persistent, shared).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import OrderedDict, deque

import numpy as np

logger = logging.getLogger(__name__)


def _normalize_query(query: str) -> str:
    q = query.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


class QueryCache:
    """Two-tier query cache with optional Redis L2 persistence.

    Tier 1 (L1): in-memory exact match via SHA-256 (LRU eviction).
    Tier 2 (L2): in-memory semantic similarity via NumPy batch dot product.
    When ``redis_url`` is provided, L2 also persists to Redis and survives
    restarts; L1 is warmed from Redis on first access.
    """

    def __init__(
        self,
        exact_enabled: bool = True,
        semantic_enabled: bool = True,
        similarity_threshold: float = 0.92,
        exact_max_size: int = 1024,
        semantic_max_size: int = 512,
        ttl_seconds: int = 3600,
        redis_url: str | None = None,
    ) -> None:
        self._exact_enabled = exact_enabled
        self._semantic_enabled = semantic_enabled
        self._similarity_threshold = similarity_threshold
        self._exact_cache: OrderedDict[str, str] = OrderedDict()
        self._semantic_cache: deque[tuple[np.ndarray, str, str, float]] = deque()
        self._exact_max_size = exact_max_size
        self._semantic_max_size = semantic_max_size
        self._ttl_seconds = ttl_seconds
        self._redis = None
        if redis_url:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(redis_url, decode_responses=True)

    def _exact_key(self, query: str) -> str:
        normalized = _normalize_query(query)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # --- exact tier ---

    def get_exact(self, query: str) -> str | None:
        if not self._exact_enabled:
            return None
        key = self._exact_key(query)
        return self._exact_cache.get(key)

    def set_exact(self, query: str, answer: str) -> None:
        if not self._exact_enabled:
            return
        key = self._exact_key(query)
        self._exact_cache[key] = answer
        if len(self._exact_cache) > self._exact_max_size:
            self._exact_cache.popitem(last=False)

    # --- semantic tier ---

    def get_semantic(self, query: str, query_embedding: list[float]) -> str | None:
        if not self._semantic_enabled or not self._semantic_cache or not query_embedding:
            return None

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return None
        q_unit = q_vec / q_norm

        now = time.monotonic()
        cutoff = now - self._ttl_seconds
        valid: list[tuple[np.ndarray, str, str, float]] = []
        while self._semantic_cache and self._semantic_cache[0][3] < cutoff:
            self._semantic_cache.popleft()
        valid = list(self._semantic_cache)

        if not valid:
            return None

        matrix = np.vstack([entry[0] for entry in valid])
        similarities = np.dot(matrix, q_unit)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self._similarity_threshold:
            return valid[best_idx][2]
        return None

    def set_semantic(self, query: str, query_embedding: list[float], answer: str) -> None:
        if not self._semantic_enabled or not query_embedding:
            return

        vec = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return
        unit_vec = vec / norm

        self._semantic_cache.append((unit_vec, query, answer, time.monotonic()))
        if len(self._semantic_cache) > self._semantic_max_size:
            self._semantic_cache.popleft()

    # --- async L2 (Redis) support ---

    async def aget(self, query: str, query_embedding: list[float] | None = None) -> str | None:
        exact = self.get_exact(query)
        if exact is not None:
            return exact
        if query_embedding is not None:
            seen = self.get_semantic(query, query_embedding)
            if seen is not None:
                return seen
        if self._redis is None:
            return None
        try:
            val = await self._redis.get(f"rag:cache:exact:{self._exact_key(query)}")
            if val is not None:
                cached = val if isinstance(val, str) else val.decode()
                self.set_exact(query, cached)
                return cached
        except Exception:
            logger.warning("Redis L2 get_exact failed", exc_info=True)
        if query_embedding is None:
            return None
        try:
            cursor = 0
            query_vec = np.array(query_embedding, dtype=np.float32)
            q_norm = np.linalg.norm(query_vec)
            if q_norm > 0:
                best_score = -1.0
                best_answer = None
                while True:
                    cursor, keys = await self._redis.scan(cursor=cursor, match="rag:cache:semantic:*", count=100)
                    for key in keys:
                        data = await self._redis.hgetall(key)
                        emb_raw = data.get("embedding")
                        ans_raw = data.get("answer")
                        if not emb_raw or not ans_raw:
                            continue
                        stored = np.array(
                            json.loads(emb_raw if isinstance(emb_raw, str) else emb_raw.decode()), dtype=np.float32
                        )
                        s_norm = np.linalg.norm(stored)
                        if s_norm == 0:
                            continue
                        score = float(np.dot(query_vec, stored) / (q_norm * s_norm))
                        if score > best_score:
                            best_score = score
                            best_answer = ans_raw if isinstance(ans_raw, str) else ans_raw.decode()
                    if cursor == 0:
                        break
                if best_score >= self._similarity_threshold and best_answer:
                    self.set_semantic(query, query_embedding, best_answer)
                    return best_answer
        except Exception:
            logger.warning("Redis L2 get_semantic failed", exc_info=True)
        return None

    async def aset_exact(self, query: str, answer: str) -> None:
        self.set_exact(query, answer)
        if self._redis is not None:
            try:
                await self._redis.setex(f"rag:cache:exact:{self._exact_key(query)}", self._ttl_seconds, answer)
            except Exception:
                logger.warning("Redis L2 set_exact failed", exc_info=True)

    async def aset_semantic(self, query: str, query_embedding: list[float], answer: str) -> None:
        self.set_semantic(query, query_embedding, answer)
        if self._redis is not None:
            try:
                idx = await self._redis.incr("rag:cache:semantic:counter")
                key = f"rag:cache:semantic:{idx}"
                await self._redis.hset(
                    key,
                    mapping={
                        "query": query,
                        "embedding": json.dumps(query_embedding),
                        "answer": answer,
                        "created_at": str(time.time()),
                    },
                )
                await self._redis.expire(key, self._ttl_seconds * 2)
            except Exception:
                logger.warning("Redis L2 set_semantic failed", exc_info=True)

    async def aclose(self) -> None:
        if self._redis is not None:
            await self._redis.close()

    # --- combined ---

    def get(self, query: str, query_embedding: list[float] | None = None) -> str | None:
        exact = self.get_exact(query)
        if exact is not None:
            return exact
        if query_embedding is not None:
            return self.get_semantic(query, query_embedding)
        return None
