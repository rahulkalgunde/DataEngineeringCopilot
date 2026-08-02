"""Two-tier query cache: exact-match (SHA-256) + semantic similarity.

Hybrid: in-memory L1 (fast) + optional Redis L2 (persistent, shared).

Every entry is scoped to a :class:`CacheScope` (tenant, role, source filter,
embedding model, collection) whose fingerprint is embedded in the key, so
cross-tenant / cross-filter leakage is structurally impossible.  Values are
:class:`CachedAnswer` envelopes (text + sources + confidence + groundedness)
so a hit reconstructs the full ``Answer`` instead of a fabricated one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import OrderedDict, deque
from dataclasses import asdict

import numpy as np

from data_engineering_copilot.domain.models import CachedAnswer, CacheScope, DocumentChunk

logger = logging.getLogger(__name__)

# Bumped whenever the stored value schema or key layout changes. Old entries
# become unreachable without an explicit flush.
CACHE_SCHEMA_VERSION = "v2"


def _normalize_query(query: str) -> str:
    q = query.lower().strip()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


def scope_fingerprint(scope: CacheScope | None) -> str:
    """16-hex fingerprint of a scope's canonical JSON + schema version."""
    scope = scope or CacheScope()
    payload = {
        "tenant_id": scope.tenant_id,
        "role": scope.role,
        "source_filter": sorted(scope.source_filter),
        "embedding_model": scope.embedding_model,
        "collection_name": scope.collection_name,
        "schema": CACHE_SCHEMA_VERSION,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


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
        redis_client=None,
    ) -> None:
        self._exact_enabled = exact_enabled
        self._semantic_enabled = semantic_enabled
        self._similarity_threshold = similarity_threshold
        self._exact_cache: OrderedDict[str, CachedAnswer] = OrderedDict()
        self._semantic_cache: deque[tuple[str, np.ndarray, str, CachedAnswer, float]] = deque()
        self._exact_max_size = exact_max_size
        self._semantic_max_size = semantic_max_size
        self._ttl_seconds = ttl_seconds
        self._redis = redis_client
        self._owns_redis = redis_client is None
        if self._redis is None and redis_url:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(redis_url, decode_responses=True)

    def _exact_key(self, query: str, scope: CacheScope | None = None) -> str:
        fp = scope_fingerprint(scope)
        normalized = _normalize_query(query)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"{fp}:{digest}"

    def _redis_exact_key(self, query: str, scope: CacheScope | None = None) -> str:
        return f"rag:cache:exact:{self._exact_key(query, scope)}"

    def _redis_semantic_namespace(self, scope: CacheScope | None = None) -> str:
        fp = scope_fingerprint(scope)
        return f"rag:cache:semantic:{fp}"

    # --- exact tier ---

    def get_exact(self, query: str, scope: CacheScope | None = None) -> CachedAnswer | None:
        if not self._exact_enabled:
            return None
        key = self._exact_key(query, scope)
        return self._exact_cache.get(key)

    def set_exact(self, query: str, answer: CachedAnswer, scope: CacheScope | None = None) -> None:
        if not self._exact_enabled:
            return
        key = self._exact_key(query, scope)
        self._exact_cache[key] = answer
        if len(self._exact_cache) > self._exact_max_size:
            self._exact_cache.popitem(last=False)

    # --- semantic tier ---

    def get_semantic(
        self, query: str, query_embedding: list[float], scope: CacheScope | None = None
    ) -> CachedAnswer | None:
        if not self._semantic_enabled or not self._semantic_cache or not query_embedding:
            return None

        fp = scope_fingerprint(scope)
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return None
        q_unit = q_vec / q_norm

        now = time.monotonic()
        cutoff = now - self._ttl_seconds
        valid: list[tuple[str, np.ndarray, str, CachedAnswer, float]] = []
        while self._semantic_cache and self._semantic_cache[0][4] < cutoff:
            self._semantic_cache.popleft()
        valid = [e for e in self._semantic_cache if e[0] == fp]

        if not valid:
            return None

        matrix = np.vstack([entry[1] for entry in valid])
        similarities = np.dot(matrix, q_unit)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self._similarity_threshold:
            return valid[best_idx][3]
        return None

    def set_semantic(
        self, query: str, query_embedding: list[float], answer: CachedAnswer, scope: CacheScope | None = None
    ) -> None:
        if not self._semantic_enabled or not query_embedding:
            return

        fp = scope_fingerprint(scope)
        vec = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return
        unit_vec = vec / norm

        self._semantic_cache.append((fp, unit_vec, query, answer, time.monotonic()))
        if len(self._semantic_cache) > self._semantic_max_size:
            self._semantic_cache.popleft()

    # --- async L2 (Redis) support ---

    async def aget(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        scope: CacheScope | None = None,
    ) -> CachedAnswer | None:
        exact = self.get_exact(query, scope)
        if exact is not None:
            return exact
        if query_embedding is not None:
            seen = self.get_semantic(query, query_embedding, scope)
            if seen is not None:
                return seen
        if self._redis is None:
            return None
        try:
            val = await self._redis.get(self._redis_exact_key(query, scope))
            if val is not None:
                cached = val if isinstance(val, str) else val.decode()
                envelope = _deserialize_envelope(cached)
                if envelope is not None:
                    self.set_exact(query, envelope, scope)
                    return envelope
        except Exception:
            logger.warning("Redis L2 get_exact failed", exc_info=True)
        if query_embedding is None:
            return None
        try:
            namespace = self._redis_semantic_namespace(scope)
            cursor = 0
            query_vec = np.array(query_embedding, dtype=np.float32)
            q_norm = np.linalg.norm(query_vec)
            if q_norm > 0:
                best_score = -1.0
                best_envelope = None
                while True:
                    cursor, keys = await self._redis.scan(cursor=cursor, match=f"{namespace}:*", count=100)
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
                            envelope = _deserialize_envelope(ans_raw if isinstance(ans_raw, str) else ans_raw.decode())
                            if envelope is not None:
                                best_envelope = envelope
                    if cursor == 0:
                        break
                if best_score >= self._similarity_threshold and best_envelope is not None:
                    self.set_semantic(query, query_embedding, best_envelope, scope)
                    return best_envelope
        except Exception:
            logger.warning("Redis L2 get_semantic failed", exc_info=True)
        return None

    async def aset_exact(self, query: str, answer: CachedAnswer, scope: CacheScope | None = None) -> None:
        self.set_exact(query, answer, scope)
        if self._redis is not None:
            try:
                await self._redis.setex(
                    self._redis_exact_key(query, scope),
                    self._ttl_seconds,
                    _serialize_envelope(answer),
                )
            except Exception:
                logger.warning("Redis L2 set_exact failed", exc_info=True)

    async def aset_semantic(
        self,
        query: str,
        query_embedding: list[float],
        answer: CachedAnswer,
        scope: CacheScope | None = None,
    ) -> None:
        self.set_semantic(query, query_embedding, answer, scope)
        if self._redis is not None:
            try:
                idx = await self._redis.incr("rag:cache:semantic:counter")
                namespace = self._redis_semantic_namespace(scope)
                key = f"{namespace}:{idx}"
                await self._redis.hset(
                    key,
                    mapping={
                        "query": query,
                        "embedding": json.dumps(query_embedding),
                        "answer": _serialize_envelope(answer),
                        "created_at": str(time.time()),
                    },
                )
                await self._redis.expire(key, self._ttl_seconds * 2)
            except Exception:
                logger.warning("Redis L2 set_semantic failed", exc_info=True)

    async def aclose(self) -> None:
        # Only close the client if this instance owns it. A shared client is
        # closed once at process shutdown.
        if self._redis is not None and self._owns_redis:
            await self._redis.close()

    # --- combined ---

    def get(
        self,
        query: str,
        query_embedding: list[float] | None = None,
        scope: CacheScope | None = None,
    ) -> CachedAnswer | None:
        exact = self.get_exact(query, scope)
        if exact is not None:
            return exact
        if query_embedding is not None:
            return self.get_semantic(query, query_embedding, scope)
        return None


def _serialize_envelope(answer: CachedAnswer) -> str:
    return json.dumps(
        {
            "text": answer.text,
            "sources": [asdict(c) for c in answer.sources],
            "confidence": answer.confidence,
            "groundedness_score": answer.groundedness_score,
            "cached_at": answer.cached_at,
        }
    )


def _deserialize_envelope(raw: str) -> CachedAnswer | None:
    try:
        data = json.loads(raw)
        sources = tuple(DocumentChunk(**s) for s in data.get("sources", []))
        return CachedAnswer(
            text=data["text"],
            sources=sources,
            confidence=float(data.get("confidence", 1.0)),
            groundedness_score=float(data.get("groundedness_score", 1.0)),
            cached_at=float(data.get("cached_at", 0.0)),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.warning("Stored cache value is not a valid CachedAnswer envelope")
        return None
