from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_engineering_copilot.domain.protocols import EmbedderProtocol

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """LRU cache mapping normalized query text to embedding vectors.

    Avoids redundant embedder API calls when the same query text is
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
    """Wraps an ``EmbedderProtocol`` with an LRU embedding cache.

    Checks the cache before delegating to the underlying embedder.
    """

    def __init__(self, embedder: EmbedderProtocol, max_size: int = 1024) -> None:
        self._inner = embedder
        self._cache = EmbeddingCache(max_size=max_size)

    async def embed_query(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        emb = await self._inner.embed_query(text)
        self._cache.set(text, emb)
        return emb

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self._inner.embed_texts(texts)

    async def close(self) -> None:
        if hasattr(self._inner, "close"):
            await self._inner.close()
