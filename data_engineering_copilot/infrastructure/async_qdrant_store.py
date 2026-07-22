"""Async Qdrant vector store implementation using qdrant_client.AsyncQdrantClient.

Provides the same interface as QdrantVectorStore but with native async/await support,
eliminating the need for ThreadPoolExecutor offloading for vector store operations.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk

logger = logging.getLogger(__name__)


class AsyncQdrantVectorStore:
    """Async wrapper around Qdrant that provides the same interface as
    ``QdrantVectorStore`` but uses ``AsyncQdrantClient`` for non-blocking IO.

    Parameters
    ----------
    url: str
        Base URL of the Qdrant HTTP API (e.g. ``http://localhost:6333``).
    collection_name: str
        Name of the collection to store/retrieve vectors.
    """

    def __init__(self, url: str, collection_name: str) -> None:
        self._url = url
        self._collection_name = collection_name
        self._client = None
        try:
            self._client = AsyncQdrantClient(url=self._url, prefer_grpc=False)
        except Exception as exc:
            logger.exception("Failed to initialise async Qdrant client: %s", exc)

    async def initialize(self) -> None:
        """Create collection and indexes if they don't exist.

        Must be called after construction and before first use.
        """
        if self._client is None:
            return
        if not await self._client.collection_exists(self._collection_name):
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self._embedding_dim(),
                    distance=models.Distance.COSINE,
                ),
            )
        try:
            await self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="url",
                field_schema="keyword",
            )
        except Exception:
            logger.debug("Payload index on 'url' already exists or could not be created.", exc_info=True)

    def _embedding_dim(self) -> int:
        return settings.embedding_dimension

    def _chunk_to_payload(self, chunk: DocumentChunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "source_name": chunk.source_name,
            "title": chunk.title,
            "url": chunk.url,
            "text": chunk.text,
            "content_hash": chunk.content_hash,
        }

    def _chunk_id_to_uuid(self, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    async def upsert_chunks(
        self,
        chunks: Iterable[DocumentChunk],
        embeddings: Iterable[Iterable[float]],
    ) -> None:
        """Insert or update a batch of chunks asynchronously."""
        if self._client is None:
            logger.warning("Qdrant client not initialized. Cannot upsert chunks.")
            return
        try:
            chunks_list = list(chunks)
            ids: list[str] = [self._chunk_id_to_uuid(chunk.chunk_id) for chunk in chunks_list]
            vectors: list[list[float]] = [list(e) for e in embeddings]
            payloads: list[dict] = [self._chunk_to_payload(chunk) for chunk in chunks_list]

            await self._client.upsert(
                collection_name=self._collection_name,
                points=models.Batch(ids=ids, vectors=vectors, payloads=payloads),
            )
        except Exception as exc:
            logger.exception("Failed to async upsert chunks to Qdrant: %s", exc)
            raise

    async def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Retrieve the most similar chunks for a query embedding asynchronously."""
        if self._client is None:
            logger.warning("Qdrant client not initialized. Returning empty results.")
            return []
        try:
            results = await self._client.query_points(
                collection_name=self._collection_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
                score_threshold=None,
            )
            retrieved: list[RetrievedChunk] = []
            points_list = results.points if hasattr(results, "points") else results
            for hit in points_list:
                payload = hit.payload or {}
                chunk = DocumentChunk(
                    chunk_id=hit.id,
                    source_name=payload.get("source_name", ""),
                    title=payload.get("title", ""),
                    url=payload.get("url", ""),
                    text=payload.get("text", ""),
                )
                distance = hit.score if isinstance(hit.score, float) else 0.0
                confidence = max(0.0, min(1.0, 1.0 - distance))
                retrieved.append(RetrievedChunk(chunk=chunk, distance=distance, confidence=confidence))
            return retrieved
        except Exception as exc:
            error_str = str(exc)
            if "404" in error_str or "Not Found" in error_str or "collection" in error_str.lower():
                logger.warning("Qdrant collection '%s' not found. Returning empty results.", self._collection_name)
                return []
            logger.exception("Failed to async query Qdrant: %s", exc)
            raise

    async def get_content_hash_for_url(self, url: str) -> str | None:
        """Retrieve stored content_hash for a given URL asynchronously."""
        if self._client is None:
            return None
        try:
            points, _ = await self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="url", match=models.MatchValue(value=url))]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if points and points[0].payload:
                return points[0].payload.get("content_hash")
            return None
        except Exception as exc:
            logger.warning("Failed to retrieve content hash for url=%s: %s", url, exc)
            return None

    async def delete_by_url(self, url: str) -> None:
        """Delete all points whose payload ``url`` field matches the given URL asynchronously."""
        if self._client is None:
            logger.warning("Qdrant client not initialized. Cannot delete by url=%s.", url)
            return
        try:
            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=[models.FieldCondition(key="url", match=models.MatchValue(value=url))])
                ),
            )
            logger.info("Deleted all points for url=%s", url)
        except Exception as exc:
            logger.warning("Failed to delete points for url=%s: %s", url, exc)

    async def count(self) -> int:
        """Return the number of points stored in the collection asynchronously."""
        if self._client is None:
            logger.warning("Qdrant client not initialized. Returning 0.")
            return 0
        try:
            collection_info = await self._client.get_collection(collection_name=self._collection_name)
            return collection_info.points_count
        except Exception as exc:
            logger.exception("Failed to get async Qdrant collection count: %s", exc)
            raise

    async def close(self) -> None:
        """Close the async client connection."""
        if self._client is not None:
            await self._client.close()
