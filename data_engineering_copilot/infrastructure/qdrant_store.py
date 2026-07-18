"""Qdrant vector store implementation.

This class mirrors the public API of ``ChromaVectorStore`` so that the rest of the
codebase can switch between the two stores without modification.

The implementation uses the official ``qdrant-client`` library.  The client is
instantiated with the URL of the Qdrant service (e.g. ``http://de_copilot_vectorstore:6333``)
and a collection name that matches the existing ``collection_name`` setting.

All methods are wrapped in ``try/except`` blocks and log errors using the
project's logging configuration.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """A thin wrapper around Qdrant that provides the same interface as
    ``QdrantVectorStore`` used elsewhere in the project.

    Parameters
    ----------
    url: str
        Base URL of the Qdrant HTTP API (e.g. ``http://de_copilot_vectorstore:6333``).
    collection_name: str
        Name of the collection to store/retrieve vectors.
    """

    def __init__(self, url: str, collection_name: str) -> None:
        self._url = url
        self._collection_name = collection_name
        self._client = None
        try:
            self._client = QdrantClient(url=self._url, prefer_grpc=False)
            # Ensure the collection exists; create if missing.
            if not self._client.collection_exists(self._collection_name):
                self._client.recreate_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(
                        size=self._embedding_dim(),
                        distance=models.Distance.COSINE,
                    ),
                )
            # Create payload index on url field for O(1) filtered lookups.
            try:
                self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name="url",
                    field_schema="keyword",
                )
            except Exception:  # pragma: no cover – index may already exist
                logger.debug(
                    "Payload index on 'url' already exists or could not be created.",
                    exc_info=True,
                )
        except Exception as exc:  # pragma: no cover – fatal init error
            logger.exception("Failed to initialise Qdrant client: %s", exc)
            raise

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #
    def _embedding_dim(self) -> int:
        """Return the dimensionality of the embedding model.

        Returns the configurable embedding_dimension from AppSettings.
        Default is 768 for the nomic-embed-text model.
        """
        return settings.embedding_dimension

    def _chunk_to_payload(self, chunk: DocumentChunk) -> dict:
        """Convert a ``DocumentChunk`` into a Qdrant payload dict."""
        return {
            "chunk_id": chunk.chunk_id,
            "source_name": chunk.source_name,
            "title": chunk.title,
            "url": chunk.url,
            "text": chunk.text,
            "content_hash": chunk.content_hash,
        }

    def _chunk_id_to_uuid(self, chunk_id: str) -> str:
        """Convert a string chunk ID to a deterministic UUID.

        Uses UUID5 (SHA-1 based) with the DNS namespace to generate a deterministic UUID
        from the chunk ID string. This ensures the same chunk ID always produces the same UUID.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    # --------------------------------------------------------------------- #
    # Public API – matches ``ChromaVectorStore``
    # --------------------------------------------------------------------- #
    def upsert_chunks(
        self,
        chunks: Iterable[DocumentChunk],
        embeddings: Iterable[Iterable[float]],
    ) -> None:
        """Insert or update a batch of chunks.

        Parameters
        ----------
        chunks: Iterable[DocumentChunk]
            The chunks to store.
        embeddings: Iterable[Iterable[float]]
            Corresponding embedding vectors; order must match ``chunks``.
        """
        # Check if client was initialized successfully
        if self._client is None:
            logger.warning(
                "Qdrant client not initialized. Cannot upsert chunks. "
                "This may be because Qdrant is not running or not reachable at %s.",
                self._url,
            )
            return
        try:
            chunks_list = list(chunks)
            ids: list[str] = [self._chunk_id_to_uuid(chunk.chunk_id) for chunk in chunks_list]
            vectors: list[list[float]] = [list(e) for e in embeddings]
            payloads: list[dict] = [self._chunk_to_payload(chunk) for chunk in chunks_list]

            self._client.upsert(
                collection_name=self._collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads,
                ),
            )
        except Exception as exc:
            logger.exception("Failed to upsert chunks to Qdrant: %s", exc)
            raise

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Retrieve the most similar chunks for a query embedding.

        Returns a list of ``RetrievedChunk`` objects sorted by similarity
        (highest confidence first).  Confidence is computed as ``1 - distance``
        because Qdrant stores cosine distance in the range ``[0, 2]``; we clamp
        the result to ``[0, 1]``.
        """
        # Check if client was initialized successfully
        if self._client is None:
            logger.warning(
                "Qdrant client not initialized. "
                "This may be because Qdrant is not running or not reachable at %s. "
                "Returning empty results.",
                self._url,
            )
            return []
        try:
            results = self._client.query_points(
                collection_name=self._collection_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
                score_threshold=None,
            )
            retrieved: list[RetrievedChunk] = []
            # QueryResponse returns a QueryResponse object with a points attribute
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
                # Qdrant returns cosine distance in [0, 2]; convert to confidence.
                distance = hit.score if isinstance(hit.score, float) else 0.0
                confidence = max(0.0, min(1.0, 1.0 - distance))
                retrieved.append(RetrievedChunk(chunk=chunk, distance=distance, confidence=confidence))
            return retrieved
        except Exception as exc:
            # Check if this is a 404 (collection not found) error
            error_str = str(exc)
            if "404" in error_str or "Not Found" in error_str or "collection" in error_str.lower():
                logger.warning(
                    "Qdrant collection '%s' not found or empty. "
                    "This may be because no ingestion has been performed yet, "
                    "or Qdrant is not running. Returning empty results.",
                    self._collection_name,
                )
                return []
            logger.exception("Failed to query Qdrant vector store: %s", exc)
            raise

    def hybrid_query(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Convenience method used by the reference RAG implementation.

        It embeds the raw query string using the project's embedding model
        and then delegates to ``query``.
        """
        try:
            embedder = SentenceTransformerEmbeddings(
                model_name=settings.embedding_model_name,
                cache_dir=settings.embedding_cache_dir,
                local_files_only=settings.embedding_local_files_only,
            )
            query_emb = embedder.embed_query(query)
            return self.query(query_emb, top_k)
        except Exception as exc:
            logger.exception("Failed to perform hybrid query: %s", exc)
            raise

    def get_content_hash_for_url(self, url: str) -> str | None:
        """Retrieve the stored content_hash for a given URL.

        Uses scroll with a payload filter on the ``url`` field.  This is efficient
        when a payload index on ``url`` exists (created at init time).

        Returns ``None`` if no chunks exist for the URL or if the client is not
        initialised.
        """
        if self._client is None:
            return None
        try:
            scroll_result = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="url",
                            match=models.MatchValue(value=url),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            points, _ = scroll_result
            if points and points[0].payload:
                return points[0].payload.get("content_hash")
            return None
        except Exception as exc:
            logger.warning("Failed to retrieve content hash for url=%s: %s", url, exc)
            return None

    def delete_by_url(self, url: str) -> None:
        """Delete all points whose payload ``url`` field matches the given URL.

        This is used to purge ghost chunks before re-indexing a page whose
        content has changed (e.g., the page shrunk from 10 chunks to 5).
        """
        if self._client is None:
            logger.warning("Qdrant client not initialized. Cannot delete by url=%s.", url)
            return
        try:
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="url",
                                match=models.MatchValue(value=url),
                            )
                        ]
                    )
                ),
            )
            logger.info("Deleted all points for url=%s", url)
        except Exception as exc:
            logger.warning("Failed to delete points for url=%s: %s", url, exc)

    def count(self) -> int:
        """Return the number of points stored in the collection."""
        # Check if client was initialized successfully
        if self._client is None:
            logger.warning(
                "Qdrant client not initialized. Cannot get collection count. "
                "This may be because Qdrant is not running or not reachable at %s.",
                self._url,
            )
            return 0
        try:
            collection_info = self._client.get_collection(collection_name=self._collection_name)
            return collection_info.points_count
        except Exception as exc:
            logger.exception("Failed to get Qdrant collection count: %s", exc)
            raise
