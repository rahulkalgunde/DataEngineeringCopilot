# Qdrant vector store alias for compatibility

from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore
from typing import List, Iterable
from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.config.settings import settings
import logging

logger = logging.getLogger(__name__)

# Qdrant-backed vector store adapter (replaces legacy ChromaVectorStore)
"""Compatibility layer that maps the historic ``ChromaVectorStore`` API to the new Qdrant implementation.

The rest of the codebase imports ``ChromaVectorStore`` from
``data_engineering_copilot.infrastructure.vector_store``.  To avoid widespread
import changes we keep the class name but delegate all operations to
``QdrantVectorStore`` which is defined in ``qdrant_store.py``.

The adapter uses the ``settings`` object to obtain the Qdrant HTTP endpoint and
collection name.  The ``persist_directory`` argument is accepted for backward
compatibility but is ignored because Qdrant stores its data in the Docker container.
"""

class VectorStoreReadError(RuntimeError):
    """Raised when the vector store cannot be read (e.g., missing collection)."""
    pass


class ChromaVectorStore:
    """Legacy name retained for compatibility; forwards to :class:`QdrantVectorStore`.

    Parameters
    ----------
    persist_directory: str
        Ignored – kept only to match the historical signature.
    collection_name: str
        Name of the Qdrant collection used for storage.
    """

    def __init__(self, persist_directory: str, collection_name: str) -> None:
        # ``persist_directory`` is unused; the Qdrant service runs in Docker.
        self._store = QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=collection_name,
        )
        logger.debug(
            "Initialized QdrantVectorStore via ChromaVectorStore adapter: %s", collection_name
        )

    # ---------------------------------------------------------------------
    # Public API – mirrors the original ChromaVectorStore methods
    # ---------------------------------------------------------------------
    def upsert_chunks(
        self,
        chunks: Iterable[DocumentChunk],
        embeddings: Iterable[Iterable[float]],
    ) -> None:
        self._store.upsert_chunks(chunks, embeddings)

    def query(self, query_embedding: List[float], top_k: int) -> List[RetrievedChunk]:
        return self._store.query(query_embedding, top_k)

    def count(self) -> int:
        return self._store.count()

    # Optional hybrid query used by the reference RAG implementation
    def hybrid_query(self, query: str, top_k: int) -> List[RetrievedChunk]:
        return self._store.hybrid_query(query, top_k)
