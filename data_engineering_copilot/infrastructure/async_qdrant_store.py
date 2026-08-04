"""Async Qdrant vector store implementation with optional hybrid search.

Dense-only mode: cosine similarity on embedding vectors (default behaviour).
Hybrid mode:    adds BM25 sparse vectors and uses Qdrant native RRF fusion
                at query time for combined dense + sparse retrieval.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Self, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import SparseIndexParams, SparseVectorParams

from data_engineering_copilot.config.settings import PROJECT_ROOT, settings
from data_engineering_copilot.domain.exceptions import VectorStoreError
from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.infrastructure.bm25_tokenizer import BM25Tokenizer

logger = logging.getLogger(__name__)


class AsyncQdrantVectorStore:
    """Async wrapper around Qdrant with optional BM25 hybrid search.

    Parameters
    ----------
    url: str
        Base URL of the Qdrant HTTP API (e.g. ``http://localhost:6333``).
    collection_name: str
        Name of the collection to store/retrieve vectors.
    hybrid_search: bool
        When True (default), upsert sparse BM25 vectors alongside dense
        embeddings and use Qdrant native RRF fusion at query time.
    """

    def __init__(
        self,
        url: str,
        collection_name: str,
        hybrid_search: bool = True,
        hybrid_rrf_k: int = 60,
        embedding_dimension: int | None = None,
        bm25_persist_path: Path | None = None,
    ) -> None:
        self._url = url
        self._collection_name = collection_name
        self._hybrid_search = hybrid_search
        self._hybrid_rrf_k = hybrid_rrf_k
        self._bm25_persist_path = bm25_persist_path or (PROJECT_ROOT / ".bm25_cache" / f"{collection_name}.json")
        self._bm25 = None
        self._bm25_loaded_from_disk = False
        if hybrid_search:
            if self._bm25_persist_path.exists():
                try:
                    self._bm25 = BM25Tokenizer.load(self._bm25_persist_path)
                    self._bm25_loaded_from_disk = True
                    logger.info("Loaded persisted BM25 tokenizer from %s", self._bm25_persist_path)
                except Exception:
                    logger.warning("Failed to load BM25 from %s, creating fresh", self._bm25_persist_path)
                    self._bm25 = BM25Tokenizer()
            else:
                self._bm25 = BM25Tokenizer()
        self._client = AsyncQdrantClient(url=self._url, prefer_grpc=False)
        self._last_query_sparse = None
        self._embedding_dimension_override = embedding_dimension
        self._bm25_desync_warned = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def initialize(self) -> None:
        """Create collection and indexes if they don't exist.

        Must be called after construction and before first use.
        """
        if self._client is None:
            raise VectorStoreError("Qdrant client not initialized.")
        if not await self._client.collection_exists(self._collection_name):
            if self._hybrid_search:
                vectors_config = {
                    "dense": models.VectorParams(
                        size=self._embedding_dim(),
                        distance=models.Distance.COSINE,
                    ),
                }
                sparse_vectors_config = {"sparse": SparseVectorParams(index=SparseIndexParams())}
            else:
                vectors_config = models.VectorParams(
                    size=self._embedding_dim(),
                    distance=models.Distance.COSINE,
                )
                sparse_vectors_config = None

            create_kwargs: dict = dict(
                collection_name=self._collection_name,
                vectors_config=vectors_config,
                on_disk_payload=True,
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=150, full_scan_threshold=10000),
            )
            if sparse_vectors_config is not None:
                create_kwargs["sparse_vectors_config"] = sparse_vectors_config

            await self._client.create_collection(**create_kwargs)
        try:
            await self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="url",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            logger.info("Payload index on 'url' already exists or could not be created.", exc_info=True)
        try:
            await self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="source_name",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            logger.info("Payload index on 'source_name' already exists or could not be created.", exc_info=True)
        try:
            await self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="chunk_type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            logger.info("Payload index on 'chunk_type' already exists or could not be created.", exc_info=True)
        try:
            await self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="section_header",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            logger.info("Payload index on 'section_header' already exists or could not be created.", exc_info=True)
        try:
            await self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="crawled_at",
                field_schema=models.PayloadSchemaType.DATETIME,
            )
        except Exception:
            logger.info("Payload index on 'crawled_at' already exists or could not be created.", exc_info=True)

    def _embedding_dim(self) -> int:
        if self._embedding_dimension_override is not None:
            return self._embedding_dimension_override
        return settings.get_embedding_dimension()

    def _chunk_to_payload(self, chunk: DocumentChunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "source_name": chunk.source_name,
            "title": chunk.title,
            "url": chunk.url,
            "text": chunk.text,
            "content_hash": chunk.content_hash,
            "section_header": chunk.section_header,
            "chunk_type": chunk.chunk_type,
            "word_count": chunk.word_count,
            "heading_path": list(chunk.heading_path),
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "crawled_at": chunk.crawled_at,
        }

    def _chunk_id_to_uuid(self, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    async def upsert_chunks(
        self,
        chunks: Iterable[DocumentChunk],
        vectors: Iterable[list[float]],
        _sub_batch_size: int = 256,
    ) -> None:
        """Insert or update a batch of chunks asynchronously.

        Splits the input into sub-batches of ``_sub_batch_size`` to stay
        within Qdrant's ``max_request_size_mb`` (default 32 MB).
        """
        if self._client is None:
            logger.warning("Qdrant client not initialized. Cannot upsert chunks.")
            return
        chunks_list = list(chunks)
        vectors_list = list(vectors)
        if not chunks_list:
            return

        for i in range(0, len(chunks_list), _sub_batch_size):
            sub_chunks = chunks_list[i : i + _sub_batch_size]
            sub_embeddings = vectors_list[i : i + _sub_batch_size]
            try:
                ids = [self._chunk_id_to_uuid(chunk.chunk_id) for chunk in sub_chunks]
                vectors = [list(e) for e in sub_embeddings]
                payloads = [self._chunk_to_payload(chunk) for chunk in sub_chunks]

                if self._hybrid_search and self._bm25 is not None:
                    await self._warn_unfrozen_bm25_desync()
                    sparse_vectors_list = [self._bm25.tokenize_query(c.text) for c in sub_chunks]
                    vectors_dict = {"dense": vectors, "sparse": sparse_vectors_list}
                else:
                    vectors_dict = vectors

                await self._client.upsert(
                    collection_name=self._collection_name,
                    points=models.Batch(ids=[str(i) for i in ids], vectors=vectors_dict, payloads=payloads),  # type: ignore[arg-type]  # qdrant ExtendedPointId union incompatible with pydantic-v1 models
                )
            except Exception as exc:
                logger.exception(
                    "Failed to async upsert chunks to Qdrant (sub-batch %d/%d): %s",
                    i // _sub_batch_size + 1,
                    (len(chunks_list) + _sub_batch_size - 1) // _sub_batch_size,
                    exc,
                )
                raise

    async def query(
        self,
        query_embedding: list[float],
        top_k: int,
        query_text: str | None = None,
        source_filter: list[str] | None = None,
        chunk_type_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most similar chunks for a query embedding asynchronously.

        When ``hybrid_search=True`` and the BM25 tokenizer has been fitted,
        uses Qdrant native prefetch + RRF fusion over dense and sparse vectors.
        Otherwise falls back to pure dense cosine search.

        Parameters
        ----------
        query_text:
            Optional raw query string.  When provided and hybrid search is active,
            the BM25 sparse vector is computed internally (eliminates the need for
            ``set_query_sparse``).
        source_filter:
            Optional list of source names to filter results by.
            Applied at the Qdrant query level for efficiency.
        chunk_type_filter:
            Optional chunk type to filter results by (e.g. "api", "code", "text").
            Applied at the Qdrant query level for efficiency.
        """
        if self._client is None:
            logger.warning("Qdrant client not initialized. Returning empty results.")
            return []

        use_hybrid = self._hybrid_search and self._bm25 is not None and self._bm25._frozen

        # Build Qdrant filter for source names and chunk type
        query_filter = None
        filter_conditions = []
        if source_filter is not None:
            if not source_filter:
                raise VectorStoreError("source_filter must not be empty; use None for 'no filter'")
            filter_conditions.append(
                models.FieldCondition(
                    key="source_name",
                    match=models.MatchAny(any=source_filter),
                )
            )
        if chunk_type_filter:
            filter_conditions.append(
                models.FieldCondition(
                    key="chunk_type",
                    match=models.MatchValue(value=chunk_type_filter),
                )
            )
        if filter_conditions:
            query_filter = models.Filter(must=filter_conditions)

        rrf_confidence_scale: float | None = None
        try:
            # Return a deeper fused pool than the requested top_k: RRF rank
            # fusion suppresses single-modality hits (a chunk only in the BM25
            # list ranks far below the dense head), so the cross-encoder
            # reranker must see a wider candidate pool to rescue them.
            fused_limit = max(top_k * 4, 40) if use_hybrid else top_k
            query_kwargs: dict = dict(
                collection_name=self._collection_name,
                limit=fused_limit,
                with_payload=True,
                score_threshold=None,
            )

            if use_hybrid:
                assert self._bm25 is not None
                sparse = self._last_query_sparse
                if sparse is None and query_text is not None:
                    sparse = self._bm25.tokenize_query(query_text)
                if sparse is not None:
                    # RRF fused scores are 1/(k+rank) summed over the dense +
                    # sparse prefetches (2). Normalize to the same 0..1 scale as
                    # cosine similarity so downstream confidence thresholds hold.
                    rrf_confidence_scale = (self._hybrid_rrf_k + 1) / 2
                    # Pull a deeper candidate pool than the final top_k: RRF is a
                    # rank-fusion, not a relevance scorer, and BM25 often places
                    # the truly relevant chunk well below the dense head.  A
                    # cross-encoder reranker (when enabled) then does the precise
                    # selection from this larger pool.
                    prefetch_limit = max(top_k * 4, 40)
                    query_kwargs["prefetch"] = [
                        models.Prefetch(
                            query=query_embedding,
                            using="dense",
                            limit=prefetch_limit,
                            filter=query_filter,
                        ),
                        models.Prefetch(
                            query=sparse,
                            using="sparse",
                            limit=prefetch_limit,
                            filter=query_filter,
                        ),
                    ]
                    query_kwargs["query"] = models.RrfQuery(rrf=models.Rrf(k=self._hybrid_rrf_k))
                else:
                    query_kwargs["query"] = query_embedding
                    if self._hybrid_search:
                        query_kwargs["using"] = "dense"
                    if query_filter is not None:
                        query_kwargs["query_filter"] = query_filter
            else:
                query_kwargs["query"] = query_embedding
                if self._hybrid_search:
                    query_kwargs["using"] = "dense"
                if query_filter is not None:
                    query_kwargs["query_filter"] = query_filter

            raw_results = await self._client.query_points(**query_kwargs)
            retrieved: list[RetrievedChunk] = []
            points_list = cast(list[models.ScoredPoint], getattr(raw_results, "points", raw_results))
            for hit in points_list:
                payload = hit.payload or {}
                chunk = DocumentChunk(
                    chunk_id=str(hit.id),
                    source_name=payload.get("source_name", ""),
                    title=payload.get("title", ""),
                    url=payload.get("url", ""),
                    text=payload.get("text", ""),
                    content_hash=payload.get("content_hash", ""),
                    section_header=payload.get("section_header", ""),
                    chunk_type=payload.get("chunk_type", "text"),
                    word_count=payload.get("word_count", 0),
                    heading_path=tuple(payload.get("heading_path", [])),
                    chunk_index=payload.get("chunk_index", 0),
                    total_chunks=payload.get("total_chunks", 0),
                    crawled_at=payload.get("crawled_at", ""),
                )
                score = float(hit.score) if hit.score is not None else 0.0
                if rrf_confidence_scale is not None and score > 0.0:
                    confidence = min(1.0, score * rrf_confidence_scale)
                else:
                    confidence = max(0.0, min(1.0, score))
                distance = 1.0 - confidence
                retrieved.append(RetrievedChunk(chunk=chunk, distance=distance, confidence=confidence))

            return retrieved
        except Exception as exc:
            error_str = str(exc)
            if "404" in error_str or "Not Found" in error_str or "collection" in error_str.lower():
                raise VectorStoreError(
                    f"Qdrant collection '{self._collection_name}' not found at {self._url}. "
                    "Run 'dec reset-index' then re-ingest data."
                ) from exc
            logger.exception("Failed to async query Qdrant: %s", exc)
            raise

    def set_query_sparse(self, sparse_vector) -> None:
        """Set the sparse vector for the next hybrid query.

        Called by the RAG service before dispatching the query to the store
        so that the store can include it in the RRF prefetch without needing
        the raw query text itself.
        """
        self._last_query_sparse = sparse_vector

    def clear_query_sparse(self) -> None:
        """Clear the sparse vector set by ``set_query_sparse``."""
        self._last_query_sparse = None

    async def _warn_unfrozen_bm25_desync(self) -> None:
        """Warn once if sparse vectors are written with an unfrozen tokenizer
        into a collection that already contains points.

        An unfrozen tokenizer assigns vocabulary ids as it first sees them, so a
        partial ingestion into a populated collection produces sparse vectors
        whose ids will not match the tokenizer fitted and persisted at the end of
        that run.  This silently breaks hybrid search for the whole collection.
        """
        if self._bm25_desync_warned or self._bm25 is None or self._bm25_loaded_from_disk:
            return
        if self._bm25._frozen:
            return
        try:
            if self._client is None:
                return
            count = await self._client.count(collection_name=self._collection_name, exact=False)
            if count.count > 0:
                logger.warning(
                    "Writing sparse vectors with an UNFROZEN BM25 tokenizer into %s which "
                    "already contains %d points. The fitted tokenizer persisted at the end of "
                    "this run will desynchronize from the stored sparse vectors, breaking hybrid "
                    "search. Re-ingest from a clean index (`dec reset-index`) instead of "
                    "incrementally with a missing/empty BM25 cache.",
                    self._collection_name,
                    count.count,
                )
        except Exception:
            logger.debug("Could not check Qdrant point count for BM25 desync warning", exc_info=True)
        self._bm25_desync_warned = True

    def fit_bm25(self, texts: list[str]) -> None:
        """Fit the BM25 tokenizer on a corpus of chunk texts.

        Must be called once after ingestion completes and before any hybrid
        queries are made.  No-op when ``hybrid_search=False``.

        A tokenizer that was loaded from the persisted cache is already in sync
        with the sparse vectors stored in Qdrant, so it is never re-fitted or
        re-persisted here.  Re-fitting on a partial corpus (e.g. a ``reenrich``
        or ``retry-failed`` run) would assign a different vocabulary id space and
        silently desynchronize every stored sparse vector from the query-side
        tokenizer, breaking hybrid search.  Only a fresh build (no persisted
        tokenizer) fits and persists.
        """
        if self._bm25 is None:
            return
        if self._bm25_loaded_from_disk:
            logger.info(
                "Skipping BM25 refit: loaded persisted tokenizer (%s) already consistent with stored sparse vectors",
                self._bm25_persist_path,
            )
            return
        self._bm25.fit(texts)
        try:
            self._bm25.save(self._bm25_persist_path)
            logger.info("Persisted BM25 tokenizer to %s", self._bm25_persist_path)
            self._bm25_loaded_from_disk = True
        except Exception:
            logger.warning("Failed to persist BM25 tokenizer to %s", self._bm25_persist_path)
        logger.info(
            "BM25 tokenizer fitted: vocab=%d corpus_size=%d avg_doc_len=%.1f",
            self._bm25.vocab_size,
            self._bm25._corpus_size,
            self._bm25._avg_doc_len,
        )

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

    async def scroll_urls(self, source_name: str) -> list[str]:
        """Return all distinct URLs stored for a given source."""
        if self._client is None:
            return []
        try:
            all_urls: set[str] = set()
            next_offset = None
            while True:
                points, next_offset = await self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=models.Filter(
                        must=[models.FieldCondition(key="source_name", match=models.MatchValue(value=source_name))]
                    ),
                    limit=100,
                    offset=next_offset,
                    with_payload=["url"],
                    with_vectors=False,
                )
                for p in points:
                    url = (p.payload or {}).get("url")
                    if url:
                        all_urls.add(url)
                if next_offset is None or next_offset == "":
                    break
            return list(all_urls)
        except Exception as exc:
            logger.warning("Failed to scroll URLs for source=%s: %s", source_name, exc)
            return []

    async def count(self) -> int:
        """Return the number of points stored in the collection asynchronously."""
        if self._client is None:
            logger.warning("Qdrant client not initialized. Returning 0.")
            return 0
        try:
            collection_info = await self._client.get_collection(collection_name=self._collection_name)
            return collection_info.points_count or 0
        except Exception as exc:
            logger.exception("Failed to get async Qdrant collection count: %s", exc)
            raise

    async def count_urls(self, source_name: str) -> int:
        """Return the number of points stored for a given source.

        Uses the ``source_name`` payload index so it is cheap enough to call at
        the start of every source ingestion (unlike :meth:`scroll_urls`, which
        paginates the whole source).  Raises on failure so callers can
        distinguish an empty index from a Qdrant error.
        """
        if self._client is None:
            logger.warning("Qdrant client not initialized. Returning 0.")
            return 0
        result = await self._client.count(
            collection_name=self._collection_name,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="source_name", match=models.MatchValue(value=source_name))]
            ),
            exact=True,
        )
        return result.count or 0

    async def close(self) -> None:
        """Close the async client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
