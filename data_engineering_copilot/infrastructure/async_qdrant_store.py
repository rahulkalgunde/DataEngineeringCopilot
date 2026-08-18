"""Async Qdrant vector store implementation with optional hybrid search.

Dense-only mode: cosine similarity on embedding vectors (default behaviour).
Hybrid mode:    adds BM25 sparse vectors and uses Qdrant native RRF fusion
                at query time for combined dense + sparse retrieval.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Self, cast

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import SparseIndexParams, SparseVectorParams

from data_engineering_copilot.config.settings import PROJECT_ROOT, resolve_active_generation, settings
from data_engineering_copilot.domain.exceptions import VectorStoreError
from data_engineering_copilot.domain.models import DocumentChunk, RetrievalFilters, RetrievedChunk
from data_engineering_copilot.infrastructure.bm25_tokenizer import BM25Tokenizer
from data_engineering_copilot.services.query_signals import (
    RRF_DENSE_WEIGHT,
    RRF_EQUAL_PROFILE,
    RRF_IDENTIFIER_SPARSE_PROFILE,
    RRF_SPARSE_WEIGHT,
)


def chunk_to_payload(chunk: DocumentChunk) -> dict:
    """Exact Qdrant point payload for a chunk (mirrors ``_chunk_to_payload``).

    Standalone so the visualizer's Pipeline Lab can preview the stored point
    without holding a store instance.
    """
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
        "doc_type": chunk.doc_type,
        "language": chunk.language,
        "spark_version": chunk.spark_version,
        "module": chunk.module,
        "source_commit": chunk.source_commit,
        "file_path": chunk.file_path,
        "license": chunk.license,
        "deployment_mode": chunk.deployment_mode,
        "parser_version": chunk.parser_version,
        "chunker_version": chunk.chunker_version,
        "index_generation": chunk.index_generation,
        "parent_content_hash": chunk.parent_content_hash,
        "segment_index": chunk.segment_index,
        "segment_total": chunk.segment_total,
        "token_count": chunk.token_count,
        "character_count": chunk.character_count,
        "representation": chunk.representation,
        "parent_chunk_id": chunk.parent_chunk_id,
    }


logger = logging.getLogger(__name__)


def _resolve_bm25_cache_path(collection_name: str) -> Path:
    """Return the persisted BM25 tokenizer path for a collection.

    When ``collection_name`` matches the active logical alias (e.g.
    ``data_engineering_docs``), the Spark generation build persisted its BM25
    tokenizer under the generation *collection* name (``data_engineering_docs__
    <generation>``). Resolve to that file so hybrid search remains active after
    alias activation instead of silently degrading to dense-only. Falls back to
    the literal ``<collection_name>.json`` cache otherwise.
    """
    active_generation = resolve_active_generation()
    base = PROJECT_ROOT / ".bm25_cache"
    if active_generation:
        generation_cache = base / f"{collection_name}__{active_generation}.json"
        if generation_cache.exists():
            return generation_cache
    return base / f"{collection_name}.json"


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
        bm25_namespace: bool = False,
    ) -> None:
        self._url = url
        self._collection_name = collection_name
        self._hybrid_search = hybrid_search
        self._hybrid_rrf_k = hybrid_rrf_k
        self._bm25_namespace = bm25_namespace
        self._expected_bm25_version = (
            BM25Tokenizer.TOKENIZER_VERSION if bm25_namespace else BM25Tokenizer.LEGACY_TOKENIZER_VERSION
        )
        if bm25_persist_path is not None:
            self._bm25_persist_path = bm25_persist_path
        else:
            # When the store targets the active alias (e.g. `data_engineering_docs`
            # pointing at a generation collection), the BM25 tokenizer was
            # persisted under the generation collection name. Resolve the cache
            # to the active generation so hybrid search stays enabled after
            # activation instead of silently degrading to dense-only.
            self._bm25_persist_path = _resolve_bm25_cache_path(collection_name)
        self._bm25 = None
        self._bm25_loaded_from_disk = False
        self._bm25_version_mismatch = False
        if hybrid_search:
            if self._bm25_persist_path.exists():
                try:
                    self._bm25 = BM25Tokenizer.load(self._bm25_persist_path)
                    if self._bm25.version != self._expected_bm25_version:
                        logger.error(
                            "BM25 tokenizer version mismatch: cache at %s is %r but %r is required. "
                            "Rebuild the index before querying or ingesting.",
                            self._bm25_persist_path,
                            self._bm25.version,
                            self._expected_bm25_version,
                        )
                        self._bm25 = None
                        self._bm25_version_mismatch = True
                    else:
                        self._bm25_loaded_from_disk = True
                        logger.info("Loaded persisted BM25 tokenizer from %s", self._bm25_persist_path)
                except ValueError as exc:
                    logger.error(
                        "Unsupported BM25 tokenizer version in %s: %s. Rebuild the index.",
                        self._bm25_persist_path,
                        exc,
                    )
                    self._bm25 = None
                    self._bm25_version_mismatch = True
                except Exception:
                    logger.warning("Failed to load BM25 from %s, creating fresh", self._bm25_persist_path)
                    self._bm25 = BM25Tokenizer(namespace=bm25_namespace)
            else:
                self._bm25 = BM25Tokenizer(namespace=bm25_namespace)
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
        for _metadata_field in ("doc_type", "language", "spark_version", "module"):
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=_metadata_field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                logger.info(
                    "Payload index on %r already exists or could not be created.",
                    _metadata_field,
                    exc_info=True,
                )
        for _metadata_field in ("index_generation", "source_commit"):
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=_metadata_field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                logger.info(
                    "Payload index on %r already exists or could not be created.",
                    _metadata_field,
                    exc_info=True,
                )
        try:
            await self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="parent_chunk_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            logger.info("Payload index on 'parent_chunk_id' already exists or could not be created.", exc_info=True)

    def _embedding_dim(self) -> int:
        if self._embedding_dimension_override is not None:
            return self._embedding_dimension_override
        return settings.get_embedding_dimension()

    def _chunk_to_payload(self, chunk: DocumentChunk) -> dict:
        return chunk_to_payload(chunk)

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
        self._require_no_bm25_version_mismatch()

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

    @staticmethod
    def _build_query_filter(filters: RetrievalFilters | None) -> list[object] | None:
        """Build a list of Qdrant FieldConditions from structured filters.

        Empty tuples are treated as "no constraint". Returns ``None`` when the
        filter is empty so callers can skip adding a ``must`` condition.

        ``modules`` matches against the ``module`` payload field OR the chunk
        ``title``: rendered API pages (e.g. ``pyspark.sql.functions.filter``)
        carry an empty ``module`` but store the dotted identifier in ``title``,
        so an exact module-only filter would silently drop them and force a
        degraded unfiltered fallback.
        """
        if filters is None or filters.is_empty:
            return None
        conditions: list[object] = []
        field_map = (
            ("source_names", "source_name"),
            ("doc_types", "doc_type"),
            ("languages", "language"),
            ("versions", "spark_version"),
            ("chunk_types", "chunk_type"),
        )
        for attr, field in field_map:
            values = getattr(filters, attr)
            if values:
                conditions.append(models.FieldCondition(key=field, match=models.MatchAny(any=list(values))))
        modules = filters.modules
        if modules:
            conditions.append(
                models.Filter(
                    should=[
                        models.FieldCondition(key="module", match=models.MatchAny(any=list(modules))),
                        models.FieldCondition(key="title", match=models.MatchAny(any=list(modules))),
                    ]
                )
            )
        return conditions or None

    async def query(
        self,
        query_embedding: list[float],
        top_k: int,
        query_text: str | None = None,
        source_filter: list[str] | None = None,
        chunk_type_filter: str | None = None,
        metadata_filters: RetrievalFilters | None = None,
        fused_limit: int | None = None,
        rrf_profile: str = RRF_EQUAL_PROFILE,
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
        fused_limit:
            Optional fused candidate-pool size to pull from Qdrant. RRF rank
            fusion suppresses single-modality hits, so a cross-encoder reranker
            must see a wider pool than ``top_k``. When None, defaults to
            ``max(top_k * 4, 40)``. Callers that rerank (the RAG service) pass
            ``max(retrieval_top_k * 8, reranker_top_k * 5)`` so expected
            evidence can survive at ranks below the reranker limit.
        rrf_profile:
            Hybrid RRF profile. ``equal_rrf`` (default) fuses dense and sparse
            prefetches with equal weights; ``identifier_sparse_rrf`` boosts the
            sparse side with weights (dense=1.0, sparse=1.25). Only candidate
            limits, filters, RRF ``k``, and query vectors are otherwise
            unchanged between profiles.
        """
        if self._client is None:
            logger.warning("Qdrant client not initialized. Returning empty results.")
            return []
        self._require_no_bm25_version_mismatch()

        use_hybrid = self._hybrid_search and self._bm25 is not None and self._bm25._frozen

        # Build Qdrant filter for source names, chunk type, and metadata filters
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
        metadata_conditions = self._build_query_filter(metadata_filters)
        if metadata_conditions is not None:
            filter_conditions.extend(metadata_conditions)
        if filter_conditions:
            query_filter = models.Filter(must=filter_conditions)

        rrf_confidence_scale: float | None = None
        try:
            # Return a deeper fused pool than the requested top_k: RRF rank
            # fusion suppresses single-modality hits (a chunk only in the BM25
            # list ranks far below the dense head), so the cross-encoder
            # reranker must see a wider candidate pool to rescue them.
            # Task 11: when a caller supplies a fused_limit (the rerank pool),
            # honour it so the fused pool exposed by Qdrant and consumed by the
            # reranker are identical.
            if use_hybrid:
                effective_fused_limit = fused_limit if fused_limit is not None else max(top_k * 4, 40)
            else:
                effective_fused_limit = top_k
            query_kwargs: dict = dict(
                collection_name=self._collection_name,
                limit=effective_fused_limit,
                with_payload=True,
                score_threshold=None,
            )

            if use_hybrid:
                self._require_frozen_bm25()
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
                    prefetch_limit = effective_fused_limit
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
                    if rrf_profile == RRF_IDENTIFIER_SPARSE_PROFILE:
                        query_kwargs["query"] = models.RrfQuery(
                            rrf=models.Rrf(
                                k=self._hybrid_rrf_k,
                                weights=[RRF_DENSE_WEIGHT, RRF_SPARSE_WEIGHT],
                            )
                        )
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
                    chunk_id=payload.get("chunk_id", str(hit.id)),
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
                    doc_type=payload.get("doc_type", ""),
                    language=payload.get("language", ""),
                    spark_version=payload.get("spark_version", ""),
                    module=payload.get("module", ""),
                    source_commit=payload.get("source_commit", ""),
                    file_path=payload.get("file_path", ""),
                    license=payload.get("license", ""),
                    deployment_mode=payload.get("deployment_mode", ""),
                    parser_version=payload.get("parser_version", ""),
                    chunker_version=payload.get("chunker_version", ""),
                    index_generation=payload.get("index_generation", ""),
                    parent_content_hash=payload.get("parent_content_hash", ""),
                    segment_index=payload.get("segment_index", -1),
                    segment_total=payload.get("segment_total", 1),
                    token_count=payload.get("token_count", 0),
                    character_count=payload.get("character_count", 0),
                    representation=payload.get("representation", ""),
                    parent_chunk_id=payload.get("parent_chunk_id", ""),
                )
                score = float(hit.score) if hit.score is not None else 0.0
                if rrf_confidence_scale is not None and score > 0.0:
                    confidence = min(1.0, score * rrf_confidence_scale)
                else:
                    confidence = max(0.0, min(1.0, score))
                distance = 1.0 - confidence
                retrieved.append(RetrievedChunk(chunk=chunk, distance=distance, confidence=confidence))

            if any(r.chunk.parent_chunk_id for r in retrieved):
                retrieved = await self._substitute_parent_context(retrieved)

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

    async def _substitute_parent_context(self, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Replace child-chunk text with its parent context.

        In a hierarchical corpus, retrieved points are small child chunks. The
        LLM needs the broader parent context, so fetch each referenced parent
        point by ID and swap the child's ``text`` for the parent's. Children
        whose parent cannot be fetched keep their own (child) text.
        """
        if self._client is None:
            return retrieved
        parent_ids = sorted({r.chunk.parent_chunk_id for r in retrieved if r.chunk.parent_chunk_id})
        if not parent_ids:
            return retrieved
        try:
            parent_points = await self._client.retrieve(
                collection_name=self._collection_name,
                ids=[self._chunk_id_to_uuid(pid) for pid in parent_ids],
                with_payload=True,
            )
        except Exception as exc:
            logger.warning("Parent context fetch failed, keeping child text: %s", exc)
            return retrieved
        parent_text = {str(p.id): (p.payload or {}).get("text", "") for p in parent_points}
        substituted: list[RetrievedChunk] = []
        for item in retrieved:
            pid = item.chunk.parent_chunk_id
            if not pid:
                substituted.append(item)
                continue
            text = parent_text.get(self._chunk_id_to_uuid(pid))
            if not text:
                substituted.append(item)
                continue
            substituted.append(
                RetrievedChunk(
                    chunk=replace(item.chunk, text=text),
                    distance=item.distance,
                    confidence=item.confidence,
                )
            )
        return substituted

    def set_query_sparse(self, sparse_vector) -> None:
        """Set the sparse vector for the next hybrid query.

        Called by the RAG service before dispatching the query to the store
        so that the store can include it in the RRF prefetch without needing
        the raw query text itself.
        """
        self._last_query_sparse = sparse_vector

    def bm25_status(self) -> dict[str, object]:
        """Report BM25/hybrid state as a JSON-safe dictionary.

        Returns ``enabled``, ``fitted``, ``loaded_from_disk``, ``cache_path``
        and ``ready``. ``ready`` is True only when hybrid search is enabled
        and the tokenizer is present and frozen. Never raises merely because
        BM25 is unavailable.
        """
        enabled = bool(self._hybrid_search)
        fitted = bool(self._bm25 is not None and self._bm25._frozen)
        loaded_from_disk = bool(self._bm25_loaded_from_disk)
        cache_path = str(self._bm25_persist_path)
        return {
            "enabled": enabled,
            "fitted": fitted,
            "loaded_from_disk": loaded_from_disk,
            "cache_path": cache_path,
            "ready": bool(enabled and fitted),
        }

    def is_hybrid_ready(self) -> bool:
        """Return True only when hybrid search can serve sparse queries.

        Hybrid search is ready when it is enabled and the BM25 tokenizer has
        been fitted (frozen). Does not raise when the tokenizer is absent or
        not yet fitted.
        """
        return bool(self._hybrid_search and self._bm25 is not None and self._bm25._frozen)

    def _require_no_bm25_version_mismatch(self) -> None:
        """Fail fast when a persisted tokenizer version does not match the
        configured namespace mode.

        Sparse vectors stored in Qdrant were produced by the tokenizer version
        persisted with the cache; tokenizing with a different version would
        silently return wrong or empty results. Called before every hybrid
        query/upsert (plan Task 7 Step 3).
        """
        if self._bm25_version_mismatch:
            raise VectorStoreError(
                f"BM25 tokenizer version mismatch at {self._bm25_persist_path}: expected "
                f"{self._expected_bm25_version!r}. Rebuild the index before querying or ingesting."
            )

    def _require_frozen_bm25(self) -> None:
        """Raise when the BM25 tokenizer is missing or not yet fitted."""
        self._require_no_bm25_version_mismatch()
        if self._bm25 is None or not self._bm25._frozen:
            raise VectorStoreError("BM25 tokenizer is not ready")

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
        self._require_no_bm25_version_mismatch()
        if self._bm25 is None:
            return
        if self._bm25_loaded_from_disk:
            logger.info(
                "Skipping BM25 refit: loaded persisted tokenizer (%s) already consistent with stored sparse vectors",
                self._bm25_persist_path,
            )
            return
        self._require_no_bm25_version_mismatch()
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

    def fit_bm25_corpus(self, texts: Sequence[str]) -> None:
        """Fit a fresh BM25 tokenizer on the complete corpus for a new generation.

        The tokenizer must be frozen (via ``fit``) before any sparse vectors are
        written. Raises ``ValueError`` when hybrid search is disabled or the
        corpus is empty.
        """
        if not self._hybrid_search:
            raise ValueError("fit_bm25_corpus requires hybrid_search_enabled=True")
        texts_list = list(texts)
        if not texts_list:
            raise ValueError("fit_bm25_corpus requires a non-empty corpus")
        self._require_no_bm25_version_mismatch()
        # Fresh tokenizer for every new generation; never reuse an active one.
        self._bm25 = BM25Tokenizer(namespace=self._bm25_namespace)
        self._bm25.fit(texts_list)
        self._bm25.save(self._bm25_persist_path)
        self._bm25_loaded_from_disk = True
        logger.info(
            "BM25 corpus fitted: vocab=%d corpus_size=%d",
            self._bm25.vocab_size,
            self._bm25._corpus_size,
        )

    async def upsert_frozen_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        vectors: Sequence[list[float]],
        _sub_batch_size: int = 256,
    ) -> None:
        """Upsert dense + sparse vectors using the frozen BM25 tokenizer.

        Requires a frozen tokenizer (see ``fit_bm25_corpus``). Writes sparse
        vectors generated from the frozen vocabulary alongside dense vectors.
        Raises ``ValueError`` for mismatched lengths or missing generation.
        """
        chunks_list = list(chunks)
        vectors_list = list(vectors)
        if len(chunks_list) != len(vectors_list):
            raise ValueError("chunks and vectors must have equal lengths")
        if not chunks_list:
            raise ValueError("upsert_frozen_chunks requires a non-empty chunk list")
        self._require_no_bm25_version_mismatch()
        if self._bm25 is None or not self._bm25._frozen:
            raise ValueError("upsert_frozen_chunks requires a frozen BM25 tokenizer")
        for chunk in chunks_list:
            if not chunk.content_hash:
                raise ValueError(f"chunk {chunk.chunk_id!r} has empty content_hash")
            if not chunk.index_generation:
                raise ValueError(f"chunk {chunk.chunk_id!r} has empty index_generation")

        if self._client is None:
            raise VectorStoreError("Qdrant client not initialized")

        for i in range(0, len(chunks_list), _sub_batch_size):
            sub_chunks = chunks_list[i : i + _sub_batch_size]
            sub_embeddings = vectors_list[i : i + _sub_batch_size]
            ids = [self._chunk_id_to_uuid(chunk.chunk_id) for chunk in sub_chunks]
            vectors_dict = {
                "dense": [list(e) for e in sub_embeddings],
                "sparse": [self._bm25.tokenize_query(c.text) for c in sub_chunks],
            }
            payloads = [self._chunk_to_payload(chunk) for chunk in sub_chunks]
            await self._client.upsert(
                collection_name=self._collection_name,
                points=models.Batch(ids=[str(i) for i in ids], vectors=vectors_dict, payloads=payloads),  # type: ignore[arg-type]
            )

    async def validate_index_generation(self, expected_points: int | None = None) -> dict[str, object]:
        """Validate a built generation's Qdrant state and BM25 readiness.

        When ``expected_points`` is provided, the point count must match exactly.
        Raises ``VectorStoreError`` when required invariants fail.
        """
        if self._client is None:
            raise VectorStoreError("Qdrant client not initialized")
        collection_info = await self._client.get_collection(collection_name=self._collection_name)
        point_count = collection_info.points_count or 0
        config = collection_info.config
        params = config.params
        sparse_configured = bool(getattr(params, "sparse_vectors", None))
        bm25_ready = bool(self._bm25 is not None and self._bm25._frozen)

        report: dict[str, object] = {
            "collection": self._collection_name,
            "point_count": point_count,
            "sparse_configured": sparse_configured,
            "bm25_ready": bm25_ready,
            "expected_points": expected_points,
        }
        if sparse_configured and not bm25_ready:
            raise VectorStoreError("Sparse vectors configured but BM25 tokenizer is not ready")
        if expected_points is not None and point_count != expected_points:
            raise VectorStoreError(f"Point count mismatch: expected {expected_points}, got {point_count}")
        report["passed"] = True
        return report

    async def get_content_hash_for_url(self, url: str, source_name: str = "") -> str | None:
        """Retrieve stored content_hash for a given URL (optionally scoped to a source) asynchronously."""
        if self._client is None:
            return None
        try:
            must: list[models.Condition] = [models.FieldCondition(key="url", match=models.MatchValue(value=url))]
            if source_name:
                must.append(models.FieldCondition(key="source_name", match=models.MatchValue(value=source_name)))
            points, _ = await self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=models.Filter(must=must),
                limit=50,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                if point.payload and point.payload.get("content_hash"):
                    return point.payload["content_hash"]
            return None
        except Exception as exc:
            logger.warning("Failed to retrieve content hash for url=%s: %s", url, exc)
            return None

    async def delete_by_url(self, url: str, source_name: str = "") -> None:
        """Delete all points whose payload ``url`` field matches (optionally scoped to a source) asynchronously."""
        if self._client is None:
            logger.warning("Qdrant client not initialized. Cannot delete by url=%s.", url)
            return
        try:
            must: list[models.Condition] = [models.FieldCondition(key="url", match=models.MatchValue(value=url))]
            if source_name:
                must.append(models.FieldCondition(key="source_name", match=models.MatchValue(value=source_name)))
            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=models.FilterSelector(filter=models.Filter(must=must)),
            )
            logger.info("Deleted all points for url=%s source=%s", url, source_name)
        except Exception as exc:
            logger.warning("Failed to delete points for url=%s source=%s: %s", url, source_name, exc)

    async def scroll_chunks_by_parent_hash(self, parent_hash: str, source_name: str = "") -> list[DocumentChunk]:
        """Return every sibling chunk sharing *parent_hash*, ordered by segment index.

        Sibling segments were produced by the lossless token-budget splitter, so
        ``"".join(texts)`` reconstructs the original parent block. Used by the
        post-retrieval sibling-rejoin step so a retrieved segment's surrounding
        context (e.g. the YARN paragraph plus its Kubernetes sibling) is
        restored into the prompt instead of being dropped at index time.
        """
        if self._client is None:
            return []
        must: list[models.Condition] = [
            models.FieldCondition(key="parent_content_hash", match=models.MatchValue(value=parent_hash))
        ]
        if source_name:
            must.append(models.FieldCondition(key="source_name", match=models.MatchValue(value=source_name)))
        siblings: list[DocumentChunk] = []
        next_offset = None
        try:
            while True:
                points, next_offset = await self._client.scroll(
                    collection_name=self._collection_name,
                    scroll_filter=models.Filter(must=must),
                    limit=100,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    siblings.append(
                        DocumentChunk(
                            chunk_id=payload.get("chunk_id", str(point.id)),
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
                            doc_type=payload.get("doc_type", ""),
                            language=payload.get("language", ""),
                            spark_version=payload.get("spark_version", ""),
                            module=payload.get("module", ""),
                            source_commit=payload.get("source_commit", ""),
                            file_path=payload.get("file_path", ""),
                            license=payload.get("license", ""),
                            deployment_mode=payload.get("deployment_mode", ""),
                            parser_version=payload.get("parser_version", ""),
                            chunker_version=payload.get("chunker_version", ""),
                            index_generation=payload.get("index_generation", ""),
                            parent_content_hash=payload.get("parent_content_hash", ""),
                            segment_index=payload.get("segment_index", -1),
                            segment_total=payload.get("segment_total", 1),
                        )
                    )
                if next_offset is None or next_offset == "":
                    break
            siblings.sort(key=lambda c: c.segment_index)
            return siblings
        except Exception as exc:
            logger.warning("Failed to scroll siblings for parent=%r: %s", parent_hash, exc)
            return []

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

    async def verify_payload_texts(self, expected: dict[str, str]) -> list[str]:
        """Compare stored Qdrant payload text to ``expected`` by chunk_id.

        Returns a list of mismatches (empty when every point's payload text
        equals the expected text). This proves the persisted/embedded text is
        exactly what was indexed (no truncation, no drift).
        """
        if self._client is None:
            return ["Qdrant client not initialized"]
        mismatches: list[str] = []
        next_offset = None
        while True:
            points, next_offset = await self._client.scroll(
                collection_name=self._collection_name,
                limit=100,
                offset=next_offset,
                with_payload=["chunk_id", "text"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                chunk_id = payload.get("chunk_id")
                stored_text = payload.get("text")
                expected_text = expected.get(chunk_id) if isinstance(chunk_id, str) else None
                if expected_text is None:
                    mismatches.append(f"Qdrant point {point.id!r} has unexpected chunk_id {chunk_id!r}")
                elif stored_text != expected_text:
                    mismatches.append(
                        f"chunk_id {chunk_id!r} payload text differs from persisted chunks.jsonl text "
                        f"({len(stored_text or '')} chars stored vs {len(expected_text)} chars expected)"
                    )
            if next_offset is None or next_offset == "":
                break
        return mismatches

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
