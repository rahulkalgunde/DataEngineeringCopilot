from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import pickle
import time
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import Executor, ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.domain.exceptions import EmbeddingError, IngestionError, VectorStoreError
from data_engineering_copilot.domain.models import DocumentChunk, IngestionEvent, ParsedDocument, RawDocument
from data_engineering_copilot.domain.protocols import (
    ChunkerProtocol,
    EmbedderProtocol,
    ParserProtocol,
    SyncRedisProtocol,
    TelemetryTracerProtocol,
    VectorStoreProtocol,
)
from data_engineering_copilot.infrastructure.async_crawler import AsyncDocumentationCrawler
from data_engineering_copilot.infrastructure.async_url_registry import AsyncUrlRegistry
from data_engineering_copilot.services.api_extractor import ApiDocExtractor
from data_engineering_copilot.services.code_block_parser import CodeBlockParser
from data_engineering_copilot.services.contextual_chunk_enricher import ContextualChunkEnricher
from data_engineering_copilot.services.text_filter import ChunkFilter

if TYPE_CHECKING:
    from data_engineering_copilot.services.spark_chunker import SparkChunker
    from data_engineering_copilot.services.spark_metadata import SparkMetadata

log = structlog.get_logger(__name__)


@dataclasses.dataclass
class _ProcessedResult:
    """Outcome of processing one raw document.

    ``disposition`` is one of:
    - ``"indexed"``: chunks were produced and are ready to flush.
    - ``"duplicate"``: content already indexed with an identical hash.
    - ``"no_content"``: page parsed to no indexable content (skip permanently).
    """

    disposition: str
    chunks: list[DocumentChunk] = dataclasses.field(default_factory=list)
    content_hash: str = ""
    parsed: ParsedDocument | None = None


class AsyncIngestionService:
    def __init__(
        self,
        settings: AppSettings,
        crawler: AsyncDocumentationCrawler,
        parser: ParserProtocol,
        chunker: ChunkerProtocol,
        embeddings: EmbedderProtocol,
        vector_store: VectorStoreProtocol,
        redis_client: SyncRedisProtocol | None = None,
        parse_executor: Executor | None = None,
        chunk_executor: Executor | None = None,
        contextual_enricher: ContextualChunkEnricher | None = None,
        api_extractor: ApiDocExtractor | None = None,
        code_block_parser: CodeBlockParser | None = None,
        chunk_filter: ChunkFilter | None = None,
        telemetry: TelemetryTracerProtocol | None = None,
        spark_chunker: SparkChunker | None = None,
    ) -> None:
        self.settings = settings
        self.crawler = crawler
        self.parser = parser
        self.chunker = chunker
        self.embeddings = embeddings
        self.vector_store = vector_store
        self._redis_client = redis_client
        self._url_registries: dict[str, AsyncUrlRegistry] = {}
        self._processing_concurrency = settings.processing_concurrency
        self._corpus_texts: list[str] = []
        self._contextual_enricher = contextual_enricher
        self._api_extractor = api_extractor
        self._code_block_parser = code_block_parser
        self._chunk_filter = chunk_filter
        self._telemetry = telemetry
        self._spark_chunker = spark_chunker

        # Enrichment queue for decoupling enrichment from main pipeline
        self._enrichment_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._enrichment_worker_task: asyncio.Task | None = None
        self._task_id: str | None = None

        if parse_executor is not None:
            self._parse_executor = parse_executor
        else:
            self._parse_executor = ThreadPoolExecutor(max_workers=settings.parse_concurrency)
        if chunk_executor is not None:
            self._chunk_executor = chunk_executor
        else:
            self._chunk_executor = ThreadPoolExecutor(max_workers=settings.chunk_concurrency)

    @staticmethod
    def _spark_metadata(parsed: ParsedDocument) -> SparkMetadata:
        """Build SparkMetadata from a parsed document's provenance fields."""
        from data_engineering_copilot.services.spark_metadata import SparkMetadata

        return SparkMetadata(
            doc_type=parsed.doc_type,
            language=parsed.language,
            spark_version=parsed.spark_version,
            module=parsed.module,
            source_commit=parsed.source_commit,
            file_path=parsed.file_path,
            license=parsed.license,
        )

    async def _process_raw(
        self,
        loop: asyncio.AbstractEventLoop,
        raw_document: RawDocument,
        on_event: Callable[[IngestionEvent], None] | None,
        make_event: Callable[..., IngestionEvent],
        enrichment_semaphore: asyncio.Semaphore | None = None,
    ) -> _ProcessedResult:
        parsed = await loop.run_in_executor(self._parse_executor, self.parser.parse, raw_document)
        if parsed is None:
            log.info(
                "async_ingestion.page_skipped",
                source=raw_document.source_name,
                url=raw_document.url,
            )
            self._emit(
                on_event,
                make_event(
                    "page_skipped",
                    source_name=raw_document.source_name,
                    url=raw_document.url,
                    message=f"Skipped page with no readable content: {raw_document.url}",
                ),
            )
            return _ProcessedResult(disposition="no_content")

        content_hash = self._compute_content_hash(parsed.text)
        stored_hash = await self._get_stored_content_hash(parsed.url, parsed.source_name)
        if stored_hash is not None and stored_hash == content_hash:
            log.info(
                "async_ingestion.page_skipped_duplicate",
                source=parsed.source_name,
                url=parsed.url,
                hash=content_hash[:12],
            )
            self._emit(
                on_event,
                make_event(
                    "page_skipped_duplicate",
                    source_name=parsed.source_name,
                    url=parsed.url,
                    title=parsed.title,
                    message=f"Skipped duplicate page (content unchanged): {parsed.url}",
                ),
            )
            return _ProcessedResult(disposition="duplicate")

        if stored_hash is not None:
            log.info("async_ingestion.content_changed", url=parsed.url)
            await self._delete_chunks_for_url(parsed.url, parsed.source_name)

        extract_sentences = getattr(self.chunker, "extract_sentences", None)
        if self._spark_chunker is not None and parsed.doc_type:
            chunks = await self._spark_chunker.chunk(parsed, self._spark_metadata(parsed))
        elif extract_sentences is not None:
            sentences = extract_sentences(parsed.text)
            if sentences is None:
                chunks = await self.chunker.chunk(parsed)
            elif not sentences:
                log.info(
                    "async_ingestion.page_skipped_no_sentences",
                    source=parsed.source_name,
                    url=parsed.url,
                )
                self._emit(
                    on_event,
                    make_event(
                        "page_skipped",
                        source_name=parsed.source_name,
                        url=parsed.url,
                        title=parsed.title,
                        message=f"Skipped page with no extractable sentences: {parsed.url}",
                    ),
                )
                return _ProcessedResult(disposition="no_content", parsed=parsed)
            else:
                embeddings = await self.embeddings.embed_texts(sentences)
                chunks = await self.chunker.chunk(parsed, embeddings)
        else:
            chunks = await self.chunker.chunk(parsed)
        crawled_at = datetime.now(UTC).isoformat()
        chunks = [dataclasses.replace(chunk, content_hash=content_hash, crawled_at=crawled_at) for chunk in chunks]

        # Queue enrichment to Redis for background worker (if enricher configured)
        if self._contextual_enricher is not None and self._task_id_explicit:
            from data_engineering_copilot.factory import get_shared_redis_client

            redis = get_shared_redis_client(self.settings.redis_url)
            queue_key = f"ingestion:{self._task_id}:enrichment_queue"
            depth_key = f"{queue_key}:depth"
            item = pickle.dumps((parsed, chunks))
            await redis.rpush(queue_key, item)
            await redis.incr(depth_key)
            await redis.expire(queue_key, 86400)
            await redis.expire(depth_key, 86400)
            # Return chunks WITHOUT enrichment; enrichment applied later in _flush_batch_tracked
        elif self._contextual_enricher is not None and enrichment_semaphore is not None:
            # Fallback for direct calls without task_id
            async with enrichment_semaphore:
                chunks = await self._contextual_enricher.enrich(parsed, chunks)
        elif self._contextual_enricher is not None:
            chunks = await self._contextual_enricher.enrich(parsed, chunks)

        return _ProcessedResult(disposition="indexed", chunks=chunks, content_hash=content_hash, parsed=parsed)

    async def _enrichment_worker(self, task_id: str) -> None:
        """Background worker that processes enrichment tasks from Redis queue."""
        if self._contextual_enricher is None:
            log.warning("enrichment_worker.no_enricher", task_id=task_id)
            return

        from data_engineering_copilot.factory import get_shared_redis_client

        redis = get_shared_redis_client(self.settings.redis_url)
        queue_key = f"ingestion:{task_id}:enrichment_queue"
        results_key = f"ingestion:{task_id}:enrichment_results"
        heartbeat_key = f"ingestion:{task_id}:enrichment_worker:heartbeat"
        depth_key = f"{queue_key}:depth"
        ttl = 86400  # 24h

        log.info("enrichment_worker.started", task_id=task_id)
        try:
            while True:
                # Heartbeat for monitor liveness check
                await redis.set(heartbeat_key, "alive", ex=10)

                # Blocking pop from Redis list (with timeout for shutdown check)
                item = await redis.blpop(queue_key, timeout=1)
                if item is None:
                    continue
                _, payload = item
                if payload == b"SHUTDOWN":
                    log.info("enrichment_worker.shutdown_received", task_id=task_id)
                    break

                try:
                    # Deserialize (pickle) - handle both bytes and str from Redis
                    payload_bytes = payload if isinstance(payload, bytes) else payload.encode()
                    parsed, chunks = pickle.loads(payload_bytes)

                    # Acquire semaphore locally in worker
                    async with asyncio.Semaphore(self.settings.enrichment_concurrency):
                        enriched = await self._contextual_enricher.enrich(parsed, chunks)

                    # Store result in Redis hash with TTL
                    await redis.hset(results_key, parsed.url, pickle.dumps(enriched))
                    await redis.expire(results_key, ttl)

                    # Track queue depth for monitor
                    await redis.decr(depth_key)
                except Exception as exc:
                    log.error(
                        "enrichment_worker.item_failed",
                        task_id=task_id,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    # Don't fail the worker, just log and continue
                    await redis.decr(depth_key)
        except asyncio.CancelledError:
            log.info("enrichment_worker.cancelled", task_id=task_id)
            raise
        except Exception as exc:
            log.error(
                "enrichment_worker.fatal_error",
                task_id=task_id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
        finally:
            log.info("enrichment_worker.stopped", task_id=task_id)

    async def _flush_batch(
        self,
        loop: asyncio.AbstractEventLoop,
        batch_chunks: list[DocumentChunk],
        on_event: Callable[[IngestionEvent], None] | None,
        make_event: Callable[..., IngestionEvent],
    ) -> None:
        if not batch_chunks:
            return

        if self._chunk_filter is not None or self._api_extractor is not None or self._code_block_parser is not None:
            batch_chunks = await loop.run_in_executor(
                self._chunk_executor,
                lambda: self._apply_enrichers(batch_chunks),
            )

        batch_size = len(batch_chunks)
        self._emit(
            on_event,
            make_event(
                "batch_embedding",
                source_name="",
                message=f"Embedding {batch_size} chunks...",
                batch_size=batch_size,
                current_phase="embedding",
            ),
        )
        embed_span = None
        try:
            texts = [chunk.text for chunk in batch_chunks]
            if self._telemetry is not None:
                embed_span = self._telemetry.start_observation(
                    name="embedding",
                    as_type="generation",
                    model=getattr(self.embeddings, "model_name", getattr(self.embeddings, "model", None)),
                    input={"batch_size": len(texts)},
                )
            batch_vectors = await self.embeddings.embed_texts(texts)
            if embed_span is not None:
                embed_span.update(usage_details={"total": len(texts)})
                embed_span.end()
        except EmbeddingError as exc:
            if embed_span is not None:
                embed_span.update(output=f"embed_failed: {exc}", level="ERROR")
                embed_span.end()
            log.error(
                "async_ingestion.embed_batch_failed",
                batch_size=len(batch_chunks),
                error=str(exc),
            )
            raise IngestionError(f"Embedding failed: {exc}") from exc
        self._emit(
            on_event,
            make_event(
                "batch_indexing",
                source_name="",
                message=f"Indexing {batch_size} chunks into vector store...",
                batch_size=batch_size,
                current_phase="indexing",
            ),
        )
        try:
            await self.vector_store.upsert_chunks(batch_chunks, batch_vectors)
            if hasattr(self.vector_store, "fit_bm25"):
                self._corpus_texts.extend(texts)
        except Exception as exc:
            log.error(
                "async_ingestion.upsert_batch_failed",
                batch_size=len(batch_chunks),
                error=str(exc),
            )
            raise VectorStoreError(f"Vector store upsert failed: {exc}") from exc

        seen: set[tuple[str, str]] = set()
        for chunk in batch_chunks:
            key = (chunk.url, chunk.source_name)
            if key not in seen and chunk.content_hash:
                seen.add(key)
                await self._set_content_hash(chunk.url, chunk.source_name, chunk.content_hash)

    def _apply_enrichers(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        if self._chunk_filter is not None:
            chunks = self._chunk_filter.extract(chunks)
        if self._api_extractor is not None:
            chunks = self._api_extractor.extract(chunks)
        if self._code_block_parser is not None:
            chunks = self._code_block_parser.extract(chunks)
        return chunks

    async def ingest(
        self,
        max_pages_per_source: int | None = None,
        source_names: Iterable[str] | None = None,
        on_event: Callable[[IngestionEvent], None] | None = None,
        task_id: str | None = None,
    ) -> int:
        # Generate task_id if not provided
        self._task_id = task_id or f"ingest-{uuid.uuid4().hex[:8]}"
        self._task_id_explicit = task_id is not None

        # Start enrichment worker only if task_id explicitly provided (for Celery integration)
        if self._contextual_enricher is not None and self._task_id_explicit:
            self._enrichment_worker_task = asyncio.create_task(self._enrichment_worker(self._task_id))
            log.info("enrichment_worker.started", task_id=self._task_id)

        if hasattr(self.vector_store, "initialize") and asyncio.iscoroutinefunction(self.vector_store.initialize):
            await self.vector_store.initialize()
        start_time = time.time()
        page_limit = max_pages_per_source if max_pages_per_source is not None else self.settings.max_pages_per_source
        selected_sources = self._selected_sources(source_names)
        log.info(
            "async_ingestion.started",
            page_limit=page_limit,
            sources=[source.name for source in selected_sources],
            task_id=self._task_id,
        )

        if not selected_sources:
            return 0

        max_parallel = max(
            1, min(len(selected_sources) * self.settings.embed_concurrency, self._processing_concurrency)
        )
        embed_semaphore = asyncio.Semaphore(max_parallel)
        enrichment_semaphore = asyncio.Semaphore(self.settings.enrichment_concurrency)

        total_chunks = 0
        errors: list[str] = []
        source_counts: dict[str, int] = {}
        full_crawl_flags: dict[str, bool] = {}

        async with asyncio.TaskGroup() as tg:
            for source in selected_sources:
                tg.create_task(
                    self._run_source_task(
                        source,
                        page_limit,
                        on_event,
                        start_time,
                        embed_semaphore,
                        enrichment_semaphore,
                        results=errors,
                        source_counts=source_counts,
                        full_crawl_flags=full_crawl_flags,
                    )
                )

        total_chunks = sum(source_counts.values())
        total_elapsed = time.time() - start_time
        log.info("async_ingestion.completed", total_chunks=total_chunks, elapsed=round(total_elapsed, 1))

        # Always fit BM25 — fit() accumulates across calls.  Warn if partial.
        all_fully_crawled = all(full_crawl_flags.get(source.name, False) for source in selected_sources)
        if not all_fully_crawled:
            log.warning("async_ingestion.bm25_partial_crawl")
        if self._corpus_texts and hasattr(self.vector_store, "fit_bm25"):
            self.vector_store.fit_bm25(self._corpus_texts)
            log.info("async_ingestion.bm25_fitted", corpus_size=len(self._corpus_texts))
            self._corpus_texts.clear()

        await self.crawler.frontier.close()

        if errors:
            await self.close()
            raise IngestionError(f"Source ingestion errors: {'; '.join(errors)}")
        self._parse_executor.shutdown(wait=True)
        self._chunk_executor.shutdown(wait=True)
        await self.close()
        return total_chunks

    def stop(self) -> None:
        self._parse_executor.shutdown(wait=True)
        self._chunk_executor.shutdown(wait=True)

    async def close(self) -> None:
        """Close underlying clients and shut down thread pools."""
        # Shutdown enrichment worker
        if self._enrichment_worker_task is not None and self._task_id_explicit:
            from data_engineering_copilot.factory import get_shared_redis_client

            redis = get_shared_redis_client(self.settings.redis_url)
            queue_key = f"ingestion:{self._task_id}:enrichment_queue"
            await redis.rpush(queue_key, b"SHUTDOWN")
            try:
                await asyncio.wait_for(self._enrichment_worker_task, timeout=30.0)
            except TimeoutError:
                log.warning("enrichment_worker.shutdown_timeout", task_id=self._task_id)
                self._enrichment_worker_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._enrichment_worker_task
            except Exception as exc:
                log.warning("enrichment_worker.shutdown_error", task_id=self._task_id, error=str(exc))
            finally:
                # Cleanup Redis keys
                for key in [
                    queue_key,
                    f"{queue_key}:depth",
                    f"ingestion:{self._task_id}:enrichment_results",
                    f"ingestion:{self._task_id}:enrichment_worker:heartbeat",
                ]:
                    with contextlib.suppress(Exception):
                        await redis.delete(key)

        for component in (self.vector_store, self.embeddings, self.crawler):
            if not hasattr(component, "close"):
                continue
            with contextlib.suppress(TypeError, AttributeError):
                await component.close()
        self._parse_executor.shutdown(wait=False)
        self._chunk_executor.shutdown(wait=False)

    async def _run_source_task(
        self,
        source,
        page_limit: int,
        on_event: Callable[[IngestionEvent], None] | None,
        start_time: float,
        embed_semaphore: asyncio.Semaphore,
        enrichment_semaphore: asyncio.Semaphore,
        results: list[str],
        source_counts: dict[str, int],
        full_crawl_flags: dict[str, bool],
    ) -> int:
        try:
            count = await self._ingest_source(
                source,
                page_limit,
                on_event,
                start_time,
                embed_semaphore,
                enrichment_semaphore,
                full_crawl_flags=full_crawl_flags,
            )
            source_counts[source.name] = count
            return count
        except Exception as exc:
            log.error("async_ingestion.source_failed", source=source.name, error=str(exc))
            results.append(f"{source.name}: {exc}")
            return 0

    async def _ingest_source(
        self,
        source,
        page_limit: int,
        on_event: Callable[[IngestionEvent], None] | None,
        start_time: float,
        embed_semaphore: asyncio.Semaphore,
        enrichment_semaphore: asyncio.Semaphore,
        full_crawl_flags: dict[str, bool] | None = None,
    ) -> int:
        log.info("async_ingestion.crawling_source", source=source.name)
        queue: asyncio.Queue[RawDocument | None] = asyncio.Queue(
            maxsize=self._processing_concurrency * 2,
        )
        batch_lock = asyncio.Lock()

        shared: dict = {
            "total_chunks": 0,
            "global_pages_fetched": 0,
            "batch_chunks": [],
            "seen_urls": set(),
            "failures": [],
            "consecutive_ollama_failures": 0,
        }
        source_pages = 0
        source_chunks = 0

        def _make_event(
            event_type: str,
            source_name: str,
            message: str,
            _shared=shared,
            **kwargs: Any,
        ) -> IngestionEvent:
            elapsed = time.time() - start_time
            return IngestionEvent(
                event_type=event_type,
                source_name=source_name,
                message=message,
                timestamp=elapsed,
                total_pages_fetched=_shared["global_pages_fetched"],
                total_chunks_indexed=_shared["total_chunks"],
                elapsed_seconds=elapsed,
                **{k: v for k, v in kwargs.items() if v is not None},
            )

        loop = asyncio.get_running_loop()

        async def worker(
            _queue=queue,
            _lock=batch_lock,
            _shared=shared,
            _mk_event=_make_event,
            _enrichment_sem=enrichment_semaphore,
        ) -> None:
            nonlocal source_pages, source_chunks
            wloop = asyncio.get_running_loop()
            while True:
                raw_doc = await _queue.get()
                if raw_doc is None:
                    _queue.task_done()
                    break
                try:
                    result = await self._process_raw(wloop, raw_doc, on_event, _mk_event, _enrichment_sem)
                    if result.disposition == "duplicate":
                        await self._mark_url_state(raw_doc.url, "processed")
                        _shared["seen_urls"].add(raw_doc.url)
                        continue
                    if result.disposition == "no_content":
                        await self._mark_url_state(raw_doc.url, "skipped")
                        _shared["seen_urls"].add(raw_doc.url)
                        continue

                    chunks = result.chunks
                    parsed = result.parsed
                    if parsed is None:
                        _shared["seen_urls"].add(raw_doc.url)
                        continue

                    pending_batch = None
                    async with _lock:
                        n_chunks = len(chunks)
                        _shared["batch_chunks"].extend(chunks)
                        _shared["seen_urls"].add(raw_doc.url)
                        source_pages += 1
                        source_chunks += n_chunks

                        self._emit(
                            on_event,
                            _mk_event(
                                "page_indexed",
                                source_name=parsed.source_name,
                                url=parsed.url,
                                title=parsed.title,
                                message=f"Indexed {n_chunks} chunks from {parsed.title}",
                                chunks_indexed=n_chunks,
                                pages_fetched=source_pages,
                                current_phase="crawling",
                            ),
                        )

                        if len(_shared["batch_chunks"]) >= self.settings.ingestion_batch_chunk_size:
                            pending_batch = list(_shared["batch_chunks"])
                            _shared["batch_chunks"].clear()

                    if pending_batch is not None:
                        async with embed_semaphore:
                            await self._flush_batch_tracked(wloop, pending_batch, on_event, _mk_event, _shared)
                except Exception as exc:
                    # Per-page isolation: record the failure, mark the URL FAILED
                    # so it is retried on a later run, and keep going.
                    _shared["failures"].append((raw_doc.url, str(exc)))

                    # Detect Ollama overload for adaptive backpressure
                    is_ollama_error = (
                        isinstance(exc, EmbeddingError)
                        or "timeout" in str(exc).lower()
                        or "503" in str(exc)
                        or "overloaded" in str(exc).lower()
                    )
                    if is_ollama_error:
                        _shared["consecutive_ollama_failures"] += 1
                        log.warning(
                            "async_ingestion.ollama_overloaded",
                            consecutive_failures=_shared["consecutive_ollama_failures"],
                            url=raw_doc.url,
                            error=str(exc)[:200],
                        )
                        if _shared["consecutive_ollama_failures"] >= 3:
                            log.warning(
                                "async_ingestion.ollama_backpressure",
                                message="Ollama overloaded, reducing effective concurrency",
                                consecutive_failures=_shared["consecutive_ollama_failures"],
                            )
                    else:
                        _shared["consecutive_ollama_failures"] = 0
                        log.error(
                            "async_ingestion.worker_failed",
                            url=raw_doc.url,
                            error_type=type(exc).__name__,
                            error=str(exc)[:500],
                        )

                    with contextlib.suppress(Exception):
                        await self._mark_url_state(raw_doc.url, "failed", str(exc)[:500])
                finally:
                    _queue.task_done()

        w_tasks = [asyncio.create_task(worker()) for _ in range(self._processing_concurrency)]

        self._emit(
            on_event,
            _make_event(
                "source_start",
                source_name=source.name,
                message=f"Crawling {source.name}",
                current_phase="crawling",
            ),
        )

        # Recover content missing from the index (e.g. after `dec reset-qdrant`)
        # before crawling, regardless of the page limit.  The post-crawl
        # reconciliation below only fires on a fully drained frontier, so a
        # page-capped run would otherwise never re-discover these URLs.
        try:
            reactivated = await self._reactivate_after_reset(source.name)
            if reactivated > 0:
                log.info(
                    "async_ingestion.reactivated_missing_at_start",
                    source=source.name,
                    count=reactivated,
                )
        except Exception as exc:
            log.warning(
                "async_ingestion.start_reactivation_failed",
                source=source.name,
                error=str(exc),
            )

        crawl_succeeded = False
        crawl_span = None
        if self._telemetry is not None:
            crawl_span = self._telemetry.start_observation(
                name="ingestion-crawl",
                input={"source": source.name, "page_limit": page_limit},
                as_type="span",
            )
        try:
            async for raw_document in self.crawler.crawl(source, max_pages=page_limit, on_event=on_event):
                await queue.put(raw_document)
            log.info("async_ingestion._ingest_source.crawl_done source=%s", source.name)
            crawl_succeeded = True
        except Exception:
            if crawl_span is not None:
                crawl_span.update(output="CrawlError", level="ERROR")
                crawl_span.end()
            # Crawler failed mid-stream: drain unconsumed items so sentinel puts
            # cannot block, then surface the failure to the caller.
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break
            raise
        else:
            if crawl_span is not None:
                crawl_span.update(output=f"crawl_complete pages={source_pages}")
                crawl_span.end()
        finally:
            for _ in range(self._processing_concurrency):
                await queue.put(None)
            await queue.join()

            async with batch_lock:
                pending_batch = list(shared["batch_chunks"])
                shared["batch_chunks"].clear()

            async with embed_semaphore:
                await self._flush_batch_tracked(loop, pending_batch, on_event, _make_event, shared)

            for w in w_tasks:
                w.cancel()
            worker_results = await asyncio.gather(*w_tasks, return_exceptions=True)
            for i, result in enumerate(worker_results):
                if isinstance(result, Exception):
                    log.error("async_ingestion.worker_failed", worker=i, error=str(result))

            # Post-crawl reconciliation for fully-drained sources:
            #   - Prune: delete Qdrant chunks for URLs with no frontier record.
            #   - Reactivate: re-discover URLs whose content is missing from
            #     Qdrant (e.g. after `dec reset-qdrant`) so the next run
            #     re-fetches and re-indexes them.
            # Guards: the crawl must have succeeded and processed at least one
            # URL; any remaining DISCOVERED records mean the run was truncated by
            # the page limit, so "unseen" URLs are just not-crawled-yet.
            if crawl_succeeded and shared["seen_urls"]:
                remaining = await self.crawler.frontier.stats(source.name)
                if remaining.get("DISCOVERED", 0) > 0:
                    log.info(
                        "async_ingestion.stale_prune_skipped_partial",
                        source=source.name,
                        remaining_discovered=remaining.get("DISCOVERED", 0),
                    )
                else:
                    indexed_urls = await self._scroll_indexed_urls(source.name)
                    # Guard against a scroll failure masquerading as an empty
                    # index: an empty URL list alongside a non-empty index means
                    # the scroll errored, so skip prune + reactivate rather than
                    # wiping every chunk or re-crawling the whole source.
                    scroll_ok = True
                    if not indexed_urls:
                        indexed_count = await self.vector_store.count_urls(source.name)
                        if indexed_count > 0:
                            scroll_ok = False
                            log.warning(
                                "async_ingestion.stale_reconcile_skipped_scroll_error",
                                source=source.name,
                                indexed_count=indexed_count,
                            )
                    if scroll_ok:
                        stale_count = await self._prune_stale_chunks(source.name, set(indexed_urls))
                        if stale_count > 0:
                            log.info("async_ingestion.stale_chunks_pruned", source=source.name, count=stale_count)
                        reactivated = await self.crawler.frontier.reactivate_missing(
                            source.name,
                            set(indexed_urls),
                            max_attempts=self.settings.frontier_max_attempts,
                        )
                        if reactivated > 0:
                            log.info(
                                "async_ingestion.reactivated_missing",
                                source=source.name,
                                count=reactivated,
                            )
                        if full_crawl_flags is not None:
                            full_crawl_flags[source.name] = True

        failures = shared["failures"]
        if failures:
            log.warning(
                "async_ingestion.partial_failures",
                source=source.name,
                failure_count=len(failures),
                failed_urls=[url for url, _ in failures[:20]],
            )

        self._emit(
            on_event,
            _make_event(
                "source_complete",
                source_name=source.name,
                message=(f"Completed {source.name}: fetched {source_pages} pages, indexed {source_chunks} chunks."),
                chunks_indexed=source_chunks,
                pages_fetched=source_pages,
                current_phase="crawling",
            ),
        )
        log.info(
            "async_ingestion.source_completed",
            source=source.name,
            pages=source_pages,
            chunks=source_chunks,
        )
        return shared["total_chunks"]

    def _selected_sources(self, source_names: Iterable[str] | None):
        if source_names is None:
            return list(self.settings.sources)

        requested_names = tuple(name.strip() for name in source_names if name.strip())
        if not requested_names:
            log.error("async_ingestion.source_selection.no_sources")
            raise ValueError("At least one documentation source must be selected.")

        sources_by_name = {source.name: source for source in self.settings.sources}
        unknown_names = sorted(set(requested_names) - set(sources_by_name))
        if unknown_names:
            available = ", ".join(sources_by_name)
            log.error(
                "async_ingestion.source_selection.unknown",
                unknown=unknown_names,
                available=available,
            )
            raise ValueError(
                f"Unknown documentation source(s): {', '.join(unknown_names)}. Available sources: {available}"
            )

        return [sources_by_name[name] for name in requested_names]

    def _compute_content_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_url_registry(self, source_name: str) -> AsyncUrlRegistry | None:
        if self._redis_client is None:
            return None
        if source_name not in self._url_registries:
            self._url_registries[source_name] = AsyncUrlRegistry(self._redis_client, source_name)
        return self._url_registries[source_name]

    async def _get_stored_content_hash(self, url: str, source_name: str = "") -> str | None:
        """Return the content hash stored in the vector store for a URL.

        The vector store (Qdrant) is the source of truth for dedup decisions.
        Redis is only written through as a cache and is never consulted for
        skip decisions, so a stale/divergent Redis hash can never cause a page
        to be skipped after the vector store was reset.
        """
        return await self.vector_store.get_content_hash_for_url(url, source_name)

    async def _set_content_hash(self, url: str, source_name: str, content_hash: str) -> None:
        registry = self._get_url_registry(source_name)
        if registry is not None:
            await registry.set_html_hash(url, content_hash)

    async def _delete_chunks_for_url(self, url: str, source_name: str = "") -> None:
        deleter = getattr(self.vector_store, "delete_by_url", None)
        if deleter is not None:
            await deleter(url, source_name)
        else:
            log.debug("vector_store.no_delete_by_url", url=url)

    async def _mark_url_state(self, url: str, disposition: str, error: str = "") -> None:
        """Transition a frontier URL based on the ingestion outcome.

        The frontier is only mutated after the actual indexing outcome is
        known (PROCESSED = indexed, SKIPPED = no indexable content, FAILED =
        transient error).  ``url_hash`` is derived deterministically from the
        URL, so no schema change is required.
        """
        frontier = self.crawler.frontier
        url_hash = frontier.hash_url(url)
        if disposition == "processed":
            await frontier.mark_processed(url_hash)
        elif disposition == "skipped":
            await frontier.mark_skipped(url_hash)
        elif disposition == "failed":
            await frontier.mark_failed(url_hash, error)

    async def _flush_batch_tracked(
        self,
        loop: asyncio.AbstractEventLoop,
        batch_chunks: list[DocumentChunk],
        on_event: Callable[[IngestionEvent], None] | None,
        make_event: Callable[..., IngestionEvent],
        shared: dict,
    ) -> None:
        """Flush a batch and update frontier state for its URLs.

        On success every URL in the batch is marked PROCESSED; on failure the
        URLs are marked FAILED (retried on a later run) and the error is
        recorded in ``shared["failures"]`` so a single batch cannot fail the
        whole run.
        """
        if not batch_chunks:
            return

        # Pull enriched chunks from Redis if task_id explicitly provided
        if self._task_id_explicit:
            from data_engineering_copilot.factory import get_shared_redis_client

            redis = get_shared_redis_client(self.settings.redis_url)
            results_key = f"ingestion:{self._task_id}:enrichment_results"
            enriched_chunks = []
            for chunk in batch_chunks:
                enriched_data = await redis.hget(results_key, chunk.url)
                if enriched_data:
                    # Handle both bytes and str from Redis
                    enriched_bytes = enriched_data if isinstance(enriched_data, bytes) else enriched_data.encode()
                    enriched_chunks.append(pickle.loads(enriched_bytes))
                else:
                    enriched_chunks.append(chunk)  # fallback to non-enriched
            batch_chunks = enriched_chunks

        urls = {chunk.url for chunk in batch_chunks if chunk.url}
        span = None
        if self._telemetry is not None:
            span = self._telemetry.start_observation(
                name="ingestion-embed-upsert",
                input={"batch_size": len(batch_chunks), "urls": len(urls)},
                as_type="span",
            )
        try:
            await self._flush_batch(loop, batch_chunks, on_event, make_event)
            shared["total_chunks"] += len(batch_chunks)
            shared["global_pages_fetched"] += len(urls)
            for url in urls:
                await self._mark_url_state(url, "processed")
            if span is not None:
                span.update(output=f"flushed chunks={len(batch_chunks)}")
                span.end()
        except Exception as exc:
            if span is not None:
                span.update(output=f"flush_failed: {exc}", level="ERROR")
                span.end()
            log.error(
                "async_ingestion.flush_failed",
                batch_size=len(batch_chunks),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            for url in urls:
                shared["failures"].append((url, str(exc)))
                with contextlib.suppress(Exception):
                    await self._mark_url_state(url, "failed", str(exc))

    async def _reactivate_after_reset(self, source_name: str) -> int:
        """Re-discover indexed URLs that went missing from the vector store.

        When the vector store holds no points for a source but the frontier has
        PROCESSED/FAILED records (e.g. after ``dec reset-qdrant``), re-discover
        them so the current run re-fetches and re-indexes them.  Runs at the
        start of ingestion, so it works even when ``max_pages`` caps the run.
        """
        count = await self.vector_store.count_urls(source_name)
        if count > 0:
            return 0
        stats = await self.crawler.frontier.stats(source_name)
        if not (stats.get("PROCESSED", 0) or stats.get("FAILED", 0)):
            return 0
        return await self.crawler.frontier.reactivate_missing(
            source_name,
            set(),
            max_attempts=self.settings.frontier_max_attempts,
        )

    async def _scroll_indexed_urls(self, source_name: str) -> list[str]:
        from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

        if not isinstance(self.vector_store, AsyncQdrantVectorStore):
            return []
        try:
            return await self.vector_store.scroll_urls(source_name)
        except Exception as exc:
            log.warning("async_ingestion.scroll_urls_failed", source=source_name, error=str(exc))
            return []

    async def _prune_stale_chunks(self, source_name: str, stored_urls: set[str]) -> int:
        """Delete chunks for URLs present in Qdrant but unknown to the frontier.

        ``stored_urls`` are the URLs currently indexed for the source; a URL is
        stale when the frontier has no record of it at all.  This runs only
        after a fully drained crawl, so URLs that are merely not-crawled-yet
        (still DISCOVERED) are never touched.
        """
        if not stored_urls:
            return 0
        try:
            known = set(await self.crawler.frontier.all_urls(source_name))
        except Exception as exc:
            log.warning("async_ingestion.stale_prune_known_urls_failed", source=source_name, error=str(exc))
            return 0
        stale = [u for u in stored_urls if u not in known]
        if not stale:
            return 0
        deleted = 0
        for url in stale:
            try:
                await self._delete_chunks_for_url(url, source_name)
                deleted += 1
            except Exception as exc:
                log.warning("async_ingestion.stale_delete_failed", url=url, error=str(exc))
        log.info(
            "async_ingestion.stale_prune_complete",
            source=source_name,
            stored=len(stored_urls),
            known=len(known),
            deleted=deleted,
        )
        return deleted

    def _emit(self, on_event: Callable[[IngestionEvent], None] | None, event: IngestionEvent) -> None:
        if on_event is not None:
            on_event(event)
