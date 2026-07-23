from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import multiprocessing
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Executor, ProcessPoolExecutor

import structlog

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.domain.exceptions import EmbeddingError, IngestionError, VectorStoreError
from data_engineering_copilot.domain.models import DocumentChunk, IngestionEvent, ParsedDocument, RawDocument
from data_engineering_copilot.domain.protocols import (
    ChunkerProtocol,
    EmbedderProtocol,
    ParserProtocol,
    VectorStoreProtocol,
)
from data_engineering_copilot.infrastructure.async_crawler import AsyncDocumentationCrawler
from data_engineering_copilot.infrastructure.async_url_registry import AsyncUrlRegistry
from data_engineering_copilot.services.chunk_enrichment import enrich_chunks

log = structlog.get_logger(__name__)


class AsyncIngestionService:
    def __init__(
        self,
        settings: AppSettings,
        crawler: AsyncDocumentationCrawler,
        parser: ParserProtocol,
        chunker: ChunkerProtocol,
        embeddings: EmbedderProtocol,
        vector_store: VectorStoreProtocol,
        redis_client: object | None = None,
        parse_executor: Executor | None = None,
        chunk_executor: Executor | None = None,
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

        if parse_executor is not None:
            self._parse_executor = parse_executor
        else:
            ctx = multiprocessing.get_context("spawn")
            self._parse_executor = ProcessPoolExecutor(max_workers=settings.parse_concurrency, mp_context=ctx)
        if chunk_executor is not None:
            self._chunk_executor = chunk_executor
        else:
            ctx = multiprocessing.get_context("spawn")
            self._chunk_executor = ProcessPoolExecutor(max_workers=settings.chunk_concurrency, mp_context=ctx)

    async def _process_raw(
        self,
        loop: asyncio.AbstractEventLoop,
        raw_document: RawDocument,
        on_event: Callable[[IngestionEvent], None] | None,
        make_event: Callable[..., IngestionEvent],
    ) -> tuple[list[DocumentChunk], str, ParsedDocument] | None:
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
            return None

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
            return None

        if stored_hash is not None:
            log.info("async_ingestion.content_changed", url=parsed.url)
            await self._delete_chunks_for_url(parsed.url)

        if hasattr(self.chunker, "extract_sentences"):
            sentences = self.chunker.extract_sentences(parsed.text)
            if not sentences:
                return None
            embeddings = await self.embeddings.embed_texts(sentences)
            chunks = await loop.run_in_executor(self._chunk_executor, self.chunker.chunk, parsed, embeddings)
        else:
            chunks = await loop.run_in_executor(self._chunk_executor, self.chunker.chunk, parsed)
        chunks = [dataclasses.replace(chunk, content_hash=content_hash) for chunk in chunks]
        chunks = enrich_chunks(chunks)
        return chunks, content_hash, parsed

    async def _flush_batch(
        self,
        loop: asyncio.AbstractEventLoop,
        batch_chunks: list[DocumentChunk],
        on_event: Callable[[IngestionEvent], None] | None,
        make_event: Callable[..., IngestionEvent],
    ) -> None:
        if not batch_chunks:
            return
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
        try:
            texts = [chunk.text for chunk in batch_chunks]
            batch_vectors = await self.embeddings.embed_texts(texts)
        except EmbeddingError as exc:
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

        batch_chunks.clear()

    async def ingest(
        self,
        max_pages_per_source: int | None = None,
        source_names: Iterable[str] | None = None,
        on_event: Callable[[IngestionEvent], None] | None = None,
    ) -> int:
        start_time = time.time()
        page_limit = max_pages_per_source or self.settings.max_pages_per_source
        selected_sources = self._selected_sources(source_names)
        log.info(
            "async_ingestion.started",
            page_limit=page_limit,
            sources=[source.name for source in selected_sources],
        )

        if not selected_sources:
            return 0

        max_parallel = min(len(selected_sources), self._processing_concurrency)
        embed_semaphore = asyncio.Semaphore(max_parallel)

        total_chunks = 0
        errors: list[str] = []
        source_counts: dict[str, int] = {}

        async with asyncio.TaskGroup() as tg:
            for source in selected_sources:
                tg.create_task(
                    self._run_source_task(
                        source,
                        page_limit,
                        on_event,
                        start_time,
                        embed_semaphore,
                        results=errors,
                        source_counts=source_counts,
                    )
                )

        total_chunks = sum(source_counts.values())
        total_elapsed = time.time() - start_time
        log.info("async_ingestion.completed", total_chunks=total_chunks, elapsed=round(total_elapsed, 1))
        await self.crawler.frontier.close()

        if errors:
            raise IngestionError(f"Source ingestion errors: {'; '.join(errors)}")
        self._parse_executor.shutdown(wait=True)
        self._chunk_executor.shutdown(wait=True)
        return total_chunks

    def stop(self) -> None:
        self._parse_executor.shutdown(wait=True)
        self._chunk_executor.shutdown(wait=True)

    async def _run_source_task(
        self,
        source,
        page_limit: int,
        on_event: Callable[[IngestionEvent], None] | None,
        start_time: float,
        embed_semaphore: asyncio.Semaphore,
        results: list[str],
        source_counts: dict[str, int],
    ) -> int:
        try:
            count = await self._ingest_source(source, page_limit, on_event, start_time, embed_semaphore)
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
        }
        source_pages = 0
        source_chunks = 0

        def _make_event(
            event_type: str,
            source_name: str,
            message: str,
            _shared=shared,
            **kwargs: object,
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
        ) -> None:
            nonlocal source_pages, source_chunks
            wloop = asyncio.get_running_loop()
            while True:
                raw_doc = await _queue.get()
                if raw_doc is None:
                    _queue.task_done()
                    break
                try:
                    result = await self._process_raw(wloop, raw_doc, on_event, _mk_event)
                    if result is None:
                        continue
                    chunks, content_hash, parsed = result

                    async with _lock:
                        n_chunks = len(chunks)
                        _shared["total_chunks"] += n_chunks
                        _shared["global_pages_fetched"] += 1
                        _shared["batch_chunks"].extend(chunks)
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
                            async with embed_semaphore:
                                await self._flush_batch(wloop, _shared["batch_chunks"], on_event, _mk_event)
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

        try:
            async for raw_document in self.crawler.crawl(source, max_pages=page_limit, on_event=on_event):
                await queue.put(raw_document)
        finally:
            for _ in range(self._processing_concurrency):
                await queue.put(None)
            await queue.join()

            async with batch_lock, embed_semaphore:
                await self._flush_batch(loop, shared["batch_chunks"], on_event, _make_event)

            for w in w_tasks:
                w.cancel()
            await asyncio.gather(*w_tasks, return_exceptions=True)

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
        registry = self._get_url_registry(source_name) if source_name else None
        if registry is not None:
            cached = await registry.get_html_hash(url)
            if cached is not None:
                return cached
        return await self.vector_store.get_content_hash_for_url(url)

    async def _set_content_hash(self, url: str, source_name: str, content_hash: str) -> None:
        registry = self._get_url_registry(source_name)
        if registry is not None:
            await registry.set_html_hash(url, content_hash)

    async def _delete_chunks_for_url(self, url: str) -> None:
        deleter = getattr(self.vector_store, "delete_by_url", None)
        if deleter is not None:
            await deleter(url)
        else:
            log.debug("vector_store.no_delete_by_url", url=url)

    def _emit(self, on_event: Callable[[IngestionEvent], None] | None, event: IngestionEvent) -> None:
        if on_event is not None:
            on_event(event)
