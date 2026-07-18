from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable

import structlog

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.domain.exceptions import EmbeddingError, IngestionError, VectorStoreError
from data_engineering_copilot.domain.models import DocumentChunk, IngestionEvent
from data_engineering_copilot.domain.protocols import (
    ChunkerProtocol,
    CrawlerProtocol,
    EmbedderProtocol,
    ParserProtocol,
    VectorStoreProtocol,
)

log = structlog.get_logger(__name__)


class IngestionService:
    def __init__(
        self,
        settings: AppSettings,
        crawler: CrawlerProtocol,
        parser: ParserProtocol,
        chunker: ChunkerProtocol,
        embeddings: EmbedderProtocol,
        vector_store: VectorStoreProtocol,
    ) -> None:
        self.settings = settings
        self.crawler = crawler
        self.parser = parser
        self.chunker = chunker
        self.embeddings = embeddings
        self.vector_store = vector_store

    def ingest(
        self,
        max_pages_per_source: int | None = None,
        source_names: Iterable[str] | None = None,
        on_event: Callable[[IngestionEvent], None] | None = None,
    ) -> int:
        import time as time_module

        start_time = time_module.time()
        page_limit = max_pages_per_source or self.settings.max_pages_per_source
        total_chunks = 0
        global_pages_fetched = 0
        selected_sources = self._selected_sources(source_names)
        log.info(
            "ingestion.started",
            page_limit=page_limit,
            sources=[source.name for source in selected_sources],
        )

        batch_chunks: list[DocumentChunk] = []

        def _make_event(
            event_type: str,
            source_name: str,
            message: str,
            **kwargs: object,
        ) -> IngestionEvent:
            elapsed = time_module.time() - start_time
            return IngestionEvent(
                event_type=event_type,
                source_name=source_name,
                message=message,
                timestamp=elapsed,
                total_pages_fetched=global_pages_fetched,
                total_chunks_indexed=total_chunks,
                elapsed_seconds=elapsed,
                **{k: v for k, v in kwargs.items() if v is not None},
            )

        def flush_batch() -> None:
            nonlocal total_chunks
            if not batch_chunks:
                return
            batch_size = len(batch_chunks)
            self._emit(
                on_event,
                _make_event(
                    "batch_embedding",
                    source_name="",
                    message=f"Embedding {batch_size} chunks...",
                    batch_size=batch_size,
                    current_phase="embedding",
                ),
            )
            try:
                batch_vectors = self.embeddings.embed_texts([chunk.text for chunk in batch_chunks])
            except EmbeddingError as exc:
                log.error(
                    "ingestion.embed_batch_failed",
                    batch_size=len(batch_chunks),
                    error=str(exc),
                )
                raise IngestionError(f"Embedding failed: {exc}") from exc
            self._emit(
                on_event,
                _make_event(
                    "batch_indexing",
                    source_name="",
                    message=f"Indexing {batch_size} chunks into vector store...",
                    batch_size=batch_size,
                    current_phase="indexing",
                ),
            )
            try:
                self.vector_store.upsert_chunks(batch_chunks, batch_vectors)
            except Exception as exc:
                log.error(
                    "ingestion.upsert_batch_failed",
                    batch_size=len(batch_chunks),
                    error=str(exc),
                )
                raise VectorStoreError(f"Vector store upsert failed: {exc}") from exc
            batch_chunks.clear()

        for source in selected_sources:
            log.info("ingestion.crawling_source", source=source.name)
            log.info("ingestion.source_started", source=source.name, page_limit=page_limit)
            source_pages_fetched = 0
            source_chunks_indexed = 0
            self._emit(
                on_event,
                _make_event(
                    "source_start",
                    source_name=source.name,
                    message=f"Crawling {source.name}",
                    current_phase="crawling",
                ),
            )
            for raw_document in self.crawler.crawl(source, max_pages=page_limit, on_event=on_event):
                source_pages_fetched += 1
                global_pages_fetched += 1
                parsed = self.parser.parse(raw_document)
                if parsed is None:
                    log.info(
                        "ingestion.page_skipped",
                        source=raw_document.source_name,
                        url=raw_document.url,
                    )
                    self._emit(
                        on_event,
                        _make_event(
                            "page_skipped",
                            source_name=raw_document.source_name,
                            url=raw_document.url,
                            message=f"Skipped page with no readable documentation content: {raw_document.url}",
                            pages_fetched=source_pages_fetched,
                        ),
                    )
                    continue

                # --- Content-hash dedup: skip embedding if page unchanged ---
                content_hash = self._compute_content_hash(parsed.text)
                stored_hash = self._get_stored_content_hash(parsed.url)
                if stored_hash is not None and stored_hash == content_hash:
                    log.info(
                        "ingestion.page_skipped_duplicate",
                        source=parsed.source_name,
                        url=parsed.url,
                        hash=content_hash[:12],
                    )
                    self._emit(
                        on_event,
                        _make_event(
                            "page_skipped_duplicate",
                            source_name=parsed.source_name,
                            url=parsed.url,
                            title=parsed.title,
                            message=f"Skipped duplicate page (content unchanged): {parsed.url}",
                            pages_fetched=source_pages_fetched,
                        ),
                    )
                    continue

                # --- Ghost-chunk cleanup: if content changed, purge old chunks ---
                if stored_hash is not None:
                    log.info(
                        "ingestion.content_changed",
                        url=parsed.url,
                    )
                    self._delete_chunks_for_url(parsed.url)

                chunks = self.chunker.chunk(parsed)
                # Stamp content_hash onto every chunk for future dedup lookups
                import dataclasses

                chunks = [dataclasses.replace(chunk, content_hash=content_hash) for chunk in chunks]
                batch_chunks.extend(chunks)

                if len(batch_chunks) >= self.settings.ingestion_batch_chunk_size:
                    flush_batch()

                total_chunks += len(chunks)
                source_chunks_indexed += len(chunks)
                log.info("ingestion.chunks_indexed", count=len(chunks), title=parsed.title)
                log.info(
                    "ingestion.page_indexed",
                    source=parsed.source_name,
                    url=parsed.url,
                    title=parsed.title,
                    chunks=len(chunks),
                    total_chunks=total_chunks,
                )
                self._emit(
                    on_event,
                    _make_event(
                        "page_indexed",
                        source_name=parsed.source_name,
                        url=parsed.url,
                        title=parsed.title,
                        message=f"Indexed {len(chunks)} chunks from {parsed.title}",
                        chunks_indexed=len(chunks),
                        pages_fetched=source_pages_fetched,
                        current_phase="crawling",
                    ),
                )

            self._emit(
                on_event,
                _make_event(
                    "source_complete",
                    source_name=source.name,
                    message=(
                        f"Completed {source.name}: fetched {source_pages_fetched} HTML pages, "
                        f"indexed {source_chunks_indexed} chunks."
                    ),
                    chunks_indexed=source_chunks_indexed,
                    pages_fetched=source_pages_fetched,
                    current_phase="crawling",
                ),
            )
            log.info(
                "ingestion.source_completed",
                source=source.name,
                pages=source_pages_fetched,
                chunks=source_chunks_indexed,
            )

        flush_batch()

        total_elapsed = time_module.time() - start_time
        log.info("ingestion.completed", total_chunks=total_chunks, elapsed=round(total_elapsed, 1))
        return total_chunks

    def _selected_sources(self, source_names: Iterable[str] | None):
        if source_names is None:
            return self.settings.sources

        requested_names = tuple(name.strip() for name in source_names if name.strip())
        if not requested_names:
            log.error("ingestion.source_selection.no_sources")
            raise ValueError("At least one documentation source must be selected.")

        sources_by_name = {source.name: source for source in self.settings.sources}
        unknown_names = sorted(set(requested_names) - set(sources_by_name))
        if unknown_names:
            available_names = ", ".join(sources_by_name)
            log.error("ingestion.source_selection.unknown", unknown=unknown_names, available=available_names)
            raise ValueError(
                f"Unknown documentation source(s): {', '.join(unknown_names)}. Available sources: {available_names}"
            )

        return tuple(sources_by_name[name] for name in requested_names)

    @staticmethod
    def _compute_content_hash(text: str) -> str:
        """Compute a deterministic SHA-256 hash of the parsed document text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _get_stored_content_hash(self, url: str) -> str | None:
        """Retrieve the content hash stored in the vector store for a given URL.

        Returns None if the vector store does not support this lookup or if no
        chunks exist for the given URL.
        """
        lookup = getattr(self.vector_store, "get_content_hash_for_url", None)
        if lookup is None:
            return None
        return lookup(url)

    def _delete_chunks_for_url(self, url: str) -> None:
        """Remove all chunks for a given URL from the vector store.

        Uses the vector store's delete_by_url if available; otherwise silently
        no-ops (graceful degradation for stores that don't support it).
        """
        deleter = getattr(self.vector_store, "delete_by_url", None)
        if deleter is not None:
            deleter(url)
        else:
            log.debug(
                "vector_store.no_delete_by_url",
                url=url,
            )

    def _emit(self, on_event: Callable[[IngestionEvent], None] | None, event: IngestionEvent) -> None:
        if on_event is not None:
            on_event(event)
