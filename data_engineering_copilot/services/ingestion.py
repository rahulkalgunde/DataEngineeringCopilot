from __future__ import annotations

import logging
from typing import Callable, Iterable

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.domain.models import DocumentChunk, IngestionEvent
from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.html_parser import DocumentationHtmlParser
from data_engineering_copilot.infrastructure.vector_store import QdrantVectorStore
from data_engineering_copilot.services.chunker import DocumentChunker


logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        settings: AppSettings,
        crawler: DocumentationCrawler,
        parser: DocumentationHtmlParser,
        chunker: DocumentChunker,
        embeddings: SentenceTransformerEmbeddings,
        vector_store: QdrantVectorStore,
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
        page_limit = max_pages_per_source or self.settings.max_pages_per_source
        total_chunks = 0
        selected_sources = self._selected_sources(source_names)
        logger.info(
            "Ingestion started page_limit=%s sources=%s",
            page_limit,
            [source.name for source in selected_sources],
        )

        batch_chunks: list[DocumentChunk] = []

        def flush_batch() -> None:
            if not batch_chunks:
                return
            try:
                batch_vectors = self.embeddings.embed_texts([chunk.text for chunk in batch_chunks])
            except RuntimeError as exc:
                logger.error(
                    "Ingestion failed to embed batch of %d chunks: %s",
                    len(batch_chunks),
                    exc,
                )
                raise
            try:
                self.vector_store.upsert_chunks(batch_chunks, batch_vectors)
            except Exception as exc:
                logger.error(
                    "Ingestion failed to upsert batch of %d chunks to vector store: %s",
                    len(batch_chunks),
                    exc,
                )
                raise
            batch_chunks.clear()

        for source in selected_sources:
            print(f"Crawling {source.name}")
            logger.info("Ingestion source started source=%s page_limit=%s", source.name, page_limit)
            source_pages_fetched = 0
            source_chunks_indexed = 0
            self._emit(
                on_event,
                IngestionEvent(
                    event_type="source_start",
                    source_name=source.name,
                    message=f"Crawling {source.name}",
                ),
            )
            for raw_document in self.crawler.crawl(source, max_pages=page_limit, on_event=on_event):
                source_pages_fetched += 1
                parsed = self.parser.parse(raw_document)
                if parsed is None:
                    logger.info("Ingestion skipped unreadable page source=%s url=%s", raw_document.source_name, raw_document.url)
                    self._emit(
                        on_event,
                        IngestionEvent(
                            event_type="page_skipped",
                            source_name=raw_document.source_name,
                            url=raw_document.url,
                            message=f"Skipped page with no readable documentation content: {raw_document.url}",
                            pages_fetched=source_pages_fetched,
                        ),
                    )
                    continue
                chunks = self.chunker.chunk(parsed)
                batch_chunks.extend(chunks)

                if len(batch_chunks) >= self.settings.ingestion_batch_chunk_size:
                    flush_batch()

                total_chunks += len(chunks)
                source_chunks_indexed += len(chunks)
                print(f"Indexed {len(chunks):>3} chunks from {parsed.title}")
                logger.info(
                    "Ingestion indexed page source=%s url=%s title=%r chunks=%s total_chunks=%s",
                    parsed.source_name,
                    parsed.url,
                    parsed.title,
                    len(chunks),
                    total_chunks,
                )
                self._emit(
                    on_event,
                    IngestionEvent(
                        event_type="page_indexed",
                        source_name=parsed.source_name,
                        url=parsed.url,
                        title=parsed.title,
                        chunks_indexed=len(chunks),
                        pages_fetched=source_pages_fetched,
                        message=f"Indexed {len(chunks)} chunks from {parsed.title}",
                    ),
                )

            self._emit(
                on_event,
                IngestionEvent(
                    event_type="source_complete",
                    source_name=source.name,
                    chunks_indexed=source_chunks_indexed,
                    pages_fetched=source_pages_fetched,
                    message=(
                        f"Completed {source.name}: fetched {source_pages_fetched} HTML pages, "
                        f"indexed {source_chunks_indexed} chunks."
                    ),
                ),
            )
            logger.info(
                "Ingestion source completed source=%s pages=%s chunks=%s",
                source.name,
                source_pages_fetched,
                source_chunks_indexed,
            )

        flush_batch()

        logger.info("Ingestion completed total_chunks=%s", total_chunks)
        return total_chunks

    def _selected_sources(self, source_names: Iterable[str] | None):
        if source_names is None:
            return self.settings.sources

        requested_names = tuple(name.strip() for name in source_names if name.strip())
        if not requested_names:
            logger.error("Ingestion source selection failed because no source names were selected")
            raise ValueError("At least one documentation source must be selected.")

        sources_by_name = {source.name: source for source in self.settings.sources}
        unknown_names = sorted(set(requested_names) - set(sources_by_name))
        if unknown_names:
            available_names = ", ".join(sources_by_name)
            logger.error("Ingestion source selection failed unknown=%s available=%s", unknown_names, available_names)
            raise ValueError(
                f"Unknown documentation source(s): {', '.join(unknown_names)}. "
                f"Available sources: {available_names}"
            )

        return tuple(sources_by_name[name] for name in requested_names)

    def _emit(self, on_event: Callable[[IngestionEvent], None] | None, event: IngestionEvent) -> None:
        if on_event is not None:
            on_event(event)
