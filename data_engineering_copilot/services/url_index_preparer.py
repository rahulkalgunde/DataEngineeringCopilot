"""Generic ``llms.txt`` url-index source preparer: resolve → parse → chunk."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import httpx

from data_engineering_copilot.config.settings import PinnedSourceConfig
from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument
from data_engineering_copilot.services.claude_docs_ingestion import strip_frontmatter
from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
from data_engineering_copilot.services.prepared_source import PreparedSource
from data_engineering_copilot.services.spark_index_builder import CoverageRecord
from data_engineering_copilot.services.url_index_resolver import UrlIndexManifest, UrlIndexResolver

_MIN_DOC_CHARS = 100


class UrlIndexPreparer:
    """Fetch a pinned ``url_index`` source and chunk it into a ``PreparedSource``.

    Parameters
    ----------
    config:
        A ``PinnedSourceConfig`` of type ``url_index``.
    cache_dir:
        Directory under which the source's ``cache_dir`` subtree is materialized.
    generation:
        Generation ID stamped on every produced chunk.
    header_chunker:
        Injected header chunker (defaults to ``HeaderAwareChunker``).
    """

    def __init__(
        self,
        config: PinnedSourceConfig,
        cache_dir: Path,
        generation: str,
        header_chunker: HeaderAwareChunker | None = None,
    ) -> None:
        self._config = config
        self._cache_dir = Path(cache_dir)
        self._generation = generation
        self._header_chunker = header_chunker or HeaderAwareChunker()

    async def prepare(self, client: httpx.AsyncClient | None = None) -> PreparedSource:
        """Resolve the index, parse every page, and chunk the corpus."""
        resolver = UrlIndexResolver(self._config, self._cache_dir)
        manifest = await resolver.resolve(client=client)
        documents = self._build_documents(manifest)
        chunks: list[DocumentChunk] = []
        coverage: list[CoverageRecord] = []
        for doc in documents:
            doc_chunks = await self._header_chunker.chunk(doc)
            doc_chunks = [self._attach_metadata(chunk) for chunk in doc_chunks]
            chunks.extend(doc_chunks)
            coverage.append(
                CoverageRecord(
                    relative_path=doc.file_path,
                    representation="web",
                    doc_type=doc.doc_type,
                    canonical_url=doc.url,
                    status="indexed" if doc_chunks else "no_content",
                    chunk_count=len(doc_chunks),
                    content_hash="",
                    failure_reason="" if doc_chunks else "parsed to empty text",
                )
            )
        return PreparedSource(
            slug=self._config.slug,
            source_name=self._config.name,
            generation=self._generation,
            commit="",
            chunks=tuple(chunks),
            coverage=tuple(coverage),
        )

    def _build_documents(self, manifest: UrlIndexManifest) -> list[ParsedDocument]:
        documents: list[ParsedDocument] = []
        for entry in manifest.entries:
            path = manifest.root / entry.relative_path
            if not path.is_file():
                continue
            text = strip_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
            if len(text.strip()) < _MIN_DOC_CHARS:
                continue
            documents.append(
                ParsedDocument(
                    source_name=manifest.source_name,
                    title=entry.title,
                    url=entry.url,
                    text=text,
                    sections=(),
                    doc_type=self._config.doc_type,
                    language="conceptual",
                    spark_version="",
                    module="",
                    source_commit="",
                    file_path=entry.relative_path,
                    license="",
                )
            )
        return documents

    def _attach_metadata(self, chunk: DocumentChunk) -> DocumentChunk:
        return replace(
            chunk,
            doc_type=chunk.doc_type or self._config.doc_type,
            language=chunk.language or "conceptual",
            source_commit="",
            file_path=chunk.file_path,
            license="",
            index_generation=self._generation,
            chunker_version="header-aware-v1",
        )
