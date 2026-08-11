"""Live 7-stage ingestion inspector for the RAG visualizer.

Replays a single documentation page through the production pipeline stages —
Raw HTML -> Markdown -> Chunking -> Quality filter -> Enrichment -> Embeddings
-> Qdrant payload — and records a per-stage snapshot that the UI can render
side-by-side. The orchestration is deliberately free of Streamlit imports and
takes its pipeline pieces as dependencies so it can be exercised hermetically
with test doubles.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument, RawDocument
from data_engineering_copilot.infrastructure.async_qdrant_store import chunk_to_payload


class ParserProtocol(Protocol):
    def parse(self, raw: RawDocument) -> ParsedDocument | None: ...


class ChunkerProtocol(Protocol):
    async def chunk(self, document: ParsedDocument) -> list[DocumentChunk]: ...


class ChunkFilterProtocol(Protocol):
    def extract(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]: ...


class ApiExtractorProtocol(Protocol):
    def extract(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]: ...


class EnricherProtocol(Protocol):
    async def enrich(self, document: ParsedDocument, chunks: list[DocumentChunk]) -> list[DocumentChunk]: ...


class EmbedderProtocol(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class VectorStoreProtocol(Protocol):
    async def upsert_chunks(self, chunks: Sequence[DocumentChunk], vectors: Sequence[list[float]]) -> None: ...


@dataclass
class LabStage:
    """Snapshot of one pipeline stage for the visualizer."""

    name: str
    input_summary: str
    output_summary: str
    payload: Any = None
    error: str | None = None


@dataclass
class LabTrace:
    """Ordered result of a live pipeline run."""

    stages: list[LabStage] = field(default_factory=list)
    raw_document: RawDocument | None = None
    markdown: ParsedDocument | None = None
    final_chunks: list[DocumentChunk] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    dry_run: bool = True

    def stage(self, name: str) -> LabStage | None:
        return next((s for s in self.stages if s.name == name), None)


def _word_count(text: str) -> int:
    return len(text.split())


class PipelineLab:
    """Stage-by-stage driver for the ingestion visualizer."""

    def __init__(
        self,
        parser: ParserProtocol,
        chunk_filter: ChunkFilterProtocol,
        chunker: ChunkerProtocol,
        api_extractor: ApiExtractorProtocol | None = None,
        enricher: EnricherProtocol | None = None,
        embedder: EmbedderProtocol | None = None,
        vector_store: VectorStoreProtocol | None = None,
        *,
        dry_run: bool = True,
    ) -> None:
        self.parser = parser
        self.chunk_filter = chunk_filter
        self.chunker = chunker
        self.api_extractor = api_extractor
        self.enricher = enricher
        self.embedder = embedder
        self.vector_store = vector_store
        self.dry_run = dry_run
        self._on_stage: Callable[[str], None] | None = None

    async def run(
        self,
        raw: RawDocument,
        on_stage: Callable[[str], None] | None = None,
    ) -> LabTrace:
        """Execute the pipeline stages and return a snapshot trace.

        Every stage is captured even on failure (``error`` set) so the UI can
        show exactly where the pipeline stopped. ``on_stage`` (if given) is
        invoked with the stage name as each stage begins.
        """
        trace = LabTrace(raw_document=raw, dry_run=self.dry_run)
        self._on_stage = on_stage

        # 1. Raw HTML source
        trace.stages.append(
            LabStage(
                name="raw",
                input_summary=raw.url,
                output_summary=f"{len(raw.html):,} chars · {raw.content_type}",
                payload={"url": raw.url, "chars": len(raw.html), "content_type": raw.content_type},
            )
        )
        if on_stage:
            on_stage("raw")

        # 2. Markdown conversion
        parsed = await self._run_stage(
            trace,
            name="markdown",
            fn=self.parser.parse,
            raw=raw,
            success=lambda out: out is not None and bool(out.text),
            output_summary_fn=lambda out: (
                f"{_word_count(out.text)} words · title={out.title!r}" if out is not None else "dropped (<40 words)"
            ),
            payload_fn=lambda out: (
                {"title": out.title, "words": _word_count(out.text), "text": out.text}
                if out is not None
                else {"text": None}
            ),
        )
        if parsed is None:
            return trace
        trace.markdown = parsed

        # 3. Header-aware chunking
        chunks = await self._run_stage(
            trace,
            name="chunk",
            fn=self.chunker.chunk,
            document=parsed,
            success=lambda out: bool(out),
            output_summary_fn=lambda out: f"{len(out)} chunks",
            payload_fn=lambda out: [self._chunk_summary(c) for c in out],
        )
        if not chunks:
            return trace
        trace.final_chunks = list(chunks)

        # 4. Quality filtering — kept vs dropped with reasons.
        before_filter = list(chunks)
        filtered = await self._run_stage(
            trace,
            name="filter",
            fn=self._filter_chunks,
            chunks=before_filter,
            success=lambda out: True,
            output_summary_fn=lambda out: f"kept {len(out)} / {len(before_filter)} chunks",
            payload_fn=lambda out: self._filter_payload(before_filter, out),
        )
        if filtered is not None:
            trace.final_chunks = list(filtered)

        if not trace.final_chunks:
            return trace

        # 5. Metadata enrichment (API extraction + optional contextual prefix)
        chunks_to_enrich = list(trace.final_chunks)
        enriched = await self._run_stage(
            trace,
            name="enrich",
            fn=self._enrich_chunks,
            document=parsed,
            chunks=chunks_to_enrich,
            success=lambda out: bool(out),
            output_summary_fn=lambda out: f"{len(out)} chunks enriched",
            payload_fn=lambda out: [self._chunk_summary(c) for c in out],
        )
        if enriched:
            trace.final_chunks = list(enriched)

        # 6. Vector embeddings (dense; preview only)
        if self.embedder is not None and trace.final_chunks:
            vectors = await self._run_stage(
                trace,
                name="embed",
                fn=self.embedder.embed_texts,
                texts=[c.text for c in trace.final_chunks],
                success=lambda out: bool(out) and len(out) == len(trace.final_chunks),
                output_summary_fn=lambda out: (
                    f"{len(out)} vectors × {len(out[0]) if out else 0} dims" if out else "embedding failed"
                ),
                payload_fn=lambda out: {
                    "count": len(out),
                    "dimension": len(out[0]) if out else 0,
                    "sample": out[0][:8] if out else [],
                },
            )
            if vectors:
                trace.vectors = list(vectors)

        # 7. Qdrant point payload (exact dict; optional live upsert)
        if trace.final_chunks:
            await self._run_stage(
                trace,
                name="qdrant",
                fn=self._store_chunks,
                chunks=trace.final_chunks,
                vectors=trace.vectors,
                success=lambda out: bool(out),
                output_summary_fn=lambda out: (
                    f"preview of {len(out)} points (dry-run)" if self.dry_run else f"upserted {len(out)} points"
                ),
                payload_fn=lambda out: chunk_to_payload(trace.final_chunks[0]),
            )

        return trace

    async def _filter_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        return self.chunk_filter.extract(list(chunks))

    async def _enrich_chunks(self, document: ParsedDocument, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        enriched = list(chunks)
        if self.api_extractor is not None:
            enriched = self.api_extractor.extract(enriched)
        if self.enricher is not None:
            enriched = await self.enricher.enrich(document, enriched)
        return enriched

    async def _store_chunks(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> list[DocumentChunk]:
        if self.dry_run or self.vector_store is None:
            return list(chunks)
        await self.vector_store.upsert_chunks(chunks, vectors)
        return list(chunks)

    def _filter_payload(self, before: list[DocumentChunk], after: list[DocumentChunk]) -> dict[str, Any]:
        after_ids = {c.chunk_id for c in after}
        dropped = [c for c in before if c.chunk_id not in after_ids]
        return {
            "kept": len(after),
            "dropped": len(dropped),
            "reasons": [
                {
                    "chunk_index": c.chunk_index,
                    "word_count": c.word_count,
                    "chunk_type": c.chunk_type,
                }
                for c in dropped
            ],
        }

    @staticmethod
    def _chunk_summary(chunk: DocumentChunk) -> dict[str, Any]:
        heading = chunk.section_header
        if not heading and chunk.heading_path:
            heading = chunk.heading_path[-1]
        return {
            "index": chunk.chunk_index,
            "heading": heading,
            "words": chunk.word_count,
            "tokens": chunk.token_count,
            "type": chunk.chunk_type,
            "text": chunk.text[:120],
        }

    async def _run_stage(
        self,
        trace: LabTrace,
        *,
        name: str,
        fn: Callable[..., Any],
        output_summary_fn: Callable[[Any], str],
        payload_fn: Callable[[Any], Any],
        success: Callable[[Any], bool],
        **kwargs: Any,
    ) -> Any:
        if self._on_stage is not None:
            self._on_stage(name)
        try:
            result = fn(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            error = None if success(result) else "stage produced no output"
            stage = LabStage(
                name=name,
                input_summary=_summarize_kwargs(kwargs),
                output_summary=output_summary_fn(result),
                payload=payload_fn(result),
                error=error,
            )
            trace.stages.append(stage)
            return result
        except Exception as exc:  # noqa: BLE001 - captured for the visualizer
            trace.stages.append(
                LabStage(
                    name=name,
                    input_summary=_summarize_kwargs(kwargs),
                    output_summary="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            return None


def _summarize_kwargs(kwargs: dict[str, Any]) -> str:
    """Compact human-readable input summary for a stage invocation."""
    parts: list[str] = []
    for key, value in kwargs.items():
        if isinstance(value, (RawDocument, ParsedDocument)):
            parts.append(f"{key}={value.url}")
        elif isinstance(value, (list, tuple)):
            parts.append(f"{key}×{len(value)}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)
