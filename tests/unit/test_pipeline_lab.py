"""Hermetic tests for the Pipeline Lab stage-by-stage inspector.

Exercises ``PipelineLab.run`` with fake pipeline pieces (no infra, no
Streamlit) to pin the 7-stage snapshot contract: ordered stages, per-stage
summaries/payloads, error capture, dry-run vs live upsert behaviour, and the
exact Qdrant point payload for the final chunk.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable

from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument, RawDocument
from data_engineering_copilot.infrastructure.async_qdrant_store import chunk_to_payload
from data_engineering_copilot.services.pipeline_lab import PipelineLab

ALL_STAGES = ["raw", "markdown", "chunk", "filter", "enrich", "embed", "qdrant"]


def _chunk(source_name: str, index: int, total: int = 3) -> DocumentChunk:
    text = f"chunk {index} with some words for testing " * 3
    return DocumentChunk(
        chunk_id=f"{source_name}-{index}",
        source_name=source_name,
        title="Fake Page",
        url="https://example.com/fake",
        text=text,
        word_count=len(text.split()),
        section_header=f"Section {index}",
        chunk_type="text",
        chunk_index=index,
        total_chunks=total,
    )


class _FakeParser:
    def __init__(self, *, drop: bool = False) -> None:
        self._drop = drop

    def parse(self, raw: RawDocument) -> ParsedDocument | None:
        if self._drop:
            return None
        return ParsedDocument(
            source_name=raw.source_name,
            title="Fake Page",
            url=raw.url,
            text="Lots of documentation words for the fake page. " * 6,
        )


class _FakeChunker:
    async def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
        return [_chunk(document.source_name, i) for i in range(3)]


class _DroppingFilter:
    def extract(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        return [c for c in chunks if c.chunk_index != 1]


class _PassFilter:
    def extract(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        return list(chunks)


class _FakeEnricher:
    async def enrich(self, document: ParsedDocument, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        return [dataclasses.replace(c, chunk_type="api") for c in chunks]


class _FakeEmbedder:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeStore:
    def __init__(self) -> None:
        self.upserts: list[tuple[list[DocumentChunk], list[list[float]]]] = []

    async def upsert_chunks(self, chunks, vectors) -> None:
        self.upserts.append((list(chunks), [list(v) for v in vectors]))


def _build_lab(**overrides) -> PipelineLab:
    defaults = {
        "parser": _FakeParser(),
        "chunk_filter": _PassFilter(),
        "chunker": _FakeChunker(),
        "api_extractor": None,
        "enricher": _FakeEnricher(),
        "embedder": _FakeEmbedder(),
        "vector_store": _FakeStore(),
        "dry_run": True,
    }
    defaults.update(overrides)
    return PipelineLab(**defaults)


def _raw(**overrides) -> RawDocument:
    defaults = {
        "source_name": "pipeline-lab",
        "url": "https://example.com/page",
        "html": "<html><h1>Fake</h1><p>Some documentation prose.</p></html>",
    }
    defaults.update(overrides)
    return RawDocument(**defaults)


def _run(
    lab: PipelineLab,
    raw: RawDocument,
    on_stage: Callable[[str], None] | None = None,
):
    return asyncio.run(lab.run(raw, on_stage=on_stage))


class TestPipelineLab:
    def test_all_seven_stages_recorded_in_order(self) -> None:
        events: list[str] = []
        trace = _run(_build_lab(), _raw(), on_stage=events.append)

        assert [s.name for s in trace.stages] == ALL_STAGES
        assert events == ALL_STAGES
        assert all(s.error is None for s in trace.stages)

    def test_stage_snapshots_carry_input_output_summaries(self) -> None:
        trace = _run(_build_lab(), _raw())

        raw_stage = trace.stage("raw")
        assert raw_stage is not None
        assert "chars" in raw_stage.output_summary
        assert raw_stage.payload == {
            "url": "https://example.com/page",
            "chars": len(_raw().html),
            "content_type": "text/html",
        }

        markdown_stage = trace.stage("markdown")
        assert markdown_stage is not None
        assert "words" in markdown_stage.output_summary
        assert markdown_stage.payload["title"] == "Fake Page"

        chunk_stage = trace.stage("chunk")
        assert chunk_stage is not None
        assert "3 chunks" in chunk_stage.output_summary
        assert len(chunk_stage.payload) == 3

    def test_filter_reports_kept_and_dropped(self) -> None:
        trace = _run(_build_lab(chunk_filter=_DroppingFilter()), _raw())

        filter_stage = trace.stage("filter")
        assert filter_stage is not None
        assert filter_stage.payload["kept"] == 2
        assert filter_stage.payload["dropped"] == 1
        reasons = filter_stage.payload["reasons"]
        assert len(reasons) == 1
        assert reasons[0]["chunk_index"] == 1

    def test_enrichment_applied_to_final_chunks(self) -> None:
        trace = _run(_build_lab(), _raw())

        assert len(trace.final_chunks) == 3
        assert all(c.chunk_type == "api" for c in trace.final_chunks)
        enrich_stage = trace.stage("enrich")
        assert enrich_stage is not None
        assert "3 chunks enriched" in enrich_stage.output_summary

    def test_embedding_stage_reports_count_and_dimension(self) -> None:
        trace = _run(_build_lab(), _raw())

        embed_stage = trace.stage("embed")
        assert embed_stage is not None
        assert embed_stage.payload["count"] == 3
        assert embed_stage.payload["dimension"] == 3
        assert len(trace.vectors) == 3

    def test_qdrant_stage_payload_is_exact_point_payload(self) -> None:
        trace = _run(_build_lab(), _raw())

        qdrant_stage = trace.stage("qdrant")
        assert qdrant_stage is not None
        assert qdrant_stage.payload == chunk_to_payload(trace.final_chunks[0])
        assert "dry-run" in qdrant_stage.output_summary

    def test_dry_run_never_touches_vector_store(self) -> None:
        store = _FakeStore()
        _run(_build_lab(vector_store=store, dry_run=True), _raw())
        assert store.upserts == []

    def test_live_mode_upserts_chunks_and_vectors(self) -> None:
        store = _FakeStore()
        trace = _run(_build_lab(vector_store=store, dry_run=False), _raw())

        assert len(store.upserts) == 1
        upserted_chunks, upserted_vectors = store.upserts[0]
        assert [c.chunk_id for c in upserted_chunks] == [c.chunk_id for c in trace.final_chunks]
        assert upserted_vectors == [[0.1, 0.2, 0.3]] * 3
        qdrant_stage = trace.stage("qdrant")
        assert qdrant_stage is not None
        assert "upserted" in qdrant_stage.output_summary

    def test_parser_drop_short_circuits_after_markdown(self) -> None:
        trace = _run(_build_lab(parser=_FakeParser(drop=True)), _raw())

        assert [s.name for s in trace.stages] == ["raw", "markdown"]
        markdown_stage = trace.stage("markdown")
        assert markdown_stage is not None
        assert markdown_stage.error is not None
        assert trace.final_chunks == []

    def test_empty_chunks_short_circuit_after_chunking(self) -> None:
        class _EmptyChunker(_FakeChunker):
            async def chunk(self, document: ParsedDocument) -> list[DocumentChunk]:
                return []

        trace = _run(_build_lab(chunker=_EmptyChunker()), _raw())

        assert [s.name for s in trace.stages] == ["raw", "markdown", "chunk"]
        chunk_stage = trace.stage("chunk")
        assert chunk_stage is not None
        assert chunk_stage.error is not None

    def test_stage_exception_is_captured_not_raised(self) -> None:
        class _BoomEmbedder(_FakeEmbedder):
            async def embed_texts(self, texts: list[str]) -> list[list[float]]:
                raise RuntimeError("embedding exploded")

        trace = _run(_build_lab(embedder=_BoomEmbedder()), _raw())

        embed_stage = trace.stage("embed")
        assert embed_stage is not None
        assert embed_stage.error is not None
        assert "RuntimeError" in embed_stage.error
        assert trace.stage("qdrant") is not None  # pipeline continued

    def test_optional_enricher_skipped_when_none(self) -> None:
        trace = _run(_build_lab(enricher=None), _raw())

        enrich_stage = trace.stage("enrich")
        assert enrich_stage is not None
        assert enrich_stage.error is None
        assert trace.final_chunks and all(c.chunk_type == "text" for c in trace.final_chunks)
