"""Tests for protocol interfaces and protocol conformance."""

from __future__ import annotations

import typing
from collections.abc import Iterable
from typing import Any

import pytest

from data_engineering_copilot.domain.models import (
    DocumentChunk,
    ParsedDocument,
    RagConfig,
    RawDocument,
    RetrievedChunk,
)
from data_engineering_copilot.domain.protocols import (
    ChunkerProtocol,
    CrawlerProtocol,
    EmbedderProtocol,
    LLMClientProtocol,
    ParserProtocol,
    RerankerProtocol,
    TelemetryTracerProtocol,
    VectorStoreProtocol,
)


def create_sample_document() -> RawDocument:
    return RawDocument(
        source_name="test_source",
        url="http://example.com/test",
        html="<html><body><h1>Test</h1><p>Test content</p></body></html>",
    )


def create_parsed_document() -> ParsedDocument:
    return ParsedDocument(
        source_name="test_source",
        title="Test",
        url="http://example.com/test",
        text="Test content",
    )


def create_chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="test_chunk",
        source_name="test_source",
        title="Test",
        url="http://example.com/test",
        text="Test content",
        content_hash="abc123",
    )


def create_retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(chunk=create_chunk(), distance=0.1, confidence=0.9)


class TestProtocolStructure:
    """Test that protocols are correctly defined."""

    def test_protocols_are_importable(self):
        assert CrawlerProtocol is not None
        assert ParserProtocol is not None
        assert ChunkerProtocol is not None
        assert EmbedderProtocol is not None
        assert VectorStoreProtocol is not None
        assert LLMClientProtocol is not None
        assert RerankerProtocol is not None
        assert TelemetryTracerProtocol is not None

    def test_crawler_protocol_has_crawl_method(self):
        assert hasattr(CrawlerProtocol, "crawl")
        hints = typing.get_type_hints(CrawlerProtocol.crawl)
        assert "source" in hints
        assert "max_pages" in hints
        assert "on_event" in hints
        assert hints["return"] == Iterable[RawDocument]

    def test_parser_protocol_has_parse_method(self):
        assert hasattr(ParserProtocol, "parse")
        hints = typing.get_type_hints(ParserProtocol.parse)
        assert "raw" in hints
        assert hints["return"] == ParsedDocument

    def test_chunker_protocol_has_chunk_method(self):
        assert hasattr(ChunkerProtocol, "chunk")
        hints = typing.get_type_hints(ChunkerProtocol.chunk)
        assert "document" in hints
        assert hints["return"] == list[DocumentChunk]

    def test_embedder_protocol_has_embed_methods(self):
        assert hasattr(EmbedderProtocol, "embed_texts")
        assert hasattr(EmbedderProtocol, "embed_query")
        embed_texts_hints = typing.get_type_hints(EmbedderProtocol.embed_texts)
        assert "texts" in embed_texts_hints
        assert embed_texts_hints["return"] == list[list[float]]
        embed_query_hints = typing.get_type_hints(EmbedderProtocol.embed_query)
        assert "text" in embed_query_hints
        assert embed_query_hints["return"] == list[float]

    def test_vector_store_protocol_has_required_methods(self):
        assert hasattr(VectorStoreProtocol, "upsert_chunks")
        assert hasattr(VectorStoreProtocol, "query")
        assert hasattr(VectorStoreProtocol, "count")
        assert hasattr(VectorStoreProtocol, "get_content_hash_for_url")
        assert hasattr(VectorStoreProtocol, "delete_by_url")

    def test_llm_client_protocol_has_generate_method(self):
        assert hasattr(LLMClientProtocol, "generate")
        hints = typing.get_type_hints(LLMClientProtocol.generate)
        assert "prompt" in hints
        assert hints["return"] is str

    def test_reranker_protocol_has_required_methods(self):
        assert hasattr(RerankerProtocol, "rerank")
        assert hasattr(RerankerProtocol, "is_available")

    def test_telemetry_tracer_protocol_has_required_methods(self):
        assert hasattr(TelemetryTracerProtocol, "start_observation")
        assert hasattr(TelemetryTracerProtocol, "flush")


class TestDomainIsolation:
    """Test domain layer clean isolation from outer config/infra layers."""

    def test_domain_protocols_does_not_import_config(self):
        import inspect
        import sys

        # Remove domain and config modules if cached to inspect imports
        sys.modules.pop("data_engineering_copilot.domain.protocols", None)

        import data_engineering_copilot.domain.protocols as proto_module

        # Read module source directly to ensure no config imports
        source = inspect.getsource(proto_module)
        assert "data_engineering_copilot.config" not in source, (
            "domain.protocols must not import from data_engineering_copilot.config"
        )

    def test_domain_configs_exist(self):
        from data_engineering_copilot.config.settings import DocumentationSource

        source_cfg = DocumentationSource(
            name="pandas",
            start_urls=("https://pandas.pydata.org",),
            allowed_domains=("pandas.pydata.org",),
        )
        assert source_cfg.name == "pandas"

        rag_cfg = RagConfig(retrieval_top_k=5, confidence_threshold=0.3)
        assert rag_cfg.retrieval_top_k == 5


class TestProtocolConformance:
    """Test that mock objects can conform to protocols."""

    def test_mock_crawler_can_conform(self):
        from unittest.mock import MagicMock

        mock = MagicMock(spec=CrawlerProtocol)
        mock.crawl.return_value = [create_sample_document()]
        assert hasattr(mock, "crawl")
        result = list(mock.crawl("source", 10))
        assert len(result) == 1

    def test_mock_parser_can_conform(self):
        from unittest.mock import MagicMock

        mock = MagicMock(spec=ParserProtocol)
        mock.parse.return_value = create_parsed_document()
        assert hasattr(mock, "parse")
        result = mock.parse(create_sample_document())
        assert result.title == "Test"

    def test_mock_chunker_can_conform(self):
        from unittest.mock import MagicMock

        mock = MagicMock(spec=ChunkerProtocol)
        mock.chunk.return_value = [create_chunk()]
        assert hasattr(mock, "chunk")
        result = mock.chunk(create_parsed_document())
        assert result[0].chunk_id == "test_chunk"

    def test_mock_embedder_can_conform(self):
        from unittest.mock import MagicMock

        mock = MagicMock(spec=EmbedderProtocol)
        mock.embed_texts.return_value = [[0.1, 0.2, 0.3]]
        mock.embed_query.return_value = [0.1, 0.2, 0.3]
        assert hasattr(mock, "embed_texts")
        assert hasattr(mock, "embed_query")

    def test_mock_vector_store_can_conform(self):
        from unittest.mock import MagicMock

        mock = MagicMock(spec=VectorStoreProtocol)
        mock.query.return_value = [create_retrieved_chunk()]
        mock.count.return_value = 42
        assert hasattr(mock, "query")
        assert hasattr(mock, "count")

    def test_mock_llm_can_conform(self):
        from unittest.mock import MagicMock

        mock = MagicMock(spec=LLMClientProtocol)
        mock.generate.return_value = "Generated answer"
        assert hasattr(mock, "generate")


class TestProtocolUsage:
    """Test that protocol types work in function annotations."""

    def test_protocol_type_hint_works(self):
        def process_document(chunker: ChunkerProtocol, document: Any) -> Any:
            return document

        result = process_document(None, create_parsed_document())
        assert result is not None

    def test_protocol_uninstantiable(self):
        with pytest.raises(TypeError):
            CrawlerProtocol()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
