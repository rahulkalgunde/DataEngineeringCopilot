"""Unit tests for the metadata-based chunker router."""

from __future__ import annotations

from data_engineering_copilot.domain.models import ParsedDocument
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.services.chunker_router import ChunkerRouter
from data_engineering_copilot.services.header_aware_chunker import HeaderAwareChunker
from data_engineering_copilot.services.spark_chunker import SparkChunker
from data_engineering_copilot.services.structured_data_chunker import StructuredDataChunker


def _doc(url: str = "https://example.com/guide", text: str = "Hello world.", **kwargs) -> ParsedDocument:
    return ParsedDocument(
        source_name="test",
        title="Test",
        url=url,
        text=text,
        doc_type=kwargs.get("doc_type", ""),
        language=kwargs.get("language", ""),
        file_path=kwargs.get("file_path", ""),
    )


def _router(spark: bool = False) -> ChunkerRouter:
    return ChunkerRouter(
        generic_strategy=DocumentChunker(),
        structured_strategy=StructuredDataChunker(),
        code_strategy=DocumentChunker(),
        guide_strategy=HeaderAwareChunker(),
        spark_chunker=SparkChunker(header_chunker=HeaderAwareChunker()) if spark else None,
    )


class TestChunkerRouter:
    def test_spark_doc_type_routes_first(self):
        route = _router(spark=True).route(_doc(doc_type="api_reference"))
        assert route.key == "spark"

    def test_spark_requires_wired_spark_chunker(self):
        route = _router(spark=False).route(_doc(doc_type="api_reference"))
        assert route.key != "spark"

    def test_json_url_routes_to_structured(self):
        route = _router().route(_doc(url="https://example.com/data/config.json"))
        assert route.key == "structured"

    def test_json_doc_type_routes_to_structured(self):
        route = _router().route(_doc(doc_type="json"))
        assert route.key == "structured"

    def test_json_text_sniff_routes_to_structured(self):
        route = _router().route(_doc(url="https://example.com/data", text='{"a": 1}'))
        assert route.key == "structured"

    def test_large_malformed_brace_text_not_structured(self):
        text = "{" + ("not json " * 3000)
        route = _router().route(_doc(url="https://example.com/data", text=text))
        assert route.key != "structured"

    def test_explicit_code_language_routes_to_code(self):
        route = _router().route(_doc(language="scala"))
        assert route.key == "code"

    def test_code_url_routes_to_code(self):
        route = _router().route(_doc(url="https://example.com/api/scala/foo.html"))
        assert route.key == "code"

    def test_code_file_path_routes_to_code(self):
        route = _router().route(_doc(file_path="sources/Foo.scala"))
        assert route.key == "code"

    def test_markdown_url_routes_to_guide(self):
        route = _router().route(_doc(url="https://example.com/guide.md"))
        assert route.key == "guide"

    def test_rst_url_routes_to_guide(self):
        route = _router().route(_doc(url="https://example.com/guide.rst"))
        assert route.key == "guide"

    def test_guide_doc_type_routes_to_guide(self):
        route = _router().route(_doc(doc_type="guide"))
        assert route.key == "guide"

    def test_unknown_metadata_routes_to_generic(self):
        route = _router().route(_doc())
        assert route.key == "generic"
        assert isinstance(route.strategy, DocumentChunker)

    def test_precedence_spark_over_structured_over_code_over_guide(self):
        with_spark = _router(spark=True)
        doc = _doc(
            url="https://example.com/api/python/data.json",
            doc_type="api_reference",
            language="python",
            text='{"a": 1}',
        )
        assert with_spark.route(doc).key == "spark"
        assert _router().route(doc).key == "structured"

        code_and_guide = _doc(url="https://example.com/guide.md", language="java")
        assert _router().route(code_and_guide).key == "code"

    def test_route_carries_reason(self):
        route = _router().route(_doc(url="https://example.com/guide.md"))
        assert route.reason

    def test_route_is_deterministic(self):
        doc = _doc(doc_type="json")
        router = _router()
        assert router.route(doc).key == router.route(doc).key
