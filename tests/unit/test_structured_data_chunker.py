"""Unit tests for the JSON structured-data chunker."""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from data_engineering_copilot.domain.models import ParsedDocument
from data_engineering_copilot.infrastructure.token_budget import count_tokens
from data_engineering_copilot.services.structured_data_chunker import StructuredDataChunker


def _doc(text: str, **kwargs) -> ParsedDocument:
    return ParsedDocument(
        source_name="json-test",
        title="Structured Doc",
        url="https://example.com/data/config.json",
        text=text,
        doc_type=kwargs.get("doc_type", "json"),
        language=kwargs.get("language", ""),
    )


def _chunk(doc: ParsedDocument, chunker: StructuredDataChunker | None = None) -> list:
    chunker = chunker or StructuredDataChunker()
    return asyncio.run(chunker.chunk(doc))


class TestStructuredDataChunker:
    def test_small_object_emits_single_bounded_chunk(self):
        doc = _doc('{"name": "Spark"}')
        chunks = _chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].section_header == "$"
        assert json.loads(chunks[0].text) == {"name": "Spark"}

    def test_flat_object_emits_one_chunk_per_key_when_oversized(self):
        doc = _doc('{"name": "Spark", "version": "4.0", "engine": "unified"}')
        chunker = StructuredDataChunker(max_chars=30)
        chunks = _chunk(doc, chunker)
        assert len(chunks) == 3
        by_path = {c.section_header: c for c in chunks}
        assert json.loads(by_path["$.name"].text) == "Spark"
        assert json.loads(by_path["$.version"].text) == "4.0"
        assert json.loads(by_path["$.engine"].text) == "unified"

    def test_every_chunk_carries_json_path_in_section_header(self):
        doc = _doc('{"users": [{"name": "a"}, {"name": "b"}]}')
        chunks = _chunk(doc)
        assert chunks
        assert all(c.section_header.startswith("$") for c in chunks)

    def test_nested_paths_preserved_with_small_budget(self):
        doc = _doc('{"a": {"b": {"c": 1, "d": 2}}}')
        chunker = StructuredDataChunker(max_chars=10)
        chunks = _chunk(doc, chunker)
        assert len(chunks) == 2
        paths = {c.section_header for c in chunks}
        assert paths == {"$.a.b.c", "$.a.b.d"}
        for c in chunks:
            assert json.loads(c.text) in (1, 2)

    def test_array_rows_emitted_per_element_with_small_budget(self):
        doc = _doc('[{"id": 1}, {"id": 2}, {"id": 3}]')
        chunker = StructuredDataChunker(max_chars=20)
        chunks = _chunk(doc, chunker)
        ids = sorted(json.loads(c.text)["id"] for c in chunks)
        assert ids == [1, 2, 3]
        paths = {c.section_header for c in chunks}
        assert any(p == "$[0]" for p in paths)

    def test_scalar_root(self):
        doc = _doc("42")
        chunks = _chunk(doc)
        assert len(chunks) == 1
        assert json.loads(chunks[0].text) == 42
        assert chunks[0].section_header == "$"

    def test_string_root(self):
        doc = _doc('"hello"')
        chunks = _chunk(doc)
        assert len(chunks) == 1
        assert json.loads(chunks[0].text) == "hello"

    def test_empty_json_returns_no_chunks(self):
        for text in ("", "   ", "\n\n"):
            assert _chunk(_doc(text)) == []

    def test_malformed_json_falls_back_to_lossless_text_chunks(self):
        doc = _doc('{"broken": 12, "also":')
        chunks = _chunk(doc)
        assert chunks, "malformed JSON must not be silently discarded"
        assert "".join(c.text for c in chunks) == '{"broken": 12, "also":'

    def test_oversized_value_content_preserved_and_budget_safe(self):
        big = "x" * 5000
        doc = _doc(json.dumps({"key": big}))
        chunker = StructuredDataChunker(max_chars=4000)
        chunks = _chunk(doc, chunker)
        assert "".join(json.loads(c.text) for c in chunks) == big
        for c in chunks:
            assert len(c.text) <= chunker.max_chars
            assert count_tokens(c.text) <= chunker.max_tokens

    def test_oversized_nested_value_route_through_dict_then_string(self):
        big = "y" * 3000
        doc = _doc(json.dumps({"outer": {"inner": [big, 1]}}))
        chunker = StructuredDataChunker(max_chars=2500)
        chunks = _chunk(doc, chunker)
        recovered = [json.loads(c.text) for c in chunks]
        assert "".join(x for x in recovered if isinstance(x, str)) == big
        assert 1 in recovered
        for c in chunks:
            assert len(c.text) <= chunker.max_chars

    def test_metadata_shape(self):
        doc = _doc('{"name": "Spark", "version": "4.0"}')
        chunks = _chunk(doc)
        for i, c in enumerate(chunks):
            assert c.chunk_type == "structured"
            assert c.chunk_index == i
            assert c.total_chunks == len(chunks)
            assert c.content_hash
            assert c.token_count > 0
            assert c.character_count == len(c.text)

    def test_deterministic_chunk_ids_and_hashes(self):
        doc = _doc('{"name": "Spark", "version": "4.0"}')
        first = _chunk(doc)
        second = _chunk(doc)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert [c.content_hash for c in first] == [c.content_hash for c in second]

    def test_content_hash_matches_sha256_of_text(self):
        doc = _doc('{"name": "Spark"}')
        chunks = _chunk(doc)
        for c in chunks:
            assert c.content_hash == hashlib.sha256(c.text.encode("utf-8")).hexdigest()

    def test_doc_metadata_passthrough(self):
        doc = _doc('{"name": "Spark"}', doc_type="json", language="json")
        chunks = _chunk(doc)
        assert chunks[0].doc_type == "json"
        assert chunks[0].language == "json"
        assert chunks[0].source_name == "json-test"
        assert chunks[0].url == "https://example.com/data/config.json"

    def test_extract_sentences_unsupported(self):
        assert StructuredDataChunker().extract_sentences("any text") is None

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            StructuredDataChunker(max_tokens=0)
        with pytest.raises(ValueError):
            StructuredDataChunker(max_chars=0)
