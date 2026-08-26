"""Tests for PreparedSource."""

from __future__ import annotations

from pathlib import Path

from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.services.prepared_source import PreparedSource
from data_engineering_copilot.services.spark_index_builder import CoverageRecord


def _make_chunk(text: str = "test") -> DocumentChunk:
    return DocumentChunk(
        chunk_id="c1",
        source_name="src",
        title="Title",
        url="http://test.com",
        text=text,
        file_path="test.md",
        content_hash="hash123",
    )


def _make_coverage(path: str = "file.md", hash_val: str = "abc123") -> CoverageRecord:
    return CoverageRecord(
        relative_path=path,
        representation="native",
        doc_type="text",
        canonical_url="http://x",
        status="indexed",
        chunk_count=1,
        content_hash=hash_val,
    )


class TestPreparedSource:
    def test_provenance_sources_maps_hash(self) -> None:
        cov = [_make_coverage("dir/file.md", "hash1234567890")]
        src = PreparedSource(
            slug="test",
            source_name="Test",
            generation="gen",
            commit="abc",
            chunks=(_make_chunk(),),
            coverage=cov,
            cache_root=Path("/tmp/cache"),
        )
        result = src.provenance_sources()
        assert "hash1234567890" not in result.values()
        assert "hash1234567890"[:12] in result.values()

    def test_provenance_skips_empty_hash(self) -> None:
        cov = [_make_coverage("file.md", "")]
        src = PreparedSource(
            slug="test",
            source_name="Test",
            generation="gen",
            commit="abc",
            chunks=(),
            coverage=cov,
            cache_root=Path("/tmp/cache"),
        )
        result = src.provenance_sources()
        assert len(result) == 0

    def test_provenance_uses_relative_path(self) -> None:
        cov = [_make_coverage("sub/file.md", "hash1234567890")]
        src = PreparedSource(
            slug="test",
            source_name="Test",
            generation="gen",
            commit="abc",
            chunks=(),
            coverage=cov,
            cache_root=Path("/tmp/cache"),
        )
        result = src.provenance_sources()
        assert any("file.md" in k for k in result)
