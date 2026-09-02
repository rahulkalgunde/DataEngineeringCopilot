"""Tests for per-source provenance writer."""

from __future__ import annotations

import json
from pathlib import Path

from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.services.prepared_source import PreparedSource
from data_engineering_copilot.services.source_manifest import (
    SourceProvenance,
    build_source_provenance,
    read_source_provenance,
    write_all_source_provenances,
    write_source_provenance,
)


def _make_chunk(chunk_id: str = "c1", text: str = "test") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_name="test",
        title="Test",
        url="http://test.com",
        text=text,
        file_path="test.md",
        content_hash="hash123",
    )


def test_source_provenance_roundtrip() -> None:
    """A provenance object can be written and read back identically."""
    prov = SourceProvenance(
        source_name="Test Source",
        slug="test-source",
        generation="test-gen-001",
        commit_hash="abc123def456",
        manifest_hash="manifest123",
        chunk_count=42,
        source_type="github",
        timestamp="2026-01-01T12:00:00Z",
        source_files={"src/file1.py": "aaa111bbb222", "src/file2.py": "ccc333ddd444"},
        generator="test-generator",
    )

    data = prov.to_dict()
    # Check top-level schema for backward compatibility
    assert "generated_at" in data
    assert "generator" in data
    assert "generation" in data
    assert "source" in data
    assert "sources" in data

    # Check source sub-dict
    source = data["source"]
    assert source["slug"] == "test-source"
    assert source["name"] == "Test Source"
    assert source["type"] == "github"
    assert source["commit"] == "abc123def456"
    assert source["manifest_hash"] == "manifest123"
    assert source["chunk_count"] == 42

    # Check top-level sources mapping (for check_derived_staleness.py)
    assert data["sources"] == {"src/file1.py": "aaa111bbb222", "src/file2.py": "ccc333ddd444"}

    # Write and read back
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        prov_path = write_source_provenance(prov, Path(tmpdir))
        read_back = read_source_provenance(prov_path)
        assert read_back == prov


def test_build_source_provenance_from_prepared_source() -> None:
    """Building provenance from a PreparedSource yields correct values."""
    pkg = PreparedSource(
        slug="test-source",
        source_name="Test Source",
        generation="test-gen-001",
        commit="abc123def456",
        chunks=(_make_chunk("c1"), _make_chunk("c2"), _make_chunk("c3")),
        coverage=(),
        cache_root=Path("/tmp"),
        manifest_hash="manifest123",
    )

    prov = build_source_provenance(pkg, generation="test-gen-001", source_type="github")

    assert prov.source_name == "Test Source"
    assert prov.slug == "test-source"
    assert prov.generation == "test-gen-001"
    assert prov.commit_hash == "abc123def456"
    assert prov.manifest_hash == "manifest123"
    assert prov.chunk_count == 3
    assert prov.source_type == "github"
    assert prov.source_files == {}  # empty coverage -> empty provenance_sources
    assert prov.generator == "pinned-index-builder"


def test_write_all_source_provenances_creates_files(tmp_path: Path) -> None:
    """Writing multiple sources creates the expected files."""
    pkg1 = PreparedSource(
        slug="spark",
        source_name="Apache Spark",
        generation="gen-001",
        commit="abc123",
        chunks=(_make_chunk("c1"),),
        coverage=(),
        cache_root=Path("."),
        manifest_hash="spark-manifest",
    )
    pkg2 = PreparedSource(
        slug="claude",
        source_name="Claude Docs",
        generation="gen-001",
        commit="def456",
        chunks=(_make_chunk("c1"), _make_chunk("c2")),
        coverage=(),
        cache_root=Path("."),
        manifest_hash="claude-manifest",
    )

    source_type_map = {"spark": "github", "claude": "url_index"}
    written = write_all_source_provenances(
        packages=[pkg1, pkg2],
        generation="gen-001",
        output_dir=tmp_path,
        source_type_map=source_type_map,
        generator="test-suite",
    )

    # Should have created two files
    assert len(written) == 2
    assert (tmp_path / "provenance-spark.json").exists()
    assert (tmp_path / "provenance-claude.json").exists()
    assert written == [
        tmp_path / "provenance-spark.json",
        tmp_path / "provenance-claude.json",
    ]

    # Check Spark provenance
    spark_prov = read_source_provenance(written[0])
    assert spark_prov.source_name == "Apache Spark"
    assert spark_prov.slug == "spark"
    assert spark_prov.generator == "test-suite"
    assert spark_prov.source_type == "github"

    # Check Claude provenance
    claude_prov = read_source_provenance(written[1])
    assert claude_prov.source_name == "Claude Docs"
    assert claude_prov.slug == "claude"
    assert claude_prov.generator == "test-suite"
    assert claude_prov.source_type == "url_index"

    # Backward-compatible schema check for check_derived_staleness.py
    spark_data = json.loads(written[0].read_text())
    assert "generated_at" in spark_data
    assert spark_data["generator"] == "test-suite"
    assert spark_data["generation"] == "gen-001"
    assert "source" in spark_data
    assert spark_data["source"]["slug"] == "spark"
    assert spark_data["source"]["chunk_count"] == 1
    assert spark_data["sources"] == {}  # empty for this test

    claude_data = json.loads(written[1].read_text())
    assert claude_data["source"]["slug"] == "claude"
    assert claude_data["source"]["chunk_count"] == 2
    assert claude_data["sources"] == {}
