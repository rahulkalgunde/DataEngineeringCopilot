"""Phase 1 tests: index generation identity and BM25 readiness reporting."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from data_engineering_copilot.domain.models import DocumentChunk
from tests.conftest import make_settings

# ------------------------------------------------------------------
# Settings: generation string validation
# ------------------------------------------------------------------


def test_index_generation_default_empty() -> None:
    settings = make_settings()
    assert settings.index_generation == ""
    assert settings.index_require_hybrid is True
    assert settings.index_validation_min_points == 1


def test_index_generation_accepts_valid_generation() -> None:
    settings = make_settings(index_generation="spark-4.0.0-fa33ea00:abc")
    assert settings.index_generation == "spark-4.0.0-fa33ea00:abc"


def test_index_generation_strips_whitespace() -> None:
    settings = make_settings(index_generation="  spark-4.0.0  ")
    assert settings.index_generation == "spark-4.0.0"


def test_index_generation_rejects_whitespace_inside() -> None:
    with pytest.raises(ValidationError):
        make_settings(index_generation="spark 4.0.0")


def test_index_generation_rejects_path_separator() -> None:
    with pytest.raises(ValidationError):
        make_settings(index_generation="spark/4.0.0")


def test_index_validation_min_points_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        make_settings(index_validation_min_points=-1)


# ------------------------------------------------------------------
# BM25 readiness accessors on AsyncQdrantVectorStore
# ------------------------------------------------------------------


@pytest.fixture
def mock_async_qdrant():
    with patch("data_engineering_copilot.infrastructure.async_qdrant_store.AsyncQdrantClient") as mock_cls:
        mock_client = mock_cls.return_value
        yield mock_client


def test_empty_cache_reports_bm25_unavailable(mock_async_qdrant, tmp_path) -> None:
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "nonexistent.json",
    )
    status = store.bm25_status()
    assert status["enabled"] is True
    assert status["fitted"] is False
    assert status["ready"] is False
    assert store.is_hybrid_ready() is False


def test_frozen_cache_reports_bm25_ready(mock_async_qdrant, tmp_path) -> None:
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=True,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    store.fit_bm25(["Apache Spark SQL structured data", "Delta Lake ACID transactions"])
    status = store.bm25_status()
    assert status["fitted"] is True
    assert status["ready"] is True
    assert store.is_hybrid_ready() is True
    assert (tmp_path / "bm25.json").exists()


def test_hybrid_disabled_reports_not_ready(mock_async_qdrant, tmp_path) -> None:
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    store = AsyncQdrantVectorStore(
        url="http://localhost:6333",
        collection_name="test",
        hybrid_search=False,
        bm25_persist_path=tmp_path / "bm25.json",
    )
    store.fit_bm25(["Apache Spark SQL structured data"])
    status = store.bm25_status()
    assert status["enabled"] is False
    assert status["ready"] is False
    assert store.is_hybrid_ready() is False


# ------------------------------------------------------------------
# CLI status helper: _get_bm25_status
# ------------------------------------------------------------------


def test_cli_bm25_status_no_cache(tmp_path, monkeypatch) -> None:
    from data_engineering_copilot import cli
    from data_engineering_copilot.config import settings as settings_module

    monkeypatch.setattr(cli, "_bm25_cache_path", lambda: tmp_path / "missing.json")

    with patch.object(settings_module, "settings") as mock_settings:
        mock_settings.qdrant_url = "http://qdrant:6333"
        mock_settings.collection_name = "data_engineering_docs"
        status = cli._get_bm25_status()
    assert status["cache_exists"] is False
    assert status["hybrid_active"] is False


def test_cli_bm25_status_with_frozen_cache(tmp_path, monkeypatch) -> None:
    from data_engineering_copilot import cli

    cache_path = tmp_path / "bm25.json"
    cache_path.write_text(
        json.dumps(
            {
                "vocab": {"spark": 0},
                "doc_freq": {"spark": 1},
                "corpus_size": 1,
                "avg_doc_len": 5.0,
                "k1": 1.2,
                "b": 0.75,
                "frozen": True,
            }
        )
    )
    monkeypatch.setattr(cli, "_bm25_cache_path", lambda: cache_path)
    monkeypatch.setattr(
        cli,
        "settings",
        type("S", (), {"qdrant_url": "http://qdrant:6333", "collection_name": "data_engineering_docs"})(),
    )

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = mock_urlopen.return_value.__enter__.return_value
        mock_resp.read.return_value = json.dumps(
            {
                "result": {
                    "status": "green",
                    "config": {"params": {"sparse_vectors": {"sparse": {}}, "vectors": {"dense": {"size": 768}}}},
                }
            }
        ).encode()
        status = cli._get_bm25_status()

    assert status["cache_exists"] is True
    assert status["cache_fitted"] is True
    assert status["sparse_configured"] is True
    assert status["hybrid_active"] is True


def test_cli_bm25_status_sparse_without_cache(tmp_path, monkeypatch) -> None:
    from data_engineering_copilot import cli

    monkeypatch.setattr(cli, "_bm25_cache_path", lambda: tmp_path / "missing.json")
    monkeypatch.setattr(
        cli,
        "settings",
        type("S", (), {"qdrant_url": "http://qdrant:6333", "collection_name": "data_engineering_docs"})(),
    )

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = mock_urlopen.return_value.__enter__.return_value
        mock_resp.read.return_value = json.dumps(
            {
                "result": {
                    "status": "green",
                    "config": {"params": {"sparse_vectors": {"sparse": {}}, "vectors": {"dense": {"size": 768}}}},
                }
            }
        ).encode()
        status = cli._get_bm25_status()

    assert status["cache_fitted"] is False
    assert status["sparse_configured"] is True
    assert status["hybrid_active"] is False


# ------------------------------------------------------------------
# BM25 tokenizer version in generation metadata (plan Task 7 Step 3)
# ------------------------------------------------------------------


def test_namespace_bm25_setting_defaults_to_off() -> None:
    settings = make_settings()
    assert settings.namespace_bm25_enabled is False


def test_namespace_bm25_setting_can_be_enabled() -> None:
    settings = make_settings(namespace_bm25_enabled=True)
    assert settings.namespace_bm25_enabled is True


def test_validate_generation_artifacts_rejects_unsupported_version() -> None:
    from data_engineering_copilot.services.spark_index_builder import validate_generation_artifacts

    failures = validate_generation_artifacts(
        generation="g1",
        expected_commit="abc",
        chunks=[_chunk("c1")],
        coverage=[],
        native_manifest_paths=["a.md"],
        bm25_tokenizer_version="namespace-v2",
    )
    assert any("Unsupported BM25 tokenizer version" in f for f in failures)


def test_validate_generation_artifacts_accepts_supported_versions() -> None:
    from data_engineering_copilot.infrastructure.bm25_tokenizer import BM25Tokenizer
    from data_engineering_copilot.services.spark_index_builder import validate_generation_artifacts

    for version in sorted(BM25Tokenizer.SUPPORTED_VERSIONS):
        failures = validate_generation_artifacts(
            generation="g1",
            expected_commit="abc",
            chunks=[_chunk("c1")],
            coverage=[],
            native_manifest_paths=["a.md"],
            bm25_tokenizer_version=version,
        )
        assert not any("version" in f for f in failures)


def test_validate_generation_artifacts_mismatch_between_expected_and_built() -> None:
    from data_engineering_copilot.services.spark_index_builder import validate_generation_artifacts

    failures = validate_generation_artifacts(
        generation="g1",
        expected_commit="abc",
        chunks=[_chunk("c1")],
        coverage=[],
        native_manifest_paths=["a.md"],
        bm25_tokenizer_version="legacy",
        expected_bm25_tokenizer_version="namespace-v1",
    )
    assert any("BM25 tokenizer version mismatch" in f for f in failures)


def _chunk(chunk_id: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_name="s",
        title="t",
        url="http://example.com/1",
        text="body",
        index_generation="g1",
        source_commit="abc",
        doc_type="guide",
    )
