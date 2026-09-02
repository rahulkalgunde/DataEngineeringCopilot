"""Task 6: DBSF vs RRF + reranker gate — failing contract."""

from __future__ import annotations


def test_query_supports_dbsf():
    import inspect

    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    assert "dbsf" in inspect.getsource(AsyncQdrantVectorStore.query).lower()


def test_settings_has_retrieval_fusion():
    from data_engineering_copilot.config.settings import AppSettings

    assert "retrieval_fusion" in AppSettings.model_fields
    field = AppSettings.model_fields["retrieval_fusion"]
    # default must be rrf
    assert field.default == "rrf"


def test_retrieval_has_rerank_helper():
    import inspect

    import data_engineering_copilot.evaluation.retrieval as mod

    # rerank evaluation helper must exist (evaluate, rerank, cross-encoder)
    src = inspect.getsource(mod)
    assert "cross" in src.lower() or "rerank" in src.lower()
    # must reference ms-marco-MiniLM or bge or CrossEncoder
    assert "CrossEncoder" in src or "cross-encoder" in src.lower()
