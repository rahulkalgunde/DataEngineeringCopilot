"""Task 4: RRF k/prefetch tuning grid script contract."""

from __future__ import annotations

from pathlib import Path


def test_tune_rrf_k_grid_exists() -> None:
    assert Path("scripts/tune_rrf_k.py").exists()


def test_tune_settings_have_prefetch_limit() -> None:
    from data_engineering_copilot.config.settings import AppSettings

    assert "retrieval_prefetch_limit" in AppSettings.model_fields


def test_tune_settings_rrf_k_default() -> None:
    from tests.conftest import make_settings

    s = make_settings()
    assert s.hybrid_rrf_k == 20
    assert s.retrieval_prefetch_limit == 100


def test_store_honors_retrieval_prefetch_limit() -> None:
    import inspect

    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

    src = inspect.getsource(AsyncQdrantVectorStore.query)
    assert "retrieval_prefetch_limit" in src
