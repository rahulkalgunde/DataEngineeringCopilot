"""Unit tests for CLI reset commands (reset-index / reset-qdrant)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from data_engineering_copilot import cli


@pytest.fixture
def bm25_path(tmp_path, monkeypatch):
    path = tmp_path / ".bm25_cache" / "test_collection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cli, "_bm25_cache_path", lambda: path)
    return path


def _patch_settings(monkeypatch, crawl_db_url: str = ""):
    from tests.conftest import make_settings

    monkeypatch.setattr(
        cli,
        "settings",
        make_settings(
            redis_url="redis://localhost:6379/0",
            qdrant_url="http://localhost:6333",
            collection_name="test_collection",
            crawl_db_url=crawl_db_url,
        ),
    )


def test_delete_bm25_cache_removes_file(bm25_path):
    bm25_path.write_text("{}")
    cli._delete_bm25_cache()
    assert not bm25_path.exists()


def test_delete_bm25_cache_missing_is_noop(bm25_path):
    cli._delete_bm25_cache()
    assert not bm25_path.exists()


def test_reset_qdrant_recreates_collection_and_deletes_bm25(monkeypatch, bm25_path):
    bm25_path.write_text("{}")
    recreated = MagicMock()
    monkeypatch.setattr(cli, "_recreate_qdrant_collection", recreated)

    cli.reset_qdrant()

    recreated.assert_called_once()
    assert not bm25_path.exists()


def test_reset_index_is_full_rebuild(monkeypatch, bm25_path):
    _patch_settings(monkeypatch)
    bm25_path.write_text("{}")
    recreated = MagicMock()
    monkeypatch.setattr(cli, "_recreate_qdrant_collection", recreated)

    redis_client = MagicMock()
    redis_client.scan_iter.return_value = []
    import data_engineering_copilot.workers.progress as progress_mod

    monkeypatch.setattr(progress_mod, "get_redis_client", lambda: redis_client)

    cli.reset_index()

    recreated.assert_called_once()
    assert not bm25_path.exists()
    redis_client.scan_iter.assert_called()


def test_reset_index_clears_crawl_redis_keys(monkeypatch, bm25_path):
    _patch_settings(monkeypatch)
    monkeypatch.setattr(cli, "_recreate_qdrant_collection", MagicMock())

    redis_client = MagicMock()
    redis_client.scan_iter.side_effect = lambda pattern: iter(
        ["crawl:url_registry:SourceA"] if pattern == "crawl:url_registry:*" else ["crawl:header:abc"]
    )
    import data_engineering_copilot.workers.progress as progress_mod

    monkeypatch.setattr(progress_mod, "get_redis_client", lambda: redis_client)

    cli.reset_index()

    deleted = redis_client.delete
    deleted.assert_called()
    all_keys = {call.args for call in deleted.call_args_list}
    assert all_keys == {("crawl:url_registry:SourceA",), ("crawl:header:abc",)}


def test_clear_query_cache_deletes_rag_cache_keys(monkeypatch):
    _patch_settings(monkeypatch)

    redis_client = MagicMock()
    redis_client.scan_iter.return_value = [
        "rag:cache:exact:fp:hash",
        "rag:cache:semantic:fp:1",
        "rag:cache:semantic:counter",
    ]
    import data_engineering_copilot.workers.progress as progress_mod

    monkeypatch.setattr(progress_mod, "get_redis_client", lambda: redis_client)

    cli.clear_query_cache()

    redis_client.scan_iter.assert_called_once_with("rag:cache:*")
    redis_client.delete.assert_called_once_with(
        "rag:cache:exact:fp:hash", "rag:cache:semantic:fp:1", "rag:cache:semantic:counter"
    )


def test_clear_query_cache_empty_is_noop(monkeypatch):
    _patch_settings(monkeypatch)

    redis_client = MagicMock()
    redis_client.scan_iter.return_value = []
    import data_engineering_copilot.workers.progress as progress_mod

    monkeypatch.setattr(progress_mod, "get_redis_client", lambda: redis_client)

    cli.clear_query_cache()

    redis_client.scan_iter.assert_called_once_with("rag:cache:*")
    redis_client.delete.assert_not_called()


def test_clear_query_cache_redis_failure_is_graceful(monkeypatch):
    _patch_settings(monkeypatch)

    redis_client = MagicMock()
    redis_client.scan_iter.side_effect = ConnectionError("redis down")
    import data_engineering_copilot.workers.progress as progress_mod

    monkeypatch.setattr(progress_mod, "get_redis_client", lambda: redis_client)

    cli.clear_query_cache()  # must not raise
