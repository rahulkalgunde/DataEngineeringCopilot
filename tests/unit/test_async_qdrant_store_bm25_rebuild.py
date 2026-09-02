"""Unit tests for rebuild_bm25_cache_from_corpus helper."""

from __future__ import annotations

import json


def test_rebuild_bm25_cache_from_corpus(tmp_path, monkeypatch):
    import data_engineering_copilot.infrastructure.async_qdrant_store as store_mod

    # Isolate BM25 cache to tmp_path/.bm25_cache so we don't pollute PROJECT_ROOT
    fake_root = tmp_path / "project_root"
    fake_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(store_mod, "PROJECT_ROOT", fake_root)

    # Also isolate settings if needed: ensure namespace handling uses param
    gen = "test-rebuild-001"
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text('{"text":"spark.sql.functions col"}\n{"text":"hello world"}\n')

    from data_engineering_copilot.infrastructure.async_qdrant_store import rebuild_bm25_cache_from_corpus

    out = rebuild_bm25_cache_from_corpus(gen, chunks, namespace=True)
    assert out.exists()
    assert out.stat().st_size > 100
    # second call idempotent (same vocab)
    out2 = rebuild_bm25_cache_from_corpus(gen, chunks, namespace=True)
    assert out.read_text() == out2.read_text()
    # ensure persist path naming
    assert out == fake_root / ".bm25_cache" / "data_engineering_docs__test-rebuild-001.json"
    # ensure vocab correctness
    payload = json.loads(out.read_text())
    assert payload["frozen"] is True
    assert payload["corpus_size"] == 2
