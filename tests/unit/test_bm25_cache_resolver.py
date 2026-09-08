"""Unit tests for the BM25 cache-path resolver used by hybrid search.

Pins ``_resolve_bm25_cache_path`` — the pure helper deciding which persisted
tokenizer file hybrid queries load from. The active-alias branch is only
reachable when the collection name matches the logical alias and the
generation-scoped cache exists; falling back to the literal cache otherwise.
"""

from __future__ import annotations

import data_engineering_copilot.infrastructure.async_qdrant_store as store_mod


def _isolate_resolver(tmp_path, monkeypatch, active_generation: str):
    """Point the resolver at a fake project root with a controllable active generation."""
    fake_root = tmp_path / "project_root"
    fake_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(store_mod, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(store_mod, "resolve_active_generation", lambda: active_generation)
    return fake_root


def test_resolver_prefers_generation_cache_when_alias_active(tmp_path, monkeypatch):
    fake_root = _isolate_resolver(tmp_path, monkeypatch, active_generation="gen-9")
    cache_dir = fake_root / ".bm25_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    gen_cache = cache_dir / "data_engineering_docs__gen-9.json"
    gen_cache.write_text("{}")
    (cache_dir / "data_engineering_docs.json").write_text("{}")

    resolved = store_mod._resolve_bm25_cache_path("data_engineering_docs")

    assert resolved == gen_cache


def test_resolver_falls_back_to_literal_when_generation_cache_missing(tmp_path, monkeypatch):
    fake_root = _isolate_resolver(tmp_path, monkeypatch, active_generation="gen-9")
    literal_cache = fake_root / ".bm25_cache" / "data_engineering_docs.json"
    literal_cache.parent.mkdir(parents=True, exist_ok=True)
    literal_cache.write_text("{}")

    resolved = store_mod._resolve_bm25_cache_path("data_engineering_docs")

    assert resolved == literal_cache


def test_resolver_uses_literal_when_no_active_generation(tmp_path, monkeypatch):
    fake_root = _isolate_resolver(tmp_path, monkeypatch, active_generation="")
    literal_cache = fake_root / ".bm25_cache" / "data_engineering_docs.json"
    literal_cache.parent.mkdir(parents=True, exist_ok=True)
    literal_cache.write_text("{}")

    resolved = store_mod._resolve_bm25_cache_path("data_engineering_docs")

    assert resolved == literal_cache


def test_resolver_returns_literal_even_when_neither_cache_exists(tmp_path, monkeypatch):
    fake_root = _isolate_resolver(tmp_path, monkeypatch, active_generation="gen-9")

    resolved = store_mod._resolve_bm25_cache_path("data_engineering_docs")

    assert resolved == fake_root / ".bm25_cache" / "data_engineering_docs.json"
    assert not resolved.exists()


def test_resolver_scopes_generation_cache_by_collection_name(tmp_path, monkeypatch):
    fake_root = _isolate_resolver(tmp_path, monkeypatch, active_generation="gen-9")
    other_gen = fake_root / ".bm25_cache" / "other_collection__gen-9.json"
    other_gen.parent.mkdir(parents=True, exist_ok=True)
    other_gen.write_text("{}")

    resolved = store_mod._resolve_bm25_cache_path("data_engineering_docs")

    assert resolved == fake_root / ".bm25_cache" / "data_engineering_docs.json"
