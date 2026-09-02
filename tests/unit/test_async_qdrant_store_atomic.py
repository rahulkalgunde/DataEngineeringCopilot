"""Task 2: alias-atomic persistence + startup warning (P0)."""

from __future__ import annotations

import json
import logging


def _patch_settings(monkeypatch, tmp_path):
    from tests.conftest import make_settings

    index_state_dir = tmp_path / ".index_state"
    index_state_dir.mkdir(parents=True, exist_ok=True)
    settings = make_settings(
        redis_url="redis://localhost:6379/0",
        qdrant_url="http://localhost:6333",
        collection_name="data_engineering_docs",
        index_state_dir=index_state_dir,
    )
    # Patch cli settings
    import data_engineering_copilot.cli as cli

    monkeypatch.setattr(cli, "settings", settings)
    return settings


def test_gen_activate_copies_bm25_cache_atomically(monkeypatch, tmp_path) -> None:
    """gen-activate must atomically copy gen cache to alias cache if alias missing (tmp+replace)."""
    import data_engineering_copilot.cli as cli
    import data_engineering_copilot.config.settings as settings_mod
    import data_engineering_copilot.infrastructure.async_qdrant_store as store_mod

    _patch_settings(monkeypatch, tmp_path)
    generation = "pinned-abc123def456"

    # Setup fake PROJECT_ROOT/.bm25_cache
    fake_root = tmp_path / "root"
    bm25_dir = fake_root / ".bm25_cache"
    bm25_dir.mkdir(parents=True)

    monkeypatch.setattr(settings_mod, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(store_mod, "PROJECT_ROOT", fake_root)
    # cli imports PROJECT_ROOT from settings_mod
    monkeypatch.setattr("data_engineering_copilot.config.settings.PROJECT_ROOT", fake_root)

    gen_cache = bm25_dir / f"data_engineering_docs__{generation}.json"
    gen_cache.write_text(json.dumps({"vocab": {"hello": 0}, "frozen": True, "corpus_size": 1}))

    alias_cache = bm25_dir / "data_engineering_docs.json"
    assert not alias_cache.exists()

    # Need validation report
    report_path = cli._validation_report_path(generation)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"generation": generation, "passed": True}))
    monkeypatch.setenv("FORCE", "1")
    monkeypatch.setattr(cli, "_qdrant_change_alias", lambda gen: None)

    result = cli.spark_activate(generation)
    assert result == 0

    # Alias must now exist and be byte-identical to gen cache (atomic copy via tmp)
    assert alias_cache.exists(), "alias BM25 cache should have been copied on activate"
    assert alias_cache.read_bytes() == gen_cache.read_bytes()
    # No tmp leftover
    assert not (bm25_dir / "data_engineering_docs.tmp").exists()
    assert not (bm25_dir / "data_engineering_docs.json.tmp").exists()


def test_gen_activate_does_not_overwrite_existing_alias(monkeypatch, tmp_path) -> None:
    import data_engineering_copilot.cli as cli
    import data_engineering_copilot.config.settings as settings_mod
    import data_engineering_copilot.infrastructure.async_qdrant_store as store_mod

    _patch_settings(monkeypatch, tmp_path)
    generation = "pinned-abc123def456"
    fake_root = tmp_path / "root"
    bm25_dir = fake_root / ".bm25_cache"
    bm25_dir.mkdir(parents=True)
    monkeypatch.setattr(settings_mod, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(store_mod, "PROJECT_ROOT", fake_root)

    gen_cache = bm25_dir / f"data_engineering_docs__{generation}.json"
    gen_cache.write_text(json.dumps({"vocab": {"hello": 0}, "frozen": True}))

    alias_cache = bm25_dir / "data_engineering_docs.json"
    alias_cache.write_text(json.dumps({"vocab": {"existing": 0}, "frozen": True}))

    report_path = cli._validation_report_path(generation)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"generation": generation, "passed": True}))
    monkeypatch.setenv("FORCE", "1")
    monkeypatch.setattr(cli, "_qdrant_change_alias", lambda gen: None)

    result = cli.spark_activate(generation)
    assert result == 0
    # Alias must NOT have been overwritten
    assert json.loads(alias_cache.read_text())["vocab"] == {"existing": 0}


def test_startup_warns_when_hybrid_cache_missing(monkeypatch, tmp_path, caplog) -> None:
    """Startup warning if hybrid_search and not _frozen and active generation exists."""
    import data_engineering_copilot.config.settings as settings_mod
    import data_engineering_copilot.infrastructure.async_qdrant_store as store_mod

    fake_root = tmp_path / "root"
    bm25_dir = fake_root / ".bm25_cache"
    bm25_dir.mkdir(parents=True)
    monkeypatch.setattr(settings_mod, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(store_mod, "PROJECT_ROOT", fake_root)

    # No cache file exists
    active_gen = "pinned-abc123def456"
    monkeypatch.setattr(store_mod, "resolve_active_generation", lambda: active_gen)

    # Ensure _resolve_bm25_cache_path returns a missing file
    # For alias "data_engineering_docs", with active generation set but gen cache not existing,
    # it will fall back to alias path which we ensure missing.
    # So warning should trigger.

    with caplog.at_level(logging.WARNING):
        store = store_mod.AsyncQdrantVectorStore(
            url="http://localhost:6333",
            collection_name="data_engineering_docs",
            hybrid_search=True,
        )
        # Need to clean up client
        try:
            import asyncio

            asyncio.run(store.close())
        except Exception:
            pass

    # Must have emitted warning containing expected substring
    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    matched = any("Hybrid enabled but BM25 cache missing" in m and active_gen in m for m in warnings)
    assert matched, f"Expected hybrid missing-cache warning with generation {active_gen!r}, got: {warnings}"
    assert any("gen-bm25-rebuild" in m for m in warnings)


def test_startup_no_warning_when_hybrid_disabled(monkeypatch, tmp_path, caplog) -> None:
    import data_engineering_copilot.config.settings as settings_mod
    import data_engineering_copilot.infrastructure.async_qdrant_store as store_mod

    fake_root = tmp_path / "root"
    bm25_dir = fake_root / ".bm25_cache"
    bm25_dir.mkdir(parents=True)
    monkeypatch.setattr(settings_mod, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(store_mod, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(store_mod, "resolve_active_generation", lambda: "pinned-abc123def456")

    with caplog.at_level(logging.WARNING):
        store = store_mod.AsyncQdrantVectorStore(
            url="http://localhost:6333",
            collection_name="data_engineering_docs",
            hybrid_search=False,
        )
        try:
            import asyncio

            asyncio.run(store.close())
        except Exception:
            pass
    warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("BM25 cache missing" in m for m in warnings)
