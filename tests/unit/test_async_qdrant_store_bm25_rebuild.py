"""Unit tests for rebuild_bm25_cache_from_corpus helper."""

from __future__ import annotations

import json

from data_engineering_copilot.domain.models import DocumentChunk


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


def test_rebuild_bm25_cache_is_breadcrumb_id_faithful(tmp_path, monkeypatch):
    """Rebuilds reproduce the build's token→id mapping.

    The builders fit on ``embedding_text_for_chunk`` (breadcrumb-prefixed) text
    in ``chunks.jsonl`` order. A rebuild that fits the raw body instead would
    assign different first-seen vocab ids and silently desync every sparse query
    from the vectors stored in Qdrant. The rebuilt tokenizer must match a direct
    fit over the breadcrumb corpus exactly (vocab order included).
    """
    import data_engineering_copilot.infrastructure.async_qdrant_store as store_mod

    fake_root = tmp_path / "project_root"
    fake_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(store_mod, "PROJECT_ROOT", fake_root)

    from data_engineering_copilot.infrastructure.async_qdrant_store import (
        rebuild_bm25_cache_from_corpus,
    )
    from data_engineering_copilot.infrastructure.bm25_tokenizer import BM25Tokenizer
    from data_engineering_copilot.services.chunker import embedding_text_for_chunk

    records = [
        {
            "chunk_id": "c1",
            "source_name": "spark-sql",
            "title": "Functions",
            "section_header": "col",
            "heading_path": ["API", "functions"],
            "text": "spark.sql.functions col returns a column",
        },
        {
            "chunk_id": "c2",
            "source_name": "spark-sql",
            "title": "Functions",
            "section_header": "lit",
            "heading_path": ["API", "functions"],
            "text": "lit creates a literal column",
        },
        {"chunk_id": "c3", "source_name": "delta", "title": "Overview", "text": "Delta Lake ACID"},
    ]
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    rebuilt = BM25Tokenizer.load(rebuild_bm25_cache_from_corpus("test-faith-001", chunks, namespace=True))

    expected_texts = [
        embedding_text_for_chunk(
            DocumentChunk(
                chunk_id=str(r.get("chunk_id", "")),
                source_name=str(r.get("source_name", "")),
                title=str(r.get("title", "")),
                url=str(r.get("url", "")),
                text=str(r.get("text", "")),
                section_header=str(r.get("section_header", "")),
                heading_path=tuple(r.get("heading_path", ())),
            )
        )
        for r in records
    ]
    direct = BM25Tokenizer(namespace=True)
    direct.fit(expected_texts)

    # Full vocab equality — same tokens, same first-seen ids, same corpus stats.
    assert rebuilt._vocab == direct._vocab
    assert rebuilt.vocab_size == direct.vocab_size
    assert rebuilt._avg_doc_len == direct._avg_doc_len
    assert rebuilt._frozen is True

    # Bonus: fitting on bare raw text would NOT match — proves the rebuild is
    # actually breadcrumb-faithful, not just idempotent.
    raw = BM25Tokenizer(namespace=True)
    raw.fit([r["text"] for r in records])
    assert raw._vocab != rebuilt._vocab


async def test_query_degrades_to_dense_when_tokenizer_missing(tmp_path, monkeypatch):
    """Explicit hybrid/BM25-only modes fail open (dense-only) when the tokenizer
    is missing or unfrozen, instead of raising "BM25 tokenizer is not ready"."""
    import data_engineering_copilot.infrastructure.async_qdrant_store as store_mod
    from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore
    from data_engineering_copilot.services.query_signals import SearchMode

    fake_root = tmp_path / "project_root"
    fake_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(store_mod, "PROJECT_ROOT", fake_root)
    monkeypatch.setattr(store_mod, "AsyncQdrantClient", lambda *a, **k: object())

    from unittest.mock import AsyncMock, MagicMock

    mock_client = AsyncMock()
    mock_hit = MagicMock()
    mock_hit.id = "id-1"
    mock_hit.score = 0.8
    mock_hit.payload = {"chunk_id": "c1", "source_name": "s", "title": "t", "url": "u", "text": "x"}
    mock_response = MagicMock()
    mock_response.points = [mock_hit]
    mock_client.query_points = AsyncMock(return_value=mock_response)

    # Persist path deliberately points at a file that does not exist → a fresh,
    # unfrozen tokenizer is created (exactly the Ask-UI failure scenario).
    store = AsyncQdrantVectorStore(
        url="http://qt",
        collection_name="missing-cache",
        hybrid_search=True,
        bm25_persist_path=fake_root / ".bm25_cache" / "missing-cache.json",
    )
    store._client = mock_client

    for mode in (
        SearchMode.HYBRID_EQUAL,
        SearchMode.HYBRID_SPARSE_BIAS,
        SearchMode.HYBRID_DENSE_BIAS,
        SearchMode.BM25_ONLY,
    ):
        results = await store.query([0.1] * 8, top_k=5, query_text="spark col", search_mode=mode)
        assert len(results) == 1
        assert mock_client.query_points.await_count > 0
        # Degraded to dense: no prefetch, dense "using"
        call_kwargs = mock_client.query_points.call_args.kwargs
        assert "prefetch" not in call_kwargs
        assert call_kwargs.get("using") == "dense"
        mock_client.query_points.reset_mock()
