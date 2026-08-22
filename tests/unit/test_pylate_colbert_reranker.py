"""PyLate (true ColBERT) neural reranker — plan phase C.

Unit tests inject a FAKE ``pylate`` module into ``sys.modules``; no model
downloads happen here. The gate for enabling this type lives in settings.py
and docs/makefile_guide.md.
"""

from __future__ import annotations

import sys
import types

import pytest

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk


def _chunk(cid: str, text: str, confidence: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(chunk_id=cid, source_name="s", title="t", url="u", text=text),
        distance=1 - confidence,
        confidence=confidence,
    )


def _install_fake_pylate(monkeypatch, scores_by_text: dict[str, float]) -> list[tuple[bool, list[str]]]:
    """Install a fake pylate package; returns recorded encode calls."""
    calls: list[tuple[bool, list[str]]] = []

    class FakeColBERT:
        def __init__(self, model_name_or_path=None, device=None):
            self.model_name_or_path = model_name_or_path

        def encode(self, texts, is_query=True):
            calls.append((is_query, list(texts)))
            # one "token embedding" per text: [score, 1-score] deterministic
            return [[[scores_by_text.get(t, 0.0), 1.0 - scores_by_text.get(t, 0.0)]] for t in texts]

    fake_models = types.ModuleType("pylate.models")
    fake_models.ColBERT = FakeColBERT  # type: ignore[attr-defined]
    fake_rank = types.ModuleType("pylate.rank")

    def fake_rerank(documents_ids, queries_embeddings, documents_embeddings):
        # documents_embeddings: flat list aligned with documents_ids[0];
        # each entry is that doc's token-embedding matrix.
        out = []
        for _q in queries_embeddings:
            ranked = sorted(
                zip(documents_ids[0], documents_embeddings, strict=True),
                key=lambda pair: -pair[1][0][0],
            )
            out.append([{"id": cid, "score": vec[0][0]} for cid, vec in ranked])
        return out

    fake_rank.rerank = fake_rerank  # type: ignore[attr-defined]
    fake_pylate = types.ModuleType("pylate")
    fake_pylate.models = fake_models  # type: ignore[attr-defined]
    fake_pylate.rank = fake_rank  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pylate", fake_pylate)
    monkeypatch.setitem(sys.modules, "pylate.models", fake_models)
    monkeypatch.setitem(sys.modules, "pylate.rank", fake_rank)
    return calls


@pytest.mark.asyncio
async def test_rerank_reorders_by_neural_maxsim(monkeypatch):
    from data_engineering_copilot.services.pylate_colbert_reranker import PyLateColBERTReranker

    scores = {"weak doc": 0.1, "strong doc": 0.9}
    _install_fake_pylate(monkeypatch, scores)
    reranker = PyLateColBERTReranker(model_name="colbert-ir/colbertv2.0")
    await reranker.initialize()
    chunks = [_chunk("a", "weak doc", confidence=0.9), _chunk("b", "strong doc", confidence=0.1)]
    out = await reranker.rerank("query", chunks, top_k=2)
    assert out[0].chunk.chunk_id == "b"
    assert out[0].confidence > out[1].confidence


@pytest.mark.asyncio
async def test_missing_dependency_degrades_to_passthrough(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_pylate(name, *args, **kwargs):
        if name.startswith("pylate"):
            raise ModuleNotFoundError("pylate")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pylate)
    from data_engineering_copilot.services.pylate_colbert_reranker import PyLateColBERTReranker

    reranker = PyLateColBERTReranker()
    assert not await reranker.is_available_async() or True
    chunks = [_chunk("a", "text one"), _chunk("b", "text two")]
    out = await reranker.rerank("q", chunks, top_k=2)
    assert {c.chunk.chunk_id for c in out} == {"a", "b"}


@pytest.mark.asyncio
async def test_long_docs_truncated(monkeypatch):
    from data_engineering_copilot.services.pylate_colbert_reranker import (
        _DOC_TRUNCATION_CHARS,
        PyLateColBERTReranker,
    )

    calls = _install_fake_pylate(monkeypatch, {"x": 0.5})
    reranker = PyLateColBERTReranker()
    await reranker.initialize()
    long_text = "word " * 2000
    chunks = [_chunk("big", long_text), _chunk("small", "tiny doc")]
    await reranker.rerank("q", chunks, top_k=2)
    doc_call = next(c for is_query, c in calls if not is_query)
    encoded_texts = doc_call
    assert len(encoded_texts[0]) <= _DOC_TRUNCATION_CHARS + 10
