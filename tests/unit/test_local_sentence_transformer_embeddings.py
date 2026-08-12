"""Hermetic tests for the local sentence-transformer embedding provider.

The 1.14GB model is never loaded; ``_load_model`` is monkeypatched to a stub
whose ``encode`` returns deterministic vectors. Verifies prefix mapping
(passage/query), the ``EmbedderProtocol`` shape, and the chain ``call`` path.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import EmbeddingRequest
from data_engineering_copilot.infrastructure.local_sentence_transformer_embeddings import (
    LocalSentenceTransformerEmbeddings,
    clear_model_cache,
)


class _StubModel:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts, batch_size=None, show_progress_bar=None, convert_to_numpy=True):
        self.calls.append(list(texts))
        return [[float(i + j) for j in range(self.dim)] for i in range(len(texts))]


def _stub_loader(model_name: str) -> _StubModel:
    return _StubModel()


@pytest.fixture(autouse=True)
def _patch_model(monkeypatch):
    from data_engineering_copilot.infrastructure import local_sentence_transformer_embeddings as mod

    stub = _StubModel()
    monkeypatch.setattr(mod, "_load_model", lambda name: stub)
    clear_model_cache()
    yield stub
    clear_model_cache()


def test_embed_texts_uses_passage_prefix(_patch_model) -> None:
    emb = LocalSentenceTransformerEmbeddings(model_name="stub")
    vectors = pytest_asyncio_run(emb.embed_texts(["chunk a", "chunk b"]))
    assert len(vectors) == 2
    assert len(vectors[0]) == 4
    # passage prefix applied (model-card format with colon)
    assert _patch_model.calls[-1] == ["passage: chunk a", "passage: chunk b"]


def test_embed_query_uses_query_prefix(_patch_model) -> None:
    emb = LocalSentenceTransformerEmbeddings(model_name="stub")
    vec = pytest_asyncio_run(emb.embed_query("what is spark?"))
    assert len(vec) == 4
    assert _patch_model.calls[-1] == ["query: what is spark?"]


def test_call_forwards_embedding_request(_patch_model) -> None:
    emb = LocalSentenceTransformerEmbeddings(model_name="stub")
    vectors = pytest_asyncio_run(emb.call(EmbeddingRequest(input_type="passage", texts=["a", "b"])))
    assert len(vectors) == 2
    assert _patch_model.calls[-1] == ["passage: a", "passage: b"]

    pytest_asyncio_run(emb.call(EmbeddingRequest(input_type="query", texts=["q"])))
    assert _patch_model.calls[-1] == ["query: q"]


def test_call_plain_list_defaults_to_passage(_patch_model) -> None:
    emb = LocalSentenceTransformerEmbeddings(model_name="stub")
    pytest_asyncio_run(emb.call(["legacy chunk"]))
    assert _patch_model.calls[-1] == ["passage: legacy chunk"]


def test_embed_texts_empty_returns_empty() -> None:
    emb = LocalSentenceTransformerEmbeddings(model_name="stub")
    assert pytest_asyncio_run(emb.embed_texts([])) == []


def test_model_and_last_usage_shape() -> None:
    emb = LocalSentenceTransformerEmbeddings(model_name="nvidia/Nemotron-3-Embed-1B-BF16")
    assert emb.model == "nvidia/Nemotron-3-Embed-1B-BF16"
    assert emb.last_usage.prompt_tokens == 0


def pytest_asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
