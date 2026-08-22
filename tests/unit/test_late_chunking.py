"""Late chunking span-pooling embedder (plan phase A).

Unit tests inject a fake encoder exposing ``encode(output_value=...)`` and a
``tokenizer`` with offset mappings so no model download happens.
"""

from __future__ import annotations

import math

import pytest
import pytest_asyncio  # noqa: F401

from data_engineering_copilot.domain.exceptions import EmbeddingError
from data_engineering_copilot.infrastructure.late_chunking import LateChunkEmbedder


class FakeTokenizer:
    """One token per word; offsets cover each word's char range."""

    def __call__(self, text, return_offsets_mapping=False):
        words = text.split()
        offsets, pos = [], 0
        for w in words:
            start = text.index(w, pos)
            offsets.append((start, start + len(w)))
            pos = start + len(w)
        if return_offsets_mapping:
            return {"offset_mapping": offsets}
        return {"input_ids": [[101]] * len(words)}


class FakeEncoder:
    def __init__(self, dim: int = 4):
        self.tokenizer = FakeTokenizer()
        self.dim = dim

    def encode(self, texts, output_value=None):
        assert output_value == "token_embeddings"
        n = len(texts[0].split())
        # token i -> one-hot(i mod dim)
        return [[[1.0 if j == i % self.dim else 0.0 for j in range(self.dim)] for i in range(n)]]


def _normalize(v):
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _make(dim: int = 4) -> LateChunkEmbedder:
    emb = LateChunkEmbedder.__new__(LateChunkEmbedder)
    emb._encoder = FakeEncoder(dim=dim)
    emb.max_tokens = 512
    return emb


class TestSpanPooling:
    @pytest.mark.asyncio
    async def test_spans_pool_their_own_tokens(self):
        emb = _make(dim=4)
        text = "alpha beta gamma delta epsilon"
        spans = [(0, 10), (11, 27)]  # "alpha beta" / "gamma delta epsilon"
        out = await emb.embed_document_spans(text, spans)
        assert len(out) == 2
        expected_first = _normalize([0.5, 0.5, 0.0, 0.0])
        assert all(abs(a - b) < 1e-6 for a, b in zip(out[0], expected_first, strict=True))

    @pytest.mark.asyncio
    async def test_output_vectors_are_unit_norm(self):
        emb = _make(dim=4)
        out = await emb.embed_document_spans("alpha beta gamma", [(0, 17)])
        norm = math.sqrt(sum(x * x for x in out[0]))
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_single_span_equals_mean_of_all_tokens(self):
        emb = _make(dim=4)
        text = "alpha beta gamma"
        out = await emb.embed_document_spans(text, [(0, len(text))])
        expected = _normalize([1 / 3, 1 / 3, 1 / 3, 0.0])
        assert all(abs(a - b) < 1e-6 for a, b in zip(out[0], expected, strict=True))


class TestContextLimit:
    @pytest.mark.asyncio
    async def test_overlong_document_raises(self):
        class LongTok(FakeTokenizer):
            def __call__(self, text, return_offsets_mapping=False):
                if return_offsets_mapping:
                    return {"offset_mapping": [(i, i + 1) for i in range(999)]}
                return {"input_ids": [[101]] * 999}

        emb = _make()
        emb._encoder.tokenizer = LongTok()
        with pytest.raises(EmbeddingError, match="context"):
            await emb.embed_document_spans("long doc text", [(0, 3)])

    @pytest.mark.asyncio
    async def test_special_token_offsets_skipped(self):
        """Offset pairs like (0, 0) mark special tokens and must be excluded."""

        class SpecialTokenizer(FakeTokenizer):
            def __call__(self, text, return_offsets_mapping=False):
                result = super().__call__(text, return_offsets_mapping)
                if return_offsets_mapping:
                    result["offset_mapping"] = [[0, 0]] + result["offset_mapping"]
                else:
                    result["input_ids"] = [[101]] + result["input_ids"]
                return result

        class WithSpecial(FakeEncoder):
            def __init__(self, dim: int = 4):
                super().__init__(dim=dim)
                self.tokenizer = SpecialTokenizer()

            def encode(self, texts, output_value=None):
                base = super().encode(texts, output_value=output_value)
                # aligned zero vector for the (0, 0) special token
                return [[[0.0] * self.dim] + base[0]]

        emb = _make(dim=4)
        emb._encoder = WithSpecial(dim=4)
        out = await emb.embed_document_spans("alpha beta", [(0, 10)])
        expected = _normalize([0.5, 0.5, 0.0, 0.0])
        assert all(abs(a - b) < 1e-6 for a, b in zip(out[0], expected, strict=True))
