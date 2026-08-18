"""Unit tests for the true sentence-preserving chunker."""

from __future__ import annotations

import asyncio
import hashlib

import pytest

from data_engineering_copilot.domain.models import ParsedDocument
from data_engineering_copilot.services.sentence_preserving_chunker import SentencePreservingChunker


def _doc(text: str, url: str = "https://example.com/guide") -> ParsedDocument:
    return ParsedDocument(
        source_name="sent-test",
        title="Sent Doc",
        url=url,
        text=text,
    )


def _chunk(doc: ParsedDocument, chunker: SentencePreservingChunker | None = None) -> list:
    chunker = chunker or SentencePreservingChunker()
    return asyncio.run(chunker.chunk(doc))


class TestSentencePreservingChunker:
    def test_complete_sentences_stay_together(self):
        text = "First sentence. Second sentence here. Third one."
        chunks = _chunk(_doc(text))
        assert len(chunks) == 1
        assert "First sentence." in chunks[0].text
        assert "Second sentence here." in chunks[0].text
        assert "Third one." in chunks[0].text

    def test_sentences_split_across_budget_bounded_chunks(self):
        chunker = SentencePreservingChunker(max_chars=40)
        text = "One sentence. Two sentence. Three sentence. Four sentence. Five sentence."
        chunks = _chunk(_doc(text), chunker)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c.text) <= chunker.max_chars

    def test_long_sentence_splits_losslessly(self):
        long = " ".join(["token"] * 300)
        text = "Short first. " + long + " Short last."
        chunker = SentencePreservingChunker(max_chars=500)
        chunks = _chunk(_doc(text), chunker)
        assert "".join(c.text for c in chunks).strip() == text.strip()

    def test_reconstruction_is_lossless(self):
        text = "Alpha sentence. Beta sentence!\n\nGamma sentence? Delta sentence."
        chunks = _chunk(_doc(text))
        assert "".join(c.text for c in chunks).strip() == text.strip()

    def test_blank_input_returns_no_chunks(self):
        for text in ("", "   ", "\n\n"):
            assert _chunk(_doc(text)) == []

    def test_malformed_text_is_not_discarded(self):
        text = "no punctuation here at all"
        chunks = _chunk(_doc(text))
        assert chunks
        assert "".join(c.text for c in chunks).strip() == text.strip()

    def test_code_blocks_preserved_within_sentence_chunks(self):
        text = "Intro sentence. ```python\ndef foo():\n    return 1\n```\nOutro sentence."
        chunks = _chunk(_doc(text))
        joined = "".join(c.text for c in chunks)
        assert "def foo():" in joined
        assert "return 1" in joined

    def test_code_blocks_stay_atomic_within_one_sentence(self):
        text = (
            "A brief intro here. ```python\ndef compute():\n    x = 1\n    return x\n```\nThen the conclusion follows. "
        )
        chunker = SentencePreservingChunker(max_chars=1000)
        chunks = _chunk(_doc(text), chunker)
        code_chunks = [c for c in chunks if "def compute()" in c.text]
        assert code_chunks, "code block must survive sentence chunking"
        assert all("return x" in c.text for c in code_chunks), "code block must not be split across chunks"

    def test_code_block_reconstruction_is_lossless(self):
        text = 'Intro. ```python\nvalue = spark.sql("select 1")\nprint(value)\n```\nOutro. `inline_code()` here.'
        chunks = _chunk(_doc(text))
        assert "".join(c.text for c in chunks).strip() == text.strip()

    def test_extract_sentences_returns_none_so_no_embeddings(self):
        assert SentencePreservingChunker().extract_sentences("anything") is None

    def test_deterministic_chunk_ids(self):
        doc = _doc("One sentence. Two sentence. Three sentence. Four sentence.")
        first = _chunk(doc, SentencePreservingChunker(max_chars=40))
        second = _chunk(doc, SentencePreservingChunker(max_chars=40))
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert [c.content_hash for c in first] == [c.content_hash for c in second]

    def test_metadata_shape(self):
        doc = _doc("One sentence. Two sentence. Three sentence. Four sentence.")
        chunker = SentencePreservingChunker(max_chars=40)
        chunks = _chunk(doc, chunker)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
            assert c.total_chunks == len(chunks)
            assert c.content_hash == hashlib.sha256(c.text.encode("utf-8")).hexdigest()
            assert c.token_count > 0
            assert c.character_count == len(c.text)
            assert c.source_name == "sent-test"

    def test_precomputed_embeddings_ignored(self):
        chunks = _chunk(_doc("One. Two."), SentencePreservingChunker(max_chars=20))
        assert chunks

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            SentencePreservingChunker(max_tokens=0)
        with pytest.raises(ValueError):
            SentencePreservingChunker(max_chars=0)
