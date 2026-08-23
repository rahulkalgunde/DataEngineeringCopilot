"""Behavioral tests for ChunkFilter (ingestion-path text quality gate).

Covers the three sparse-drop reasons (empty / low_word_count,
low_alpha_density, high_repetition), the disabled passthrough, and noise
cleanup on retained chunks — using real DocumentChunk objects.
"""

from __future__ import annotations

from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.services.text_filter import ChunkFilter


def _chunk(text: str, chunk_id: str = "c1") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_name="spark",
        title="T",
        url="https://example.com/doc",
        text=text,
        content_hash="h",
        word_count=0,
    )


GOOD_TEXT = (
    "Spark resolves transformations lazily through a directed acyclic graph "
    "of stages. Each action triggers job scheduling across executors."
)


class TestSparseDrops:
    def test_empty_text_dropped(self):
        assert ChunkFilter().is_sparse("") is True

    def test_whitespace_only_dropped(self):
        assert ChunkFilter().is_sparse("   \n\t  ") is True

    def test_low_word_count_dropped(self):
        assert ChunkFilter().is_sparse("only a few words here") is True

    def test_low_alpha_density_dropped(self):
        filler = " ".join(f"word{i}" for i in range(20))
        noisy = "{[(<>)]}" * 40 + " " + filler
        f = ChunkFilter()
        sparse, metrics = f._is_sparse(noisy)
        assert sparse is True
        assert metrics["reason"] == "low_alpha_density"

    def test_high_repetition_dropped(self):
        line = "the executor recycled cached broadcast blocks today"
        repeated = "\n".join([line] * 10)
        assert ChunkFilter().is_sparse(repeated) is True


class TestRetainedChunks:
    def test_good_text_kept_and_cleaned(self):
        raw = GOOD_TEXT + "\n\n\n" + "org.apache.spark.sql.DataFrame rules the stage."
        result = ChunkFilter().extract([_chunk(raw)])
        assert len(result) == 1
        cleaned = result[0].text
        assert "org.apache.spark" not in cleaned
        assert "{[(" not in cleaned
        assert "\n\n" not in cleaned
        assert result[0].word_count == len(cleaned.split())

    def test_log_lines_stripped(self):
        raw = GOOD_TEXT + "\n23/08/26 12:00:00 INFO BlockManager: Registered block manager"
        result = ChunkFilter().extract([_chunk(raw)])
        assert len(result) == 1
        assert "BlockManager" not in result[0].text

    def test_disabled_filter_passthrough(self):
        chunk = _chunk("{[(<>)]} tiny")
        result = ChunkFilter(enabled=False).extract([chunk])
        assert result == [chunk]

    def test_mixed_batch_drops_only_sparse(self):
        chunks = [_chunk(GOOD_TEXT, "good"), _chunk("", "empty"), _chunk("short", "sparse")]
        result = ChunkFilter().extract(chunks)
        assert [c.chunk_id for c in result] == ["good"]

    def test_chunk_immutability_preserved_via_replace(self):
        original = _chunk(GOOD_TEXT)
        result = ChunkFilter().extract([original])
        assert original.text == GOOD_TEXT
        assert result[0].chunk_id == original.chunk_id


class TestProcessChunk:
    def test_sparse_returns_none(self):
        assert ChunkFilter().process_chunk("") is None

    def test_clean_returns_cleaned_text(self):
        out = ChunkFilter().process_chunk(GOOD_TEXT + " {[(<>)]}")
        assert out is not None
        assert "{" not in out
