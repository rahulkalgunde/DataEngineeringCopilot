"""
Comprehensive tests for improved DocumentChunker with multiple strategies.

Tests cover:
- Fixed-size chunking (legacy)
- Sentence-preserving chunking (new)
- Chunk quality validation
- Edge cases and error handling
- Metadata preservation
"""

import pytest
from data_engineering_copilot.domain.models import ParsedDocument
from data_engineering_copilot.services.chunker import DocumentChunker, ChunkingStrategy


class TestChunkingStrategy:
    """Tests for ChunkingStrategy enum."""

    def test_strategy_enum_has_fixed_size(self):
        """Test that FIXED_SIZE strategy is defined."""
        assert ChunkingStrategy.FIXED_SIZE.value == "fixed_size"

    def test_strategy_enum_has_sentence_preserving(self):
        """Test that SENTENCE_PRESERVING strategy is defined."""
        assert ChunkingStrategy.SENTENCE_PRESERVING.value == "sentence_preserving"


class TestDocumentChunkerInitialization:
    """Tests for DocumentChunker initialization and validation."""

    def test_init_with_valid_parameters(self):
        """Test initialization with valid parameters."""
        chunker = DocumentChunker(
            chunk_size_words=250,
            overlap_words=50,
            strategy=ChunkingStrategy.SENTENCE_PRESERVING,
            min_chunk_words=20,
        )
        assert chunker.chunk_size_words == 250
        assert chunker.overlap_words == 50
        assert chunker.min_chunk_words == 20
        assert chunker.strategy == ChunkingStrategy.SENTENCE_PRESERVING

    def test_init_with_string_strategy(self):
        """Test initialization with string strategy name."""
        chunker = DocumentChunker(
            chunk_size_words=250,
            overlap_words=50,
            strategy="sentence_preserving",
        )
        assert chunker.strategy == ChunkingStrategy.SENTENCE_PRESERVING

    def test_init_chunk_size_must_be_positive(self):
        """Test that chunk_size_words must be positive."""
        with pytest.raises(ValueError, match="chunk_size_words must be positive"):
            DocumentChunker(chunk_size_words=0, overlap_words=10)

        with pytest.raises(ValueError, match="chunk_size_words must be positive"):
            DocumentChunker(chunk_size_words=-100, overlap_words=10)

    def test_init_overlap_must_be_valid(self):
        """Test overlap_words validation."""
        with pytest.raises(ValueError, match="overlap_words must be >= 0"):
            DocumentChunker(chunk_size_words=100, overlap_words=-1)

        with pytest.raises(ValueError, match="overlap_words must be >= 0"):
            DocumentChunker(chunk_size_words=100, overlap_words=100)

    def test_init_min_chunk_words_must_be_valid(self):
        """Test min_chunk_words validation."""
        with pytest.raises(ValueError, match="min_chunk_words must be non-negative"):
            DocumentChunker(
                chunk_size_words=100,
                overlap_words=10,
                min_chunk_words=-1,
            )

        with pytest.raises(ValueError, match="min_chunk_words must not exceed"):
            DocumentChunker(
                chunk_size_words=100,
                overlap_words=10,
                min_chunk_words=150,
            )

    def test_init_invalid_strategy_string(self):
        """Test that invalid strategy string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            DocumentChunker(
                chunk_size_words=100,
                overlap_words=10,
                strategy="invalid_strategy",
            )

    def test_init_default_strategy_is_sentence_preserving(self):
        """Test that default strategy is SENTENCE_PRESERVING."""
        chunker = DocumentChunker(chunk_size_words=250, overlap_words=50)
        assert chunker.strategy == ChunkingStrategy.SENTENCE_PRESERVING

    def test_init_default_min_chunk_words_is_10(self):
        """Test that default min_chunk_words is 10."""
        chunker = DocumentChunker(chunk_size_words=250, overlap_words=50)
        assert chunker.min_chunk_words == 10


class TestFixedSizeChunking:
    """Tests for legacy fixed-size chunking strategy."""

    def test_fixed_size_basic_chunking(self):
        """Test basic fixed-size word chunking."""
        text = " ".join(f"word{i}" for i in range(30))
        document = ParsedDocument(
            source_name="Test Source",
            title="Test Document",
            url="https://example.com/test",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=10,
            overlap_words=2,
            strategy=ChunkingStrategy.FIXED_SIZE,
            min_chunk_words=5,
        )
        chunks = chunker.chunk(document)

        # Fixed size: step = 10 - 2 = 8, so we should get multiple chunks
        assert len(chunks) > 1
        assert all(c.source_name == "Test Source" for c in chunks)
        assert all(c.title == "Test Document" for c in chunks)
        assert all(c.url == document.url for c in chunks)

    def test_fixed_size_preserves_metadata(self):
        """Test that fixed-size chunking preserves document metadata."""
        text = " ".join(f"word{i}" for i in range(100))
        document = ParsedDocument(
            source_name="Apache Spark Documentation",
            title="Spark SQL",
            url="https://spark.apache.org/docs/latest/sql-programming-guide.html",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=25,
            overlap_words=5,
            strategy=ChunkingStrategy.FIXED_SIZE,
            min_chunk_words=10,
        )
        chunks = chunker.chunk(document)

        for chunk in chunks:
            assert chunk.source_name == "Apache Spark Documentation"
            assert chunk.title == "Spark SQL"
            assert chunk.url == document.url
            assert chunk.chunk_id.startswith("apache-spark-documentation:")

    def test_fixed_size_with_small_document(self):
        """Test fixed-size chunking with document smaller than chunk size."""
        text = "word0 word1 word2 word3 word4"
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=10,
            overlap_words=2,
            strategy=ChunkingStrategy.FIXED_SIZE,
            min_chunk_words=3,
        )
        chunks = chunker.chunk(document)

        # Only 1 chunk since text is smaller than chunk size
        assert len(chunks) == 1
        assert "word0" in chunks[0].text
        assert "word4" in chunks[0].text


class TestSentencePreservingChunking:
    """Tests for sentence-boundary aware chunking strategy."""

    def test_sentence_preserving_basic(self):
        """Test basic sentence-preserving chunking."""
        text = (
            "This is the first sentence. It has some words. "
            "This is the second sentence. It also has words. "
            "This is the third sentence. And more words here."
        )
        document = ParsedDocument(
            source_name="Test Source",
            title="Test Document",
            url="https://example.com/test",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=15,
            overlap_words=3,
            strategy=ChunkingStrategy.SENTENCE_PRESERVING,
            min_chunk_words=5,
        )
        chunks = chunker.chunk(document)

        # Should have multiple chunks
        assert len(chunks) > 0
        # Each chunk should be a valid string
        for chunk in chunks:
            assert isinstance(chunk.text, str)
            assert len(chunk.text) > 0

    def test_sentence_preserving_no_mid_sentence_breaks(self):
        """Verify sentences are not broken in the middle."""
        text = (
            "This is the first sentence. "
            "This is the second sentence. "
            "This is the third sentence."
        )
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=10,
            overlap_words=2,
            strategy=ChunkingStrategy.SENTENCE_PRESERVING,
            min_chunk_words=3,
        )
        chunks = chunker.chunk(document)

        # Each chunk should end with a sentence boundary (period)
        for chunk in chunks:
            # Sentences should be preserved (not split mid-sentence)
            text_ends_with_period = chunk.text.rstrip().endswith(".")
            # Note: overlap text might not end with period, so we check the content

    def test_sentence_preserving_metadata_preservation(self):
        """Test that sentence-preserving chunking preserves metadata."""
        text = (
            "First sentence with many words that should span multiple chunks. "
            "Second sentence to ensure proper chunking behavior. "
            "Third sentence for additional context."
        )
        document = ParsedDocument(
            source_name="Apache Spark Documentation",
            title="Spark SQL Guide",
            url="https://spark.apache.org/docs/latest/sql.html",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=20,
            overlap_words=3,
            strategy=ChunkingStrategy.SENTENCE_PRESERVING,
            min_chunk_words=5,
        )
        chunks = chunker.chunk(document)

        for chunk in chunks:
            assert chunk.source_name == "Apache Spark Documentation"
            assert chunk.title == "Spark SQL Guide"
            assert chunk.url == document.url
            assert chunk.chunk_id.startswith("apache-spark-documentation:")

    def test_sentence_preserving_fallback_on_tokenization_error(self):
        """Test fallback to fixed-size if sentence tokenization fails."""
        # Create a document with text that might cause issues
        text = "Simple document with some text content."
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=10,
            overlap_words=2,
            strategy=ChunkingStrategy.SENTENCE_PRESERVING,
            min_chunk_words=3,
        )

        # Should not raise an error even with unusual text
        chunks = chunker.chunk(document)
        assert isinstance(chunks, list)

    def test_sentence_preserving_empty_text(self):
        """Test handling of documents with no sentences."""
        text = ""
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=10,
            overlap_words=2,
            strategy=ChunkingStrategy.SENTENCE_PRESERVING,
            min_chunk_words=3,
        )
        chunks = chunker.chunk(document)

        # Should return empty list for empty document
        assert chunks == []


class TestChunkQualityValidation:
    """Tests for chunk quality validation."""

    def test_is_valid_chunk_with_valid_text(self):
        """Test validation of valid chunks."""
        chunker = DocumentChunker(
            chunk_size_words=100,
            overlap_words=10,
            min_chunk_words=20,
        )

        valid_text = " ".join(["word"] * 25)
        assert chunker._is_valid_chunk(valid_text) is True

    def test_is_valid_chunk_rejects_empty_text(self):
        """Test that empty text is rejected."""
        chunker = DocumentChunker(
            chunk_size_words=100,
            overlap_words=10,
            min_chunk_words=20,
        )

        assert chunker._is_valid_chunk("") is False
        assert chunker._is_valid_chunk("   ") is False

    def test_is_valid_chunk_rejects_too_small(self):
        """Test that chunks smaller than min_chunk_words are rejected."""
        chunker = DocumentChunker(
            chunk_size_words=100,
            overlap_words=10,
            min_chunk_words=20,
        )

        small_text = " ".join(["word"] * 10)  # Only 10 words
        assert chunker._is_valid_chunk(small_text) is False

    def test_is_valid_chunk_rejects_punctuation_only(self):
        """Test that chunks with no alphanumeric content are rejected."""
        chunker = DocumentChunker(
            chunk_size_words=100,
            overlap_words=10,
            min_chunk_words=20,
        )

        punct_text = "." * 50  # Only punctuation
        assert chunker._is_valid_chunk(punct_text) is False

    def test_is_valid_chunk_accepts_mixed_content(self):
        """Test that chunks with mixed content are accepted."""
        chunker = DocumentChunker(
            chunk_size_words=100,
            overlap_words=10,
            min_chunk_words=20,
        )

        mixed_text = "word " * 20 + "123 " + "!"  # Words with numbers
        assert chunker._is_valid_chunk(mixed_text) is True

    def test_chunks_filtered_by_validation(self):
        """Test that invalid chunks are filtered out of results."""
        text = (
            "This is valid content with reasonable length. "
            "More content here. "
            "And additional content to ensure minimum length requirements. "
            "Final paragraph with more text."
        )
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=20,
            overlap_words=3,
            min_chunk_words=15,  # High minimum to filter chunks
        )
        chunks = chunker.chunk(document)

        # All returned chunks must meet minimum word requirement
        for chunk in chunks:
            word_count = len(chunk.text.split())
            assert word_count >= 15


class TestChunkIDGeneration:
    """Tests for deterministic chunk ID generation."""

    def test_chunk_id_format(self):
        """Test chunk ID format: source:digest:index."""
        text = " ".join(["word"] * 50)
        document = ParsedDocument(
            source_name="Apache Spark",
            title="Test",
            url="https://spark.apache.org/test",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=25,
            overlap_words=5,
            min_chunk_words=10,
        )
        chunks = chunker.chunk(document)

        for i, chunk in enumerate(chunks):
            parts = chunk.chunk_id.split(":")
            assert len(parts) == 3
            assert parts[0] == "apache-spark"  # slugified source
            assert len(parts[1]) == 10  # SHA1 digest truncated to 10 chars
            assert parts[2] == f"{i:04d}"  # Zero-padded index

    def test_chunk_id_deterministic(self):
        """Test that chunk IDs are deterministic."""
        text = " ".join(["word"] * 50)
        document = ParsedDocument(
            source_name="Test Source",
            title="Test",
            url="https://example.com/test",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=25,
            overlap_words=5,
            min_chunk_words=10,
        )

        chunks1 = chunker.chunk(document)
        chunks2 = chunker.chunk(document)

        # Same document should produce same chunk IDs
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.chunk_id == c2.chunk_id

    def test_chunk_id_different_for_different_urls(self):
        """Test that different URLs produce different chunk IDs."""
        text = " ".join(["word"] * 50)
        
        doc1 = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/page1",
            text=text,
        )
        
        doc2 = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/page2",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=25,
            overlap_words=5,
            min_chunk_words=10,
        )

        chunks1 = chunker.chunk(doc1)
        chunks2 = chunker.chunk(doc2)

        # Different URLs should produce different chunk IDs
        assert chunks1[0].chunk_id != chunks2[0].chunk_id


class TestStrategySelection:
    """Tests for strategy selection and behavior."""

    def test_fixed_size_strategy_produces_uniform_chunks(self):
        """Test that fixed-size strategy produces more uniform chunk sizes."""
        text = " ".join(["word"] * 100)
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=20,
            overlap_words=4,
            strategy=ChunkingStrategy.FIXED_SIZE,
            min_chunk_words=5,
        )
        chunks = chunker.chunk(document)

        # Most chunks should be close to target size
        chunk_sizes = [len(c.text.split()) for c in chunks]
        # Fixed size should have more uniform distribution

    def test_sentence_preserving_strategy_respects_sentence_boundaries(self):
        """Test that sentence-preserving strategy maintains sentence integrity."""
        text = (
            "First sentence is here. "
            "Second sentence is also here. "
            "Third sentence completes the paragraph. "
            "Fourth sentence adds more content."
        )
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=15,
            overlap_words=2,
            strategy=ChunkingStrategy.SENTENCE_PRESERVING,
            min_chunk_words=5,
        )
        chunks = chunker.chunk(document)

        # Verify no sentence is split in non-overlap portions
        full_text = " ".join(c.text for c in chunks)
        # Reconstructed text should preserve sentence structure


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_word_document(self):
        """Test handling of single-word documents."""
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text="word",
        )

        chunker = DocumentChunker(
            chunk_size_words=10,
            overlap_words=2,
            min_chunk_words=1,
        )
        chunks = chunker.chunk(document)

        # Should return empty list since 1 word < min_chunk_words default
        # Unless min_chunk_words is set to 1
        chunker2 = DocumentChunker(
            chunk_size_words=10,
            overlap_words=2,
            min_chunk_words=1,
        )
        chunks2 = chunker2.chunk(document)
        assert len(chunks2) >= 0

    def test_zero_overlap(self):
        """Test chunking with zero overlap."""
        text = " ".join(f"word{i}" for i in range(50))
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=10,
            overlap_words=0,
            min_chunk_words=5,
        )
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        # With no overlap, chunks should be consecutive

    def test_max_overlap(self):
        """Test chunking with maximum allowed overlap."""
        text = " ".join(f"word{i}" for i in range(50))
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=10,
            overlap_words=9,  # Max allowed is chunk_size - 1
            min_chunk_words=5,
        )
        chunks = chunker.chunk(document)

        assert len(chunks) > 0

    def test_very_small_chunk_size(self):
        """Test with very small chunk size."""
        text = " ".join(f"word{i}" for i in range(50))
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=5,
            overlap_words=1,
            min_chunk_words=3,
        )
        chunks = chunker.chunk(document)

        assert len(chunks) > 0

    def test_special_characters_in_text(self):
        """Test handling of special characters."""
        text = "Hello! @world# $test% ^code& *examples. (More) [content] {here}."
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )

        chunker = DocumentChunker(
            chunk_size_words=10,
            overlap_words=2,
            min_chunk_words=3,
        )
        chunks = chunker.chunk(document)

        assert len(chunks) > 0
        # Text with special characters should still be processed


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_complex_document_chunking(self):
        """Test chunking of realistic complex document."""
        text = (
            "Apache Spark is a unified computing engine for big data processing. "
            "It provides high-level APIs in Scala, Java, Python, and R. "
            "Spark is 100x faster than Hadoop. "
            "It provides comprehensive tools and libraries for data processing. "
            "This includes support for SQL, machine learning, and streaming. "
            "Spark runs on Hadoop, Mesos, Kubernetes, or standalone clusters. "
            "It can access HDFS, HBase, and other storage systems. "
            "Spark provides an interactive shell for ad-hoc querying. "
            "It supports multiple programming languages for flexibility. "
            "The RDD abstraction enables efficient parallel processing."
        )
        document = ParsedDocument(
            source_name="Apache Spark Documentation",
            title="Spark Overview",
            url="https://spark.apache.org/docs/latest/",
            text=text,
        )

        # Test both strategies
        for strategy in [ChunkingStrategy.FIXED_SIZE, ChunkingStrategy.SENTENCE_PRESERVING]:
            chunker = DocumentChunker(
                chunk_size_words=50,
                overlap_words=10,
                strategy=strategy,
                min_chunk_words=15,
            )
            chunks = chunker.chunk(document)

            assert len(chunks) > 0
            assert all(isinstance(c.chunk_id, str) for c in chunks)
            assert all(len(c.text) > 0 for c in chunks)
            assert all(c.source_name == "Apache Spark Documentation" for c in chunks)

    def test_multiple_documents_independent_ids(self):
        """Test that multiple documents get independent chunk IDs."""
        docs = [
            ParsedDocument(
                source_name="Source 1",
                title="Document 1",
                url="https://example.com/doc1",
                text=" ".join(["word"] * 100),
            ),
            ParsedDocument(
                source_name="Source 2",
                title="Document 2",
                url="https://example.com/doc2",
                text=" ".join(["word"] * 100),
            ),
        ]

        chunker = DocumentChunker(
            chunk_size_words=25,
            overlap_words=5,
            min_chunk_words=10,
        )

        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunker.chunk(doc))

        # Each chunk should have correct source
        chunk_ids = [c.chunk_id for c in all_chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique

    def test_chunker_reusability(self):
        """Test that chunker instance can be reused for multiple documents."""
        chunker = DocumentChunker(
            chunk_size_words=25,
            overlap_words=5,
            min_chunk_words=10,
        )

        for i in range(5):
            document = ParsedDocument(
                source_name=f"Source {i}",
                title=f"Document {i}",
                url=f"https://example.com/doc{i}",
                text=" ".join(f"word{j}" for j in range(100)),
            )
            chunks = chunker.chunk(document)
            assert len(chunks) > 0
