"""
Comprehensive tests for SemanticChunker.

Tests cover:
- Initialization and validation
- Semantic clustering algorithm
- Chunk merging with constraints
- Quality validation
- Edge cases and error handling
- Integration tests with realistic documents
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from data_engineering_copilot.domain.models import ParsedDocument
from data_engineering_copilot.services.semantic_chunker import SemanticChunker


class MockEmbeddingModel:
    """Mock embedding model for testing without actual Ollama."""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate deterministic mock embeddings based on text content.
        Semantic similarity: similar texts get similar embeddings.
        """
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text
            # Similar words/patterns will have correlated embeddings
            words = text.lower().split()
            embedding = [0.0] * self.embedding_dim

            # Hash words into embedding dimensions
            for word in words:
                word_hash = hash(word) % self.embedding_dim
                embedding[word_hash] += 1.0 / len(words)

            # Normalize
            norm = sum(x * x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
            else:
                embedding = [1.0 / (self.embedding_dim**0.5)] * self.embedding_dim

            embeddings.append(embedding)

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self.embed_texts([text])[0]


class TestSemanticChunkerInitialization:
    """Tests for SemanticChunker initialization and validation."""

    def test_init_with_valid_parameters(self):
        """Test initialization with valid parameters."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=250,
            overlap_words=50,
            embedding_model=model,
            min_semantic_similarity=0.5,
            min_chunk_words=20,
        )
        assert chunker.chunk_size_words == 250
        assert chunker.overlap_words == 50
        assert chunker.min_semantic_similarity == 0.5
        assert chunker.min_chunk_words == 20

    def test_init_chunk_size_must_be_positive(self):
        """Test that chunk_size_words must be positive."""
        model = MockEmbeddingModel()
        with pytest.raises(ValueError, match="chunk_size_words must be positive"):
            SemanticChunker(chunk_size_words=0, overlap_words=10, embedding_model=model)

    def test_init_overlap_must_be_valid(self):
        """Test overlap_words validation."""
        model = MockEmbeddingModel()
        with pytest.raises(ValueError, match="overlap_words must be >= 0"):
            SemanticChunker(chunk_size_words=100, overlap_words=-1, embedding_model=model)

        with pytest.raises(ValueError, match="overlap_words must be >= 0"):
            SemanticChunker(chunk_size_words=100, overlap_words=100, embedding_model=model)

    def test_init_similarity_must_be_in_range(self):
        """Test min_semantic_similarity validation."""
        model = MockEmbeddingModel()
        with pytest.raises(ValueError, match="min_semantic_similarity must be between"):
            SemanticChunker(
                chunk_size_words=100,
                overlap_words=10,
                embedding_model=model,
                min_semantic_similarity=-0.1,
            )

        with pytest.raises(ValueError, match="min_semantic_similarity must be between"):
            SemanticChunker(
                chunk_size_words=100,
                overlap_words=10,
                embedding_model=model,
                min_semantic_similarity=1.5,
            )

    def test_init_embedding_model_required(self):
        """Test that embedding_model is required."""
        with pytest.raises(ValueError, match="embedding_model must not be None"):
            SemanticChunker(
                chunk_size_words=100,
                overlap_words=10,
                embedding_model=None,
            )

    def test_init_default_max_chunk_words(self):
        """Test that max_chunk_words defaults to 1.5x chunk_size_words."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=250,
            overlap_words=50,
            embedding_model=model,
        )
        assert chunker.max_chunk_words == int(250 * 1.5)

    def test_init_custom_max_chunk_words(self):
        """Test that max_chunk_words can be customized."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=250,
            overlap_words=50,
            embedding_model=model,
            max_chunk_words=500,
        )
        assert chunker.max_chunk_words == 500


class TestSemanticClustering:
    """Tests for semantic clustering algorithm."""

    async def test_clustering_groups_similar_sentences(self):
        """Test that clustering groups semantically similar sentences."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_semantic_similarity=0.5,
            min_chunk_words=5,
        )

        # Create sentences with clear semantic similarity
        text = (
            "Python is a programming language. Python is widely used. "
            "Java is another language. Java is also popular. "
            "The sky is blue. Birds fly in the sky."
        )
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )

        chunks = await chunker.chunk(document)
        assert len(chunks) > 0
        # Chunks should group related sentences
        assert all(len(c.text) > 0 for c in chunks)

    async def test_chunk_creates_embedding_span_with_telemetry(self):
        calls: list[tuple[str, dict]] = []

        class StubTelemetry:
            def start_observation(self, name, **kwargs):
                calls.append((name, kwargs))
                return _StubSpan()

        class _StubSpan:
            def update(self, **kwargs):
                return self

            def end(self):
                return self

        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_semantic_similarity=0.5,
            min_chunk_words=5,
            telemetry=StubTelemetry(),
        )
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text="Python is a programming language. Python is widely used. Java is another language.",
        )
        chunks = await chunker.chunk(document)
        assert len(chunks) > 0
        assert len(calls) == 1
        name, kwargs = calls[0]
        assert name == "semantic-chunk-embedding"
        assert kwargs["as_type"] == "generation"

    def test_clustering_creates_clusters_from_embeddings(self):
        """Test internal clustering mechanism."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_semantic_similarity=0.3,
        )

        sentences = [
            "Machine learning is AI.",
            "Deep learning is machine learning.",
            "Neural networks use deep learning.",
            "Cats are animals.",
            "Dogs are animals too.",
        ]

        embeddings = model.embed_texts(sentences)
        clusters = chunker._cluster_sentences(sentences, embeddings)

        # Should create clusters
        assert len(clusters) > 0
        # All sentences should be assigned to some cluster
        total_assigned = sum(len(cluster) for cluster in clusters)
        assert total_assigned == len(sentences)

    def test_clustering_respects_similarity_threshold(self):
        """Test that high similarity threshold creates more clusters."""
        model = MockEmbeddingModel()

        # High threshold: stringent clustering
        chunker_strict = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_semantic_similarity=0.9,
        )

        # Low threshold: loose clustering
        chunker_loose = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_semantic_similarity=0.1,
        )

        sentences = [
            "Cat is an animal.",
            "Dog is an animal.",
            "Bird is also an animal.",
        ]

        embeddings = model.embed_texts(sentences)

        clusters_strict = chunker_strict._cluster_sentences(sentences, embeddings)
        clusters_loose = chunker_loose._cluster_sentences(sentences, embeddings)

        # Strict threshold should produce more clusters (less merging)
        assert len(clusters_strict) >= len(clusters_loose)


class TestChunkMerging:
    """Tests for chunk merging with size constraints."""

    async def test_merging_respects_target_size(self):
        """Test that merging respects target chunk size."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=50,
            overlap_words=5,
            embedding_model=model,
            min_semantic_similarity=0.3,
            min_chunk_words=10,
        )

        text = (
            "First topic sentence one. First topic sentence two. "
            "Second topic sentence one. Second topic sentence two. "
            "Third topic sentence one. Third topic sentence two."
        )
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )

        chunks = await chunker.chunk(document)
        for chunk in chunks:
            word_count = len(chunk.text.split())
            # Allow some variance, but generally should be near target
            assert word_count <= chunker.max_chunk_words

    async def test_merging_respects_max_size_hard_limit(self):
        """Test that max_chunk_words is enforced as hard limit."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_semantic_similarity=0.1,  # Very loose to allow large clusters
            min_chunk_words=5,
            max_chunk_words=150,  # Hard limit
        )

        # Create document with few large clusters
        text = "Topic A. " * 30 + "Topic B. " * 30

        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=text,
        )

        chunks = await chunker.chunk(document)
        for chunk in chunks:
            word_count = len(chunk.text.split())
            # All chunks must respect hard limit
            assert word_count <= 150


class TestChunkQualityValidation:
    """Tests for chunk quality validation."""

    def test_is_valid_chunk_with_valid_text(self):
        """Test validation of valid chunks."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_chunk_words=20,
        )

        valid_text = " ".join(["word"] * 25)
        assert chunker._is_valid_chunk(valid_text) is True

    def test_is_valid_chunk_rejects_empty_text(self):
        """Test that empty text is rejected."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_chunk_words=20,
        )

        assert chunker._is_valid_chunk("") is False
        assert chunker._is_valid_chunk("   ") is False

    def test_is_valid_chunk_rejects_too_small(self):
        """Test that chunks smaller than min_chunk_words are rejected."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_chunk_words=20,
        )

        small_text = " ".join(["word"] * 10)
        assert chunker._is_valid_chunk(small_text) is False

    def test_is_valid_chunk_rejects_punctuation_only(self):
        """Test that chunks with no alphanumeric content are rejected."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_chunk_words=20,
        )

        punct_text = "." * 50
        assert chunker._is_valid_chunk(punct_text) is False


class TestChunkIDGeneration:
    """Tests for chunk ID generation."""

    async def test_chunk_id_format_includes_semantic_marker(self):
        """Test that semantic chunk IDs are deterministic UUID v5 values."""
        import uuid

        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_chunk_words=10,
        )

        text = " ".join(["word"] * 50)
        document = ParsedDocument(
            source_name="Test Source",
            title="Test",
            url="https://example.com/test",
            text=text,
        )

        document_chunks = await chunker.chunk(document)
        for chunk in document_chunks:
            parsed = uuid.UUID(chunk.chunk_id)
            assert parsed.version == 5
            # UUID is the sole component of chunk_id, so no separator is present
            assert "semantic" not in chunk.chunk_id


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    async def test_empty_document(self):
        """Test handling of empty document."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_chunk_words=10,
        )

        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text="",
        )

        chunks = await chunker.chunk(document)
        assert chunks == []

    async def test_single_sentence_document(self):
        """Test handling of single sentence."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_chunk_words=3,
        )

        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text="This is a single sentence.",
        )

        chunks = await chunker.chunk(document)
        assert len(chunks) <= 1

    async def test_document_with_very_long_sentences(self):
        """Test handling of documents with long sentences."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=50,
            overlap_words=5,
            embedding_model=model,
            min_chunk_words=10,
            max_chunk_words=100,
        )

        # Create a long sentence
        long_sentence = " ".join([f"word{i}" for i in range(100)]) + "."
        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text=long_sentence,
        )

        chunks = await chunker.chunk(document)
        # Should still respect max_chunk_words (allow small extra for [Source: ...] enrichment)
        for chunk in chunks:
            word_count = len(chunk.text.split())
            assert word_count <= 110

    async def test_embedding_model_failure_raises_chunking_error(self):
        """Embedding failures raise ChunkingError so the URL is marked FAILED
        (retryable) instead of silently vanishing from the index (plan task 4b)."""
        from data_engineering_copilot.domain.exceptions import ChunkingError

        model = MagicMock()
        model.embed_texts.side_effect = Exception("Embedding failed")

        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_chunk_words=10,
        )

        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text="This is a test document.",
        )

        with pytest.raises(ChunkingError, match="embedding failed"):
            await chunker.chunk(document)

    async def test_tokenizer_failure_raises_not_silent_empty(self, monkeypatch):
        """Sentence-tokenizer failure must raise, not silently return [].

        ``SemanticChunker.extract_sentences`` returns ``None`` when ``sent_tokenize``
        raises (e.g. missing ``punkt_tab``). Previously ``chunk()`` conflated that
        ``None`` with an empty ``[]`` and silently dropped the page from the index
        with a misleading "No sentences found" log. It must now surface the failure
        instead.
        """
        import data_engineering_copilot.services.semantic_chunker as sc

        def _boom(text: str) -> list[str]:
            raise LookupError("punkt_tab not downloaded")

        monkeypatch.setattr(sc, "_ensure_punkt_tab", lambda: None)
        monkeypatch.setattr(sc, "sent_tokenize", _boom)

        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_chunk_words=10,
        )

        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/test",
            text="This is a test document.",
        )

        with pytest.raises(RuntimeError, match="Sentence tokenization failed"):
            await chunker.chunk(document)

    async def test_extract_sentences_returns_none_on_tokenizer_failure(self, monkeypatch):
        """Pin the three-valued extract_sentences contract: None on failure, not []."""
        import data_engineering_copilot.services.semantic_chunker as sc

        def _boom(text: str) -> list[str]:
            raise LookupError("punkt_tab not downloaded")

        monkeypatch.setattr(sc, "_ensure_punkt_tab", lambda: None)
        monkeypatch.setattr(sc, "sent_tokenize", _boom)

        assert SemanticChunker.extract_sentences("Some text here.") is None

    async def test_extract_sentences_returns_empty_list_for_empty_text(self, monkeypatch):
        """Empty text yields [] (genuinely no sentences), distinct from None."""
        import data_engineering_copilot.services.semantic_chunker as sc

        monkeypatch.setattr(sc, "_ensure_punkt_tab", lambda: None)
        monkeypatch.setattr(sc, "sent_tokenize", lambda text: [])
        assert SemanticChunker.extract_sentences("") == []


class TestIntegration:
    """Integration tests with realistic documents."""

    async def test_realistic_document(self):
        """Test with realistic multi-topic document."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_semantic_similarity=0.3,
            min_chunk_words=15,
        )

        text = (
            "Python is a programming language created by Guido van Rossum. "
            "It is known for its simplicity and readability. "
            "Python supports multiple programming paradigms. "
            "Java is another popular programming language. "
            "Java runs on the Java Virtual Machine. "
            "Java is used for enterprise applications. "
            "Machine learning is a subset of artificial intelligence. "
            "It enables computers to learn from data. "
            "Neural networks are inspired by the human brain."
        )

        document = ParsedDocument(
            source_name="Programming Topics",
            title="Overview",
            url="https://example.com/programming",
            text=text,
        )

        chunks = await chunker.chunk(document)
        assert len(chunks) > 0
        # Each chunk should have meaningful content
        assert all(len(c.text) > 0 for c in chunks)
        assert all(c.source_name == "Programming Topics" for c in chunks)

    async def test_chunker_reusability(self):
        """Test that chunker can be reused for multiple documents."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_chunk_words=15,
        )

        for i in range(3):
            document = ParsedDocument(
                source_name=f"Source {i}",
                title=f"Document {i}",
                url=f"https://example.com/doc{i}",
                text="This is a test document. " * 20,
            )

            chunks = await chunker.chunk(document)
            assert len(chunks) > 0
            assert all(c.source_name == f"Source {i}" for c in chunks)

    async def test_multiple_documents_independence(self):
        """Test that documents produce independent chunks."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_chunk_words=15,
        )

        docs = [
            ParsedDocument(
                source_name=f"Source {i}",
                title=f"Document {i}",
                url=f"https://example.com/doc{i}",
                text="Topic X sentence. " * 15,
            )
            for i in range(2)
        ]

        all_chunks = []
        for doc in docs:
            all_chunks.extend(await chunker.chunk(doc))

        # All chunks should be unique
        chunk_ids = [c.chunk_id for c in all_chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    async def test_metadata_preservation(self):
        """Test that chunk metadata is preserved."""
        model = MockEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_chunk_words=15,
        )

        document = ParsedDocument(
            source_name="Apache Spark Documentation",
            title="Spark SQL",
            url="https://spark.apache.org/docs/latest/sql.html",
            text="Spark is a big data framework. " * 20,
        )

        chunks = await chunker.chunk(document)
        for chunk in chunks:
            assert chunk.source_name == "Apache Spark Documentation"
            assert chunk.title == "Spark SQL"
            assert chunk.url == document.url
            assert chunk.chunk_index >= 0
            assert chunk.total_chunks == len(chunks)


class RecordingEmbeddingModel:
    """Deterministic embedder that records every batch it is asked to embed."""

    def __init__(self, embedding_dim: int = 8):
        self.embedding_dim = embedding_dim
        self.embedded_batches: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embedded_batches.append(list(texts))
        return [[1.0] + [0.0] * (self.embedding_dim - 1) for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class TestCodeMaskingInSemanticChunker:
    async def test_embedder_receives_masked_tokenization_and_chunks_have_original_code(self):
        """Code spans must never be split across sentences, and output chunks
        must contain the original code (unmasked)."""
        model = RecordingEmbeddingModel()
        chunker = SemanticChunker(
            chunk_size_words=100,
            overlap_words=10,
            embedding_model=model,
            min_semantic_similarity=0.1,
            min_chunk_words=1,
        )

        document = ParsedDocument(
            source_name="Test",
            title="Test",
            url="https://example.com/code",
            text=(
                "Python is used for data. "
                "```python\ndef foo():\n    return 42\n```\n"
                "Spark is a cluster engine. "
                'Use `spark.sql("select 1")` for SQL. '
                "Done here."
            ),
        )

        chunks = await chunker.chunk(document)
        assert chunks, "semantic chunking must not drop a valid document"

        # The embedder receives sentences from masked tokenization: code stays
        # atomic in one sentence (never fragmented by its own punctuation).
        embedded = [s for batch in model.embedded_batches for s in batch]
        assert embedded, "embedder must be invoked"
        code_sentences = [s for s in embedded if "def foo()" in s]
        assert code_sentences, "the code sentence must be embedded"
        assert all("return 42" in s for s in code_sentences), "code block must stay in one sentence"

        # Output chunks must contain the original code (unmasked).
        joined = "".join(c.text for c in chunks)
        assert "def foo():" in joined
        assert "return 42" in joined
        assert "spark.sql" in joined


class _OrthogonalEmbedder:
    """Unit vector per sentence index -> every sentence lands in its own cluster."""

    def __init__(self, fail: bool = False, drop: int = 0) -> None:
        self.fail = fail
        self.drop = drop

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("provider down")
        keep = len(texts) - self.drop
        return [[1.0 if i == j else 0.0 for j in range(len(texts))] for i in range(keep)]


class TestSemanticChunkerOverlapBoundary:
    """Task 4a: overlap must carry the END of the previous chunk."""

    @pytest.mark.asyncio
    async def test_overlap_comes_from_end_of_previous_chunk(self):
        s1 = "Spark caches datasets in memory for reuse across iterative queries."
        s2 = "Airflow schedules workflows using directed acyclic graphs called dags."
        s3 = "Delta Lake brings acid transactions to data lakes at scale."
        document = ParsedDocument(source_name="s", title="t", url="u", text=f"{s1} {s2} {s3}")
        chunker = SemanticChunker(
            chunk_size_words=10,
            overlap_words=5,
            embedding_model=_OrthogonalEmbedder(),
            min_semantic_similarity=0.9,
            min_chunk_words=1,
        )
        chunks = await chunker.chunk(document)
        assert len(chunks) >= 2, f"expected multiple chunks, got {len(chunks)}"
        prev_tail = chunks[0].text.split()[-5:]
        next_head = chunks[1].text.split()[:5]
        assert next_head == prev_tail, f"overlap must repeat previous chunk tail {prev_tail}, got {next_head}"


class TestSemanticChunkerFailureContract:
    """Task 4b: embedding failures must raise, never silently drop the page."""

    @pytest.fixture
    def document(self):
        return ParsedDocument(
            source_name="s", title="t", url="u", text="One sentence here. Another sentence there. A third one."
        )

    @pytest.mark.asyncio
    async def test_embed_failure_raises_chunking_error(self, document):
        from data_engineering_copilot.domain.exceptions import ChunkingError

        chunker = SemanticChunker(chunk_size_words=50, overlap_words=5, embedding_model=_OrthogonalEmbedder(fail=True))
        with pytest.raises(ChunkingError, match="embedding failed"):
            await chunker.chunk(document)

    @pytest.mark.asyncio
    async def test_embedding_count_mismatch_raises_chunking_error(self, document):
        from data_engineering_copilot.domain.exceptions import ChunkingError

        chunker = SemanticChunker(chunk_size_words=50, overlap_words=5, embedding_model=_OrthogonalEmbedder(drop=1))
        with pytest.raises(ChunkingError, match="count mismatch"):
            await chunker.chunk(document)
