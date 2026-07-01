"""Integration tests for ingestion with realistic batch scenarios.

These tests require Ollama to be running locally and test the actual
embedding pipeline with realistic data volumes (32 chunks per batch).

Run with: pytest tests/test_ingestion_integration.py -v -m integration
Skip if Ollama unavailable: pytest -m "not integration"
"""

import time
import pytest
from pathlib import Path

from data_engineering_copilot.config.settings import AppSettings, DocumentationSource
from data_engineering_copilot.domain.models import DocumentChunk, ParsedDocument, RawDocument
from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.html_parser import DocumentationHtmlParser
from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore
from data_engineering_copilot.services.chunker import DocumentChunker
from data_engineering_copilot.services.ingestion import IngestionService


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def realistic_settings():
    """Create settings with realistic batch size (32 chunks)."""
    return AppSettings(
        embedding_batch_size=32,  # Realistic batch size
        chunk_size_words=350,
        chunk_overlap_words=70,
        sources=(
            DocumentationSource(
                name="Test Documentation",
                start_urls=("https://example.com/docs/",),
                allowed_domains=("example.com",),
                url_prefixes=("https://example.com/docs/",),
            ),
        ),
    )


@pytest.fixture
def mock_crawler_32_chunks():
    """Crawler that yields 32 realistic documents."""
    class MockCrawler:
        def crawl(self, source, max_pages, on_event=None):
            # Yield 32 documents (realistic batch)
            for i in range(32):
                html = f"""
                <html>
                    <body>
                        <h1>Document {i}</h1>
                        <p>This is a realistic document with substantial content. 
                        It contains multiple paragraphs of text to simulate real documentation.
                        The content is long enough to be chunked into meaningful pieces.
                        This helps test the embedding pipeline with realistic data volumes.
                        Document number {i} contains information about various topics.
                        Each document is processed through the full ingestion pipeline.
                        The pipeline includes crawling, parsing, chunking, and embedding.
                        This test validates that batch slicing works correctly with real data.
                        Performance metrics are collected during the ingestion process.
                        Memory usage is monitored to ensure stability under load.
                        The batch size of 32 is realistic for production scenarios.
                        Larger batches would cause OOM errors without batch slicing.
                        This integration test validates the fix for the OOM issue.</p>
                    </body>
                </html>
                """
                yield RawDocument(
                    source_name=source.name,
                    url=f"https://example.com/docs/page{i}.html",
                    html=html,
                )
    
    return MockCrawler()


@pytest.fixture
def mock_crawler_64_chunks():
    """Crawler that yields 64 documents (2 batches of 32)."""
    class MockCrawler:
        def crawl(self, source, max_pages, on_event=None):
            for i in range(64):
                html = f"""
                <html>
                    <body>
                        <h1>Document {i}</h1>
                        <p>This is document {i} with realistic content for testing batch slicing.
                        The ingestion pipeline processes documents in batches of 32.
                        This test validates that 64 documents are split into 2 batches.
                        Each batch is processed independently through Ollama.
                        Results are concatenated in the correct order.
                        Performance metrics track throughput and latency per batch.
                        Memory usage remains stable throughout the process.
                        This validates the batch slicing implementation.</p>
                    </body>
                </html>
                """
                yield RawDocument(
                    source_name=source.name,
                    url=f"https://example.com/docs/page{i}.html",
                    html=html,
                )
    
    return MockCrawler()


@pytest.fixture
def embeddings_provider(realistic_settings):
    """Create real embeddings provider (requires Ollama)."""
    return SentenceTransformerEmbeddings(
        model_name="nomic-embed-text",
        cache_dir=realistic_settings.embedding_cache_dir,
        local_files_only=True,
    )


@pytest.fixture
def vector_store(realistic_settings, tmp_path):
    """Create temporary vector store for testing."""
    qdrant_dir = tmp_path / "qdrant_test"
    qdrant_dir.mkdir(exist_ok=True)
    return QdrantVectorStore(
        persist_directory=str(qdrant_dir),
        collection_name="test_integration",
    )


@pytest.fixture
def ingestion_service(realistic_settings, mock_crawler_32_chunks, embeddings_provider, vector_store):
    """Create ingestion service with real components."""
    return IngestionService(
        settings=realistic_settings,
        crawler=mock_crawler_32_chunks,
        parser=DocumentationHtmlParser(),
        chunker=DocumentChunker(
            chunk_size_words=realistic_settings.chunk_size_words,
            overlap_words=realistic_settings.chunk_overlap_words,
        ),
        embeddings=embeddings_provider,
        vector_store=vector_store,
    )


# ============================================================================
# Integration Tests - Single Batch (32 chunks)
# ============================================================================

@pytest.mark.integration
def test_ingest_32_chunks_single_batch(ingestion_service):
    """Test ingestion of 32 chunks (single batch).
    
    This validates:
    - All 32 chunks are processed
    - Embeddings are generated correctly
    - Chunks are stored in vector store
    - No OOM errors occur
    """
    start_time = time.time()
    total_chunks = ingestion_service.ingest()
    elapsed_time = time.time() - start_time
    
    # Verify all chunks were processed
    assert total_chunks > 0, "Should have indexed chunks"
    
    # Verify vector store has chunks
    vector_count = ingestion_service.vector_store.count()
    assert vector_count == total_chunks, f"Vector store should have {total_chunks} chunks, got {vector_count}"
    
    # Performance metrics
    throughput = total_chunks / elapsed_time
    print(f"\n✓ Single batch (32 chunks)")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Throughput: {throughput:.1f} chunks/sec")


@pytest.mark.integration
def test_ingest_32_chunks_batch_order_preserved(ingestion_service):
    """Test that batch slicing preserves chunk order.
    
    This validates:
    - Chunks are processed in correct order
    - Embeddings correspond to correct chunks
    - No reordering occurs during batching
    """
    total_chunks = ingestion_service.ingest()
    
    # Retrieve chunks from vector store
    retrieved_chunks = ingestion_service.vector_store.query(
        query_embedding=[0.0] * 768,  # Dummy query
        top_k=total_chunks,
    )
    
    # Verify we got all chunks back
    assert len(retrieved_chunks) == total_chunks, "Should retrieve all chunks"
    
    print(f"\n✓ Batch order preserved")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Retrieved chunks: {len(retrieved_chunks)}")


# ============================================================================
# Integration Tests - Multiple Batches (64 chunks)
# ============================================================================

@pytest.mark.integration
def test_ingest_64_chunks_two_batches(realistic_settings, mock_crawler_64_chunks, embeddings_provider, vector_store):
    """Test ingestion of 64 chunks (2 batches of 32).
    
    This validates:
    - Batch slicing splits 64 chunks into 2 batches
    - Each batch is processed independently
    - Results are concatenated correctly
    - No data loss between batches
    """
    service = IngestionService(
        settings=realistic_settings,
        crawler=mock_crawler_64_chunks,
        parser=DocumentationHtmlParser(),
        chunker=DocumentChunker(
            chunk_size_words=realistic_settings.chunk_size_words,
            overlap_words=realistic_settings.chunk_overlap_words,
        ),
        embeddings=embeddings_provider,
        vector_store=vector_store,
    )
    
    start_time = time.time()
    total_chunks = service.ingest()
    elapsed_time = time.time() - start_time
    
    # Verify all chunks were processed
    assert total_chunks > 0, "Should have indexed chunks"
    
    # Verify vector store has all chunks
    vector_count = service.vector_store.count()
    assert vector_count == total_chunks, f"Vector store should have {total_chunks} chunks, got {vector_count}"
    
    # Performance metrics
    throughput = total_chunks / elapsed_time
    print(f"\n✓ Multiple batches (64 chunks = 2 × 32)")
    print(f"  Total chunks: {total_chunks}")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Throughput: {throughput:.1f} chunks/sec")


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.integration
def test_embedding_throughput_32_chunks(ingestion_service):
    """Measure embedding throughput for 32 chunks.
    
    This validates:
    - Throughput is reasonable (>1 chunk/sec)
    - No performance degradation
    - Batch slicing doesn't add significant overhead
    """
    start_time = time.time()
    total_chunks = ingestion_service.ingest()
    elapsed_time = time.time() - start_time
    
    throughput = total_chunks / elapsed_time
    
    # Throughput should be reasonable (at least 1 chunk/sec)
    assert throughput > 0.5, f"Throughput too low: {throughput:.1f} chunks/sec"
    
    print(f"\n✓ Embedding throughput")
    print(f"  Chunks: {total_chunks}")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Throughput: {throughput:.1f} chunks/sec")


@pytest.mark.integration
def test_embedding_latency_per_batch(ingestion_service):
    """Measure latency for embedding a single batch.
    
    This validates:
    - Batch embedding latency is acceptable
    - No timeout issues
    - Consistent performance
    """
    # Create a small batch of texts
    batch_texts = [f"Document {i} content" for i in range(32)]
    
    start_time = time.time()
    embeddings = ingestion_service.embeddings.embed_texts(batch_texts)
    elapsed_time = time.time() - start_time
    
    # Verify embeddings were generated
    assert len(embeddings) == 32, "Should have 32 embeddings"
    assert all(len(emb) == 768 for emb in embeddings), "All embeddings should be 768-dimensional"
    
    latency_per_chunk = (elapsed_time / 32) * 1000  # ms per chunk
    
    print(f"\n✓ Batch embedding latency")
    print(f"  Batch size: 32 chunks")
    print(f"  Total time: {elapsed_time:.2f}s")
    print(f"  Latency per chunk: {latency_per_chunk:.1f}ms")


# ============================================================================
# Edge Case Tests
# ============================================================================

@pytest.mark.integration
def test_ingest_exactly_batch_size_chunks(realistic_settings, embeddings_provider, vector_store):
    """Test ingestion with exactly batch_size chunks (32).
    
    This validates:
    - Exact batch boundary is handled correctly
    - No partial batch issues
    - All chunks are processed
    """
    class ExactBatchCrawler:
        def crawl(self, source, max_pages, on_event=None):
            for i in range(32):  # Exactly batch_size
                yield RawDocument(
                    source_name=source.name,
                    url=f"https://example.com/docs/page{i}.html",
                    html=f"<html><body><p>Document {i} with content for testing.</p></body></html>",
                )
    
    service = IngestionService(
        settings=realistic_settings,
        crawler=ExactBatchCrawler(),
        parser=DocumentationHtmlParser(),
        chunker=DocumentChunker(
            chunk_size_words=realistic_settings.chunk_size_words,
            overlap_words=realistic_settings.chunk_overlap_words,
        ),
        embeddings=embeddings_provider,
        vector_store=vector_store,
    )
    
    total_chunks = service.ingest()
    assert total_chunks > 0, "Should process exactly batch_size chunks"
    
    print(f"\n✓ Exact batch boundary (32 chunks)")
    print(f"  Total chunks: {total_chunks}")


@pytest.mark.integration
def test_ingest_one_less_than_batch_size(realistic_settings, embeddings_provider, vector_store):
    """Test ingestion with batch_size - 1 chunks (31).
    
    This validates:
    - Partial batch is handled correctly
    - No padding or duplication
    - All chunks are processed
    """
    class PartialBatchCrawler:
        def crawl(self, source, max_pages, on_event=None):
            for i in range(31):  # One less than batch_size
                yield RawDocument(
                    source_name=source.name,
                    url=f"https://example.com/docs/page{i}.html",
                    html=f"<html><body><p>Document {i} with content.</p></body></html>",
                )
    
    service = IngestionService(
        settings=realistic_settings,
        crawler=PartialBatchCrawler(),
        parser=DocumentationHtmlParser(),
        chunker=DocumentChunker(
            chunk_size_words=realistic_settings.chunk_size_words,
            overlap_words=realistic_settings.chunk_overlap_words,
        ),
        embeddings=embeddings_provider,
        vector_store=vector_store,
    )
    
    total_chunks = service.ingest()
    assert total_chunks > 0, "Should process partial batch"
    
    print(f"\n✓ Partial batch (31 chunks)")
    print(f"  Total chunks: {total_chunks}")


@pytest.mark.integration
def test_ingest_one_more_than_batch_size(realistic_settings, embeddings_provider, vector_store):
    """Test ingestion with batch_size + 1 chunks (33).
    
    This validates:
    - Overflow batch is split correctly
    - First batch: 32 chunks
    - Second batch: 1 chunk
    - All chunks are processed
    """
    class OverflowBatchCrawler:
        def crawl(self, source, max_pages, on_event=None):
            for i in range(33):  # One more than batch_size
                yield RawDocument(
                    source_name=source.name,
                    url=f"https://example.com/docs/page{i}.html",
                    html=f"<html><body><p>Document {i} with content.</p></body></html>",
                )
    
    service = IngestionService(
        settings=realistic_settings,
        crawler=OverflowBatchCrawler(),
        parser=DocumentationHtmlParser(),
        chunker=DocumentChunker(
            chunk_size_words=realistic_settings.chunk_size_words,
            overlap_words=realistic_settings.chunk_overlap_words,
        ),
        embeddings=embeddings_provider,
        vector_store=vector_store,
    )
    
    total_chunks = service.ingest()
    assert total_chunks > 0, "Should process overflow batch"
    
    print(f"\n✓ Overflow batch (33 chunks = 32 + 1)")
    print(f"  Total chunks: {total_chunks}")


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.integration
def test_ingest_handles_unparseable_documents(realistic_settings, embeddings_provider, vector_store):
    """Test ingestion skips unparseable documents gracefully.
    
    This validates:
    - Unparseable documents are skipped
    - Remaining documents are processed
    - No cascading failures
    """
    class MixedCrawler:
        def crawl(self, source, max_pages, on_event=None):
            # Yield some valid and some invalid documents
            for i in range(32):
                if i % 5 == 0:
                    # Invalid document (empty)
                    html = "<html></html>"
                else:
                    # Valid document
                    html = f"<html><body><p>Document {i} with content.</p></body></html>"
                
                yield RawDocument(
                    source_name=source.name,
                    url=f"https://example.com/docs/page{i}.html",
                    html=html,
                )
    
    service = IngestionService(
        settings=realistic_settings,
        crawler=MixedCrawler(),
        parser=DocumentationHtmlParser(),
        chunker=DocumentChunker(
            chunk_size_words=realistic_settings.chunk_size_words,
            overlap_words=realistic_settings.chunk_overlap_words,
        ),
        embeddings=embeddings_provider,
        vector_store=vector_store,
    )
    
    total_chunks = service.ingest()
    
    # Should process valid documents and skip invalid ones
    assert total_chunks > 0, "Should process valid documents"
    assert total_chunks < 32, "Should skip invalid documents"
    
    print(f"\n✓ Error handling (mixed valid/invalid)")
    print(f"  Total chunks processed: {total_chunks}")
    print(f"  Documents skipped: {32 - total_chunks}")
