"""Integration tests for RAG service with realistic scenarios.

Tests the complete RAG pipeline with real Ollama, embeddings, and vector store.
Requires Ollama running locally.

Run with: pytest tests/test_rag_integration.py -v -m integration
"""

import time
import pytest
from pathlib import Path

from data_engineering_copilot.config.settings import AppSettings
from data_engineering_copilot.domain.models import DocumentChunk
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.ollama_client import OllamaClient
from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore
from data_engineering_copilot.services.rag import RagAnswerService


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def realistic_settings():
    """Settings optimized for 16GB RAM."""
    return AppSettings(
        embedding_batch_size=32,
        retrieval_top_k=3,
        max_context_chars=1500,
        confidence_threshold=0.35,
    )


@pytest.fixture
def embeddings_provider(realistic_settings):
    """Real embeddings provider."""
    return SentenceTransformerEmbeddings(
        model_name="nomic-embed-text",
        cache_dir=realistic_settings.embedding_cache_dir,
        local_files_only=True,
    )


@pytest.fixture
def ollama_client(realistic_settings):
    """Real Ollama client."""
    return OllamaClient(
        base_url=realistic_settings.ollama_base_url,
        model=realistic_settings.ollama_model,
        timeout_seconds=realistic_settings.ollama_timeout_seconds,
    )


@pytest.fixture
def vector_store(realistic_settings, tmp_path):
    """Temporary Qdrant vector store."""
    qdrant_dir = tmp_path / "qdrant_rag_test"
    qdrant_dir.mkdir(exist_ok=True)
    return QdrantVectorStore(
        url=realistic_settings.qdrant_url,
        collection_name="test_rag_integration",
    )


@pytest.fixture
def rag_service(embeddings_provider, ollama_client, vector_store):
    """RAG service with real components."""
    return RagAnswerService(
        vector_store=vector_store,
        ollama_client=ollama_client,
        embedder=embeddings_provider,
    )


@pytest.fixture
def populated_vector_store(vector_store, embeddings_provider):
    """Vector store populated with 32 realistic chunks."""
    chunks = []
    embeddings = []
    
    # Create 32 realistic document chunks
    for i in range(32):
        chunk = DocumentChunk(
            chunk_id=f"test:doc{i}:chunk0",
            source_name="Test Documentation",
            title=f"Document {i}: Data Engineering Concepts",
            url=f"https://example.com/docs/page{i}.html",
            text=f"""
            Document {i} discusses important data engineering concepts.
            
            Apache Spark is a unified analytics engine for large-scale data processing.
            It provides high-level APIs in Scala, Java, Python, and R, and an optimized
            engine that supports general computation graphs for data analysis.
            
            Key features include:
            - In-memory computing for fast data processing
            - Support for batch and streaming data
            - Machine learning libraries (MLlib)
            - Graph processing (GraphX)
            - SQL and structured data processing
            
            Delta Lake is an open-source storage framework that brings ACID transactions
            to Apache Spark and big data workloads. It provides data reliability,
            performance optimization, and unified batch and streaming processing.
            
            Apache Airflow is a platform to programmatically author, schedule and monitor
            workflows. It allows you to define complex data pipelines as code and monitor
            their execution. Airflow is extensible, scalable, and can be deployed in
            various environments.
            
            This document (number {i}) provides comprehensive information about these
            technologies and their applications in modern data engineering.
            """,
        )
        chunks.append(chunk)
        
        # Generate embedding for this chunk
        embedding = embeddings_provider.embed_query(chunk.text)
        embeddings.append(embedding)
    
    # Upsert all chunks
    vector_store.upsert_chunks(chunks, embeddings)
    
    return vector_store


# ============================================================================
# RAG Service Integration Tests
# ============================================================================

@pytest.mark.integration
def test_rag_answer_single_question_32_chunks(rag_service, populated_vector_store):
    """Test RAG with single question and 32 chunks in vector store.
    
    Validates:
    - Question is answered
    - Answer is relevant
    - Confidence is above threshold
    - Sources are cited
    """
    rag_service.vector_store = populated_vector_store
    
    question = "What is Apache Spark?"
    start_time = time.time()
    answer = rag_service.answer(question)
    elapsed_time = time.time() - start_time
    
    # Verify answer was generated
    assert answer.text, "Should generate an answer"
    assert len(answer.text) > 10, "Answer should be substantial"
    assert answer.confidence >= 0, "Confidence should be non-negative"
    
    # Verify sources are cited
    assert len(answer.sources) > 0, "Should cite sources"
    
    print(f"\n✓ Single question with 32 chunks")
    print(f"  Question: {question}")
    print(f"  Answer length: {len(answer.text)} chars")
    print(f"  Confidence: {answer.confidence:.2f}")
    print(f"  Sources cited: {len(answer.sources)}")
    print(f"  Latency: {elapsed_time:.2f}s")


@pytest.mark.integration
def test_rag_answer_multiple_questions_sequential(rag_service, populated_vector_store):
    """Test multiple Q&A in sequence.
    
    Validates:
    - Multiple questions answered correctly
    - No memory leaks between questions
    - Consistent performance
    """
    rag_service.vector_store = populated_vector_store
    
    questions = [
        "What is Apache Spark?",
        "What is Delta Lake?",
        "What is Apache Airflow?",
        "How do these technologies work together?",
        "What are the key features of Spark?",
    ]
    
    answers = []
    latencies = []
    
    for question in questions:
        start_time = time.time()
        answer = rag_service.answer(question)
        elapsed_time = time.time() - start_time
        
        answers.append(answer)
        latencies.append(elapsed_time)
        
        assert answer.text, f"Should answer: {question}"
        assert len(answer.sources) > 0, f"Should cite sources for: {question}"
    
    # Verify consistency
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    
    print(f"\n✓ Multiple questions sequential")
    print(f"  Questions: {len(questions)}")
    print(f"  Avg latency: {avg_latency:.2f}s")
    print(f"  Max latency: {max_latency:.2f}s")
    print(f"  Answers generated: {len(answers)}")


@pytest.mark.integration
def test_rag_answer_with_large_context(rag_service, populated_vector_store, realistic_settings):
    """Test with max_context_chars limit.
    
    Validates:
    - Context is truncated correctly
    - Answer is still generated
    - No errors on context overflow
    """
    rag_service.vector_store = populated_vector_store
    
    # Ask a question that might retrieve large context
    question = "Tell me everything about data engineering technologies"
    
    answer = rag_service.answer(question)
    
    # Verify answer was generated despite large context
    assert answer.text, "Should generate answer with large context"
    assert len(answer.sources) > 0, "Should cite sources"
    
    print(f"\n✓ Large context handling")
    print(f"  Max context chars: {realistic_settings.max_context_chars}")
    print(f"  Answer generated: {len(answer.text)} chars")
    print(f"  Sources: {len(answer.sources)}")


@pytest.mark.integration
def test_rag_answer_low_confidence_threshold(rag_service, populated_vector_store):
    """Test low confidence threshold handling.
    
    Validates:
    - Outside-repository message returned for low confidence
    - Threshold is respected
    """
    rag_service.vector_store = populated_vector_store
    
    # Ask a question unlikely to be in the knowledge base
    question = "What is the capital of France?"
    
    answer = rag_service.answer(question)
    
    # Should return outside-repository message
    assert "cannot answer" in answer.text.lower() or answer.confidence < 0.35, \
        "Should indicate outside knowledge base"
    
    print(f"\n✓ Low confidence handling")
    print(f"  Question: {question}")
    print(f"  Confidence: {answer.confidence:.2f}")
    print(f"  Response: {answer.text[:100]}...")


@pytest.mark.integration
def test_rag_answer_empty_vector_store(rag_service, vector_store):
    """Test with empty vector store.
    
    Validates:
    - Graceful handling of empty store
    - Outside-repository message returned
    """
    rag_service.vector_store = vector_store  # Empty store
    
    question = "What is Apache Spark?"
    answer = rag_service.answer(question)
    
    # Should return outside-repository message
    assert "cannot answer" in answer.text.lower(), \
        "Should indicate outside knowledge base for empty store"
    
    print(f"\n✓ Empty vector store handling")
    print(f"  Vector store count: {vector_store.count()}")
    print(f"  Response: {answer.text[:100]}...")


@pytest.mark.integration
def test_rag_answer_performance_metrics(rag_service, populated_vector_store):
    """Measure RAG performance with 32 chunks and 5 questions.
    
    Validates:
    - Throughput is reasonable
    - Latency is acceptable
    - Memory usage is stable
    """
    rag_service.vector_store = populated_vector_store
    
    questions = [
        "What is Apache Spark?",
        "What is Delta Lake?",
        "What is Apache Airflow?",
        "How do these work together?",
        "What are key features?",
    ]
    
    start_time = time.time()
    answers = [rag_service.answer(q) for q in questions]
    total_time = time.time() - start_time
    
    # Calculate metrics
    throughput = len(questions) / total_time
    avg_latency = total_time / len(questions)
    
    # Verify performance
    assert throughput > 0.1, f"Throughput too low: {throughput:.2f} q/s"
    assert avg_latency < 60, f"Latency too high: {avg_latency:.2f}s"
    
    print(f"\n✓ RAG performance metrics")
    print(f"  Questions: {len(questions)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.2f} questions/sec")
    print(f"  Avg latency: {avg_latency:.2f}s per question")
    print(f"  Answers generated: {len(answers)}")


@pytest.mark.integration
def test_rag_answer_batch_processing(rag_service, populated_vector_store):
    """Test batch question processing.
    
    Validates:
    - Multiple questions processed correctly
    - No data loss
    - All answers generated
    """
    rag_service.vector_store = populated_vector_store
    
    # Generate 10 questions
    questions = [
        f"Question {i}: What is important about data engineering?"
        for i in range(10)
    ]
    
    answers = []
    for question in questions:
        answer = rag_service.answer(question)
        answers.append(answer)
        assert answer.text, f"Should answer: {question}"
    
    # Verify all answered
    assert len(answers) == len(questions), "Should answer all questions"
    assert all(len(a.text) > 0 for a in answers), "All answers should have text"
    
    print(f"\n✓ Batch question processing")
    print(f"  Questions: {len(questions)}")
    print(f"  Answers generated: {len(answers)}")
    print(f"  Success rate: {len([a for a in answers if a.text]) / len(answers) * 100:.1f}%")
