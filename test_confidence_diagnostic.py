#!/usr/bin/env python3
"""
Diagnostic Script: Capture Confidence Scores for RAG Queries
============================================================

Tests the RAG system and logs:
- Number of chunks retrieved
- Confidence scores of each chunk
- Top confidence value
- Whether it passes the threshold
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore


def test_query(question: str):
    """Test a single query and capture confidence scores."""
    print(f"\n{'='*80}")
    print(f"TESTING QUERY: {question}")
    print(f"{'='*80}\n")
    
    # Initialize components
    print("Initializing embedder...")
    embedder = SentenceTransformerEmbeddings(
        model_name=settings.embedding_model_name,
        cache_dir=settings.embedding_cache_dir,
        local_files_only=settings.embedding_local_files_only,
    )
    
    print("Initializing vector store...")
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.collection_name,
    )
    
    # Check vector store status
    try:
        chunk_count = vector_store.count()
        print(f"✅ Vector store is healthy: {chunk_count} chunks indexed\n")
    except Exception as exc:
        print(f"❌ Vector store query failed: {exc}\n")
        return
    
    if chunk_count == 0:
        print("❌ Vector store is EMPTY! No chunks have been indexed.")
        print("   Please run: python main.py ingest --source 'Apache Spark Documentation'\n")
        return
    
    # Embed query
    print(f"Embedding query (dimension: {settings.embedding_dimension})...")
    try:
        query_emb = embedder.embed_query(question)
        print(f"✅ Query embedding successful: {len(query_emb)} dimensions\n")
    except Exception as exc:
        print(f"❌ Embedding failed: {exc}\n")
        return
    
    # Retrieve chunks
    print(f"Retrieving top {settings.retrieval_top_k} chunks (threshold: {settings.confidence_threshold:.2f})...")
    try:
        retrieved_chunks = vector_store.query(query_emb, top_k=settings.retrieval_top_k)
        print(f"✅ Retrieved {len(retrieved_chunks)} chunks\n")
    except Exception as exc:
        print(f"❌ Vector store query failed: {exc}\n")
        return
    
    # Analysis
    if not retrieved_chunks:
        print("❌ NO CHUNKS RETRIEVED")
        print("   The vector store may be empty or query embedding failed.\n")
        return
    
    print(f"{'CHUNK':<8} {'CONFIDENCE':<15} {'DISTANCE':<12} {'SOURCE':<25} {'TITLE':<30}")
    print("-" * 100)
    
    for idx, chunk in enumerate(retrieved_chunks, 1):
        status = "✅ PASS" if chunk.confidence >= settings.confidence_threshold else "❌ FAIL"
        source = chunk.chunk.source_name[:24]
        title = (chunk.chunk.title[:28] if chunk.chunk.title else "N/A")
        print(f"{idx:<8} {chunk.confidence:<15.4f} {chunk.distance:<12.4f} {source:<25} {title:<30} {status}")
    
    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("="*80)
    
    top_confidence = retrieved_chunks[0].confidence
    passed_threshold = top_confidence >= settings.confidence_threshold
    
    print(f"• Top chunk confidence: {top_confidence:.4f}")
    print(f"• Confidence threshold: {settings.confidence_threshold:.2f}")
    print(f"• Threshold check: {'✅ PASSED' if passed_threshold else '❌ FAILED'}")
    
    if not passed_threshold:
        gap = settings.confidence_threshold - top_confidence
        print(f"\n⚠️  GAP: Need {gap:.4f} more confidence to pass threshold")
        new_threshold = round(top_confidence - 0.05, 2)
        print(f"💡 SUGGESTION: Lower threshold to {new_threshold} to allow this query")
    else:
        print(f"\n✅ Confidence check PASSED - should generate answer")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Test with the problematic query
    test_query("What is Apache Spark")
    
    # Test with a few more variations
    test_query("What are the key features of Apache Spark")
    test_query("How does Apache Spark work")
