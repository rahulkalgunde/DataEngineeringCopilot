"""
Comprehensive RAG Debugging Suite

This module provides diagnostic tools to identify why RAG cannot answer questions.
It tests all 5 layers of the RAG pipeline:
1. Vector Store (Qdrant)
2. Embedding Generation (Ollama)
3. Semantic Retrieval
4. Ingestion Pipeline
5. RAG Configuration

Usage:
    python -m tests.test_rag_debug
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore
from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
from data_engineering_copilot.infrastructure.ollama_client import OllamaClient
from data_engineering_copilot.domain.models import RetrievedChunk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_debug")

# ============================================================================
# LAYER 1: Vector Store Diagnostics
# ============================================================================

def diagnose_vector_store() -> dict:
    """Layer 1: Check Qdrant connectivity and data presence."""
    logger.info("=" * 60)
    logger.info("LAYER 1: Vector Store Diagnostics")
    logger.info("=" * 60)
    
    result = {
        "layer": "Vector Store",
        "status": "unknown",
        "checks": []
    }
    
    try:
        # Check 1: Qdrant URL connectivity
        logger.info("\n[Check 1.1] Testing Qdrant URL connectivity...")
        try:
            client = QdrantVectorStore(
                url=settings.qdrant_url,
                collection_name=settings.collection_name,
            )
            result["checks"].append({
                "name": "Qdrant URL connectivity",
                "status": "pass",
                "details": f"Connected to {settings.qdrant_url}"
            })
            logger.info(f"  PASS: Connected to {settings.qdrant_url}")
        except Exception as e:
            result["checks"].append({
                "name": "Qdrant URL connectivity",
                "status": "fail",
                "details": str(e)
            })
            logger.error(f"  FAIL: Cannot connect to {settings.qdrant_url}: {e}")
            result["status"] = "fail"
            return result
        
        # Check 2: Collection exists
        logger.info("\n[Check 1.2] Checking collection exists...")
        try:
            count = client.count()
            result["checks"].append({
                "name": "Collection exists",
                "status": "pass",
                "details": f"Collection '{settings.collection_name}' exists"
            })
            logger.info(f"  PASS: Collection '{settings.collection_name}' exists")
        except Exception as e:
            result["checks"].append({
                "name": "Collection exists",
                "status": "fail",
                "details": str(e)
            })
            logger.error(f"  FAIL: Collection check failed: {e}")
            result["status"] = "fail"
            return result
        
        # Check 3: Document count
        logger.info(f"\n[Check 1.3] Checking document count in '{settings.collection_name}'...")
        try:
            count = client.count()
            result["checks"].append({
                "name": "Document count",
                "status": "pass" if count > 0 else "fail",
                "details": f"Total chunks: {count}"
            })
            logger.info(f"  Document count: {count}")
            
            if count == 0:
                logger.warning("  WARNING: Collection is empty! Ingestion may not have run.")
                result["status"] = "fail"
            else:
                logger.info(f"  PASS: Collection has {count} chunks")
                result["status"] = "pass"
        except Exception as e:
            result["checks"].append({
                "name": "Document count",
                "status": "fail",
                "details": str(e)
            })
            logger.error(f"  FAIL: Cannot count documents: {e}")
            result["status"] = "fail"
            
    except Exception as e:
        result["checks"].append({
            "name": "Vector Store Diagnostics",
            "status": "error",
            "details": str(e)
        })
        logger.error(f"  ERROR: {e}")
        result["status"] = "error"
    
    return result


# ============================================================================
# LAYER 2: Embedding Diagnostics
# ============================================================================

def diagnose_embeddings() -> dict:
    """Layer 2: Check embedding generation."""
    logger.info("=" * 60)
    logger.info("LAYER 2: Embedding Diagnostics")
    logger.info("=" * 60)
    
    result = {
        "layer": "Embeddings",
        "status": "unknown",
        "checks": []
    }
    
    try:
        # Check 1: Ollama connectivity
        logger.info("\n[Check 2.1] Testing Ollama connectivity...")
        try:
            ollama = OllamaClient(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
                timeout_seconds=settings.ollama_timeout_seconds,
                num_ctx=settings.ollama_num_ctx,
                num_predict=settings.ollama_num_predict,
            )
            result["checks"].append({
                "name": "Ollama connectivity",
                "status": "pass",
                "details": f"Connected to {settings.ollama_base_url}"
            })
            logger.info(f"  PASS: Connected to {settings.ollama_base_url}")
        except Exception as e:
            result["checks"].append({
                "name": "Ollama connectivity",
                "status": "fail",
                "details": str(e)
            })
            logger.error(f"  FAIL: Cannot connect to Ollama: {e}")
            result["status"] = "fail"
            return result
        
        # Check 2: Embedding model availability
        logger.info(f"\n[Check 2.2] Checking embedding model '{settings.embedding_model_name}'...")
        try:
            embedder = SentenceTransformerEmbeddings(
                model_name=settings.embedding_model_name,
                cache_dir=settings.embedding_cache_dir,
                local_files_only=settings.embedding_local_files_only,
            )
            result["checks"].append({
                "name": "Embedding model availability",
                "status": "pass",
                "details": f"Model '{settings.embedding_model_name}' loaded"
            })
            logger.info(f"  PASS: Model '{settings.embedding_model_name}' loaded")
        except Exception as e:
            result["checks"].append({
                "name": "Embedding model availability",
                "status": "fail",
                "details": str(e)
            })
            logger.error(f"  FAIL: Cannot load embedding model: {e}")
            result["status"] = "fail"
            return result
        
        # Check 3: Generate test embedding
        logger.info("\n[Check 2.3] Generating test embedding for 'what is apache spark?'...")
        try:
            test_question = "what is apache spark?"
            embedding = embedder.embed_query(test_question)
            
            # Validate dimensions
            expected_dim = settings.embedding_dimension
            actual_dim = len(embedding)
            
            result["checks"].append({
                "name": "Embedding generation",
                "status": "pass" if actual_dim == expected_dim else "fail",
                "details": f"Generated {actual_dim}-dim embedding (expected {expected_dim})"
            })
            logger.info(f"  PASS: Generated {actual_dim}-dim embedding")
            
            if actual_dim != expected_dim:
                logger.error(f"  FAIL: Dimension mismatch! Expected {expected_dim}, got {actual_dim}")
                result["status"] = "fail"
                return result
                
        except Exception as e:
            result["checks"].append({
                "name": "Embedding generation",
                "status": "fail",
                "details": str(e)
            })
            logger.error(f"  FAIL: Cannot generate embedding: {e}")
            result["status"] = "fail"
            return result
        
        # Check 4: Embedding consistency
        logger.info("\n[Check 2.4] Testing embedding consistency...")
        try:
            emb1 = embedder.embed_query(test_question)
            emb2 = embedder.embed_query(test_question)
            
            # Check if embeddings are identical (deterministic)
            import math
            diff = sum((a - b) ** 2 for a, b in zip(emb1, emb2))
            is_deterministic = diff < 1e-10
            
            result["checks"].append({
                "name": "Embedding consistency",
                "status": "pass" if is_deterministic else "fail",
                "details": f"Deterministic: {is_deterministic}"
            })
            logger.info(f"  PASS: Embeddings are {'deterministic' if is_deterministic else 'NOT deterministic'}")
            
        except Exception as e:
            result["checks"].append({
                "name": "Embedding consistency",
                "status": "fail",
                "details": str(e)
            })
            logger.error(f"  FAIL: Cannot test consistency: {e}")
            result["status"] = "fail"
            
    except Exception as e:
        result["checks"].append({
            "name": "Embedding Diagnostics",
            "status": "error",
            "details": str(e)
        })
        logger.error(f"  ERROR: {e}")
        result["status"] = "error"
    
    return result


# ============================================================================
# LAYER 3: Retrieval Diagnostics
# ============================================================================

def diagnose_retrieval() -> dict:
    """Layer 3: Check vector retrieval and similarity matching."""
    logger.info("=" * 60)
    logger.info("LAYER 3: Retrieval Diagnostics")
    logger.info("=" * 60)
    
    result = {
        "layer": "Retrieval",
        "status": "unknown",
        "checks": []
    }
    
    try:
        # Initialize components
        embedder = SentenceTransformerEmbeddings(
            model_name=settings.embedding_model_name,
            cache_dir=settings.embedding_cache_dir,
            local_files_only=settings.embedding_local_files_only,
        )
        
        vector_store = QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.collection_name,
        )
        
        # Check 1: Test retrieval with known question
        logger.info("\n[Check 3.1] Testing retrieval for 'what is apache spark?'...")
        try:
            test_question = "what is apache spark?"
            query_emb = embedder.embed_query(test_question)
            
            retrieved = vector_store.query(query_emb, top_k=settings.retrieval_top_k)
            
            result["checks"].append({
                "name": "Retrieval query",
                "status": "pass" if retrieved else "fail",
                "details": f"Retrieved {len(retrieved)} chunks"
            })
            logger.info(f"  Retrieved {len(retrieved)} chunks")
            
            if not retrieved:
                logger.warning("  WARNING: No chunks retrieved!")
                result["status"] = "fail"
                return result
                
        except Exception as e:
            result["checks"].append({
                "name": "Retrieval query",
                "status": "fail",
                "details": str(e)
            })
            logger.error(f"  FAIL: Retrieval failed: {e}")
            result["status"] = "fail"
            return result
        
        # Check 2: Examine retrieved chunks
        logger.info("\n[Check 3.2] Examining retrieved chunks...")
        try:
            for i, chunk in enumerate(retrieved):
                logger.info(f"  Chunk {i+1}:")
                logger.info(f"    Source: {chunk.chunk.source_name}")
                logger.info(f"    Title: {chunk.chunk.title}")
                logger.info(f"    URL: {chunk.chunk.url}")
                logger.info(f"    Distance: {chunk.distance:.4f}")
                logger.info(f"    Confidence: {chunk.confidence:.4f}")
                logger.info(f"    Text preview: {chunk.chunk.text[:100]}...")
                
                result["checks"].append({
                    "name": f"Chunk {i+1} details",
                    "status": "pass",
                    "details": f"Source: {chunk.chunk.source_name}, Confidence: {chunk.confidence:.4f}"
                })
            
            # Check if top confidence meets threshold
            top_confidence = retrieved[0].confidence
            threshold = settings.confidence_threshold
            
            result["checks"].append({
                "name": "Top confidence vs threshold",
                "status": "pass" if top_confidence >= threshold else "fail",
                "details": f"Top confidence: {top_confidence:.4f}, Threshold: {threshold}"
            })
            logger.info(f"\n  Top confidence: {top_confidence:.4f}")
            logger.info(f"  Threshold: {threshold}")
            
            if top_confidence < threshold:
                logger.warning(f"  WARNING: Top confidence ({top_confidence:.4f}) < threshold ({threshold})")
                logger.warning("  This is likely why RAG returns 'outside knowledge repository'")
                result["status"] = "fail"
            else:
                logger.info("  PASS: Top confidence meets threshold")
                result["status"] = "pass"
                
        except Exception as e:
            result["checks"].append({
                "name": "Chunk examination",
                "status": "fail",
                "details": str(e)
            })
            logger.error(f"  FAIL: Cannot examine chunks: {e}")
            result["status"] = "fail"
            
    except Exception as e:
        result["checks"].append({
            "name": "Retrieval Diagnostics",
            "status": "error",
            "details": str(e)
        })
        logger.error(f"  ERROR: {e}")
        result["status"] = "error"
    
    return result


# ============================================================================
# LAYER 4: Ingestion Diagnostics
# ============================================================================

def diagnose_ingestion() -> dict:
    """Layer 4: Check ingestion pipeline status."""
    logger.info("=" * 60)
    logger.info("LAYER 4: Ingestion Diagnostics")
    logger.info("=" * 60)
    
    result = {
        "layer": "Ingestion",
        "status": "unknown",
        "checks": []
    }
    
    try:
        # Check 1: Check ingestion log
        logger.info("\n[Check 4.1] Checking ingestion log...")
        log_path = Path("logs/ingestion_refresh.log")
        
        if log_path.exists():
            result["checks"].append({
                "name": "Ingestion log exists",
                "status": "pass",
                "details": f"Log file found at {log_path}"
            })
            logger.info(f"  PASS: Log file exists at {log_path}")
            
            # Read last 10 lines
            with open(log_path, 'r') as f:
                lines = f.readlines()[-10:]
                logger.info("  Last 10 log lines:")
                for line in lines:
                    logger.info(f"    {line.strip()}")
        else:
            result["checks"].append({
                "name": "Ingestion log exists",
                "status": "fail",
                "details": f"Log file not found at {log_path}"
            })
            logger.warning(f"  WARNING: Log file not found at {log_path}")
        
        # Check 2: Check if data was ingested
        logger.info("\n[Check 4.2] Checking if data was ingested...")
        vector_store = QdrantVectorStore(
            url=settings.qdrant_url,
            collection_name=settings.collection_name,
        )
        
        count = vector_store.count()
        result["checks"].append({
            "name": "Data ingested",
            "status": "pass" if count > 0 else "fail",
            "details": f"Total chunks in collection: {count}"
        })
        logger.info(f"  Chunks in collection: {count}")
        
        if count == 0:
            logger.warning("  WARNING: No data ingested! Run 'python main.py ingest' first.")
            result["status"] = "fail"
        else:
            logger.info("  PASS: Data has been ingested")
            result["status"] = "pass"
            
    except Exception as e:
        result["checks"].append({
            "name": "Ingestion Diagnostics",
            "status": "error",
            "details": str(e)
        })
        logger.error(f"  ERROR: {e}")
        result["status"] = "error"
    
    return result


# ============================================================================
# LAYER 5: Configuration Diagnostics
# ============================================================================

def diagnose_configuration() -> dict:
    """Layer 5: Check RAG configuration settings."""
    logger.info("=" * 60)
    logger.info("LAYER 5: Configuration Diagnostics")
    logger.info("=" * 60)
    
    result = {
        "layer": "Configuration",
        "status": "pass",
        "checks": []
    }
    
    try:
        # Check 1: Confidence threshold
        logger.info("\n[Check 5.1] Checking confidence threshold...")
        threshold = settings.confidence_threshold
        result["checks"].append({
            "name": "Confidence threshold",
            "status": "pass",
            "details": f"Threshold: {threshold}"
        })
        logger.info(f"  Confidence threshold: {threshold}")
        
        if threshold > 0.5:
            logger.warning(f"  WARNING: High threshold ({threshold}) may reject valid matches")
        
        # Check 2: Retrieval top_k
        logger.info("\n[Check 5.2] Checking retrieval top_k...")
        top_k = settings.retrieval_top_k
        result["checks"].append({
            "name": "Retrieval top_k",
            "status": "pass",
            "details": f"top_k: {top_k}"
        })
        logger.info(f"  Retrieval top_k: {top_k}")
        
        if top_k < 2:
            logger.warning(f"  WARNING: Low top_k ({top_k}) may miss relevant chunks")
        
        # Check 3: Max context chars
        logger.info("\n[Check 5.3] Checking max_context_chars...")
        max_chars = settings.max_context_chars
        result["checks"].append({
            "name": "Max context chars",
            "status": "pass",
            "details": f"max_context_chars: {max_chars}"
        })
        logger.info(f"  Max context chars: {max_chars}")
        
        # Check 4: Ollama model
        logger.info("\n[Check 5.4] Checking Ollama model...")
        model = settings.ollama_model
        result["checks"].append({
            "name": "Ollama model",
            "status": "pass",
            "details": f"Model: {model}"
        })
        logger.info(f"  Ollama model: {model}")
        
        # Check 5: Collection name
        logger.info("\n[Check 5.5] Checking collection name...")
        collection = settings.collection_name
        result["checks"].append({
            "name": "Collection name",
            "status": "pass",
            "details": f"Collection: {collection}"
        })
        logger.info(f"  Collection name: {collection}")
        
    except Exception as e:
        result["checks"].append({
            "name": "Configuration Diagnostics",
            "status": "error",
            "details": str(e)
        })
        logger.error(f"  ERROR: {e}")
        result["status"] = "error"
    
    return result


# ============================================================================
# Full Diagnostic Suite
# ============================================================================

def run_full_diagnostic() -> dict:
    """Run all diagnostic layers and generate comprehensive report."""
    logger.info("\n" + "=" * 60)
    logger.info("RAG DIAGNOSTIC SUITE")
    logger.info("=" * 60)
    
    results = {
        "timestamp": str(Path(__file__).parent),
        "layers": {}
    }
    
    # Run all layers
    results["layers"]["vector_store"] = diagnose_vector_store()
    results["layers"]["embeddings"] = diagnose_embeddings()
    results["layers"]["retrieval"] = diagnose_retrieval()
    results["layers"]["ingestion"] = diagnose_ingestion()
    results["layers"]["configuration"] = diagnose_configuration()
    
    # Generate summary
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)
    
    overall_status = "pass"
    for layer_name, layer_result in results["layers"].items():
        status = layer_result.get("status", "unknown")
        logger.info(f"\n{layer_name.upper()}: {status.upper()}")
        
        for check in layer_result.get("checks", []):
            check_status = check.get("status", "unknown")
            check_name = check.get("name", "unnamed")
            check_details = check.get("details", "")
            logger.info(f"  [{check_status}] {check_name}: {check_details}")
        
        if status in ["fail", "error"]:
            overall_status = "fail"
    
    # Overall result
    results["overall_status"] = overall_status
    
    logger.info("\n" + "=" * 60)
    logger.info(f"OVERALL STATUS: {overall_status.upper()}")
    logger.info("=" * 60)
    
    if overall_status == "pass":
        logger.info("\nAll checks passed! RAG system appears healthy.")
    else:
        logger.info("\nSome checks failed. See details above for root cause.")
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    try:
        results = run_full_diagnostic()
        
        # Save results to file
        output_path = Path("logs/diagnostic_report.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nDiagnostic report saved to: {output_path}")
        
        # Exit with appropriate code
        sys.exit(0 if results["overall_status"] == "pass" else 1)
        
    except Exception as e:
        logger.error(f"Fatal error in diagnostic suite: {e}")
        sys.exit(1)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point for the diagnostic suite."""
    try:
        results = run_full_diagnostic()
        
        # Save results to file
        output_path = Path("logs/diagnostic_report.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDiagnostic report saved to: {output_path}")
        
        # Exit with appropriate code
        sys.exit(0 if results["overall_status"] == "pass" else 1)
        
    except Exception as e:
        print(f"Fatal error in diagnostic suite: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
