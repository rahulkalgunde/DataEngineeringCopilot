"""Golden evaluation suite — 10 Q&A pairs with mocked embedder.

Marked @pytest.mark.evaluation — excluded from fast CI, runs nightly.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.services.rag_evaluation import RAGEvaluator

pytestmark = pytest.mark.evaluation

GOLDEN_QA: list[dict] = [
    {
        "query": "What is Apache Spark?",
        "relevant_chunks": [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="g1",
                    source_name="spark",
                    title="Spark Overview",
                    url="http://x",
                    text="Apache Spark is a unified analytics engine for large-scale data processing.",
                ),
                distance=0.1,
                confidence=0.9,
            ),
        ],
        "relevant_ids": {"g1"},
    },
    {
        "query": "How does Delta Lake provide ACID transactions?",
        "relevant_chunks": [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="g2",
                    source_name="delta",
                    title="Delta Lake",
                    url="http://x",
                    text="Delta Lake provides ACID transactions on data lakes via a transaction log.",
                ),
                distance=0.15,
                confidence=0.85,
            ),
        ],
        "relevant_ids": {"g2"},
    },
    {
        "query": "What is the difference between DataFrame and Dataset?",
        "relevant_chunks": [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="g3",
                    source_name="spark",
                    title="Spark Types",
                    url="http://x",
                    text="DataFrame is a distributed collection of data organized into named columns. Dataset is a type-safe version of DataFrame.",
                ),
                distance=0.2,
                confidence=0.8,
            ),
        ],
        "relevant_ids": {"g3"},
    },
    {
        "query": "How to read Parquet files in Spark?",
        "relevant_chunks": [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="g4",
                    source_name="spark",
                    title="Spark SQL",
                    url="http://x",
                    text="Use spark.read.parquet('path') to read Parquet files into a DataFrame.",
                ),
                distance=0.1,
                confidence=0.9,
            ),
        ],
        "relevant_ids": {"g4"},
    },
    {
        "query": "What is the purpose of the Catalyst optimizer?",
        "relevant_chunks": [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="g5",
                    source_name="spark",
                    title="Catalyst",
                    url="http://x",
                    text="Catalyst is Spark's extensible query optimizer that applies rule-based and cost-based optimization.",
                ),
                distance=0.12,
                confidence=0.88,
            ),
        ],
        "relevant_ids": {"g5"},
    },
    {
        "query": "How does Spark handle partitioning?",
        "relevant_chunks": [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="g6",
                    source_name="spark",
                    title="Partitioning",
                    url="http://x",
                    text="Spark partitions data across clusters for parallel processing. partitionBy() writes data in partition directories.",
                ),
                distance=0.18,
                confidence=0.82,
            ),
        ],
        "relevant_ids": {"g6"},
    },
    {
        "query": "What is a broadcast join in Spark?",
        "relevant_chunks": [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="g7",
                    source_name="spark",
                    title="Joins",
                    url="http://x",
                    text="A broadcast join sends the small table to all executors, avoiding shuffle for large table joins.",
                ),
                distance=0.14,
                confidence=0.86,
            ),
        ],
        "relevant_ids": {"g7"},
    },
    {
        "query": "How to configure Spark SQL shuffle partitions?",
        "relevant_chunks": [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="g8",
                    source_name="spark",
                    title="Config",
                    url="http://x",
                    text="Set spark.sql.shuffle.partitions to control the number of partitions after a shuffle operation. Default is 200.",
                ),
                distance=0.1,
                confidence=0.9,
            ),
        ],
        "relevant_ids": {"g8"},
    },
    {
        "query": "What is streaming in Apache Spark?",
        "relevant_chunks": [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="g9",
                    source_name="spark",
                    title="Streaming",
                    url="http://x",
                    text="Spark Structured Streaming enables continuous incremental processing of real-time data streams.",
                ),
                distance=0.16,
                confidence=0.84,
            ),
        ],
        "relevant_ids": {"g9"},
    },
    {
        "query": "How to cache a DataFrame in Spark?",
        "relevant_chunks": [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="g10",
                    source_name="spark",
                    title="Caching",
                    url="http://x",
                    text="Use df.cache() or df.persist() to store a DataFrame in memory for reuse across operations.",
                ),
                distance=0.11,
                confidence=0.89,
            ),
        ],
        "relevant_ids": {"g10"},
    },
]


class TestGoldenEvaluation:
    @pytest.mark.parametrize("qa", GOLDEN_QA, ids=[q["query"][:30] for q in GOLDEN_QA])
    def test_retrieval_quality(self, qa: dict):
        evaluator = RAGEvaluator()
        result = evaluator.evaluate(
            query=qa["query"],
            answer=f"Answer about {qa['query']}",
            retrieved_chunks=qa["relevant_chunks"],
            relevant_chunk_ids=qa["relevant_ids"],
        )
        assert result.retrieval_recall == 1.0, f"Recall should be 1.0 for query: {qa['query']}"
        assert result.retrieval_precision == 1.0, f"Precision should be 1.0 for query: {qa['query']}"

    def test_overall_score_range(self):
        evaluator = RAGEvaluator()
        for qa in GOLDEN_QA:
            result = evaluator.evaluate(
                query=qa["query"],
                answer=f"Answer about {qa['query']}",
                retrieved_chunks=qa["relevant_chunks"],
                relevant_chunk_ids=qa["relevant_ids"],
            )
            assert 0.0 <= result.overall_score <= 1.0

    def test_golden_count(self):
        assert len(GOLDEN_QA) == 10

    def test_key_term_coverage(self):
        evaluator = RAGEvaluator()
        qa = GOLDEN_QA[0]
        result = evaluator.evaluate(
            query=qa["query"],
            answer=f"Answer about {qa['query']}",
            retrieved_chunks=qa["relevant_chunks"],
            relevant_chunk_ids=qa["relevant_ids"],
        )
        assert result.key_term_coverage > 0.5, f"Key-term coverage should be >50% for query: {qa['query']}"
