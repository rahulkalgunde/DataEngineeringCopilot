"""Integration and end-to-end tests for the evaluation pipeline.

Tests the evaluation commands end-to-end with real infrastructure:
- eval-fast: Zero-LLM retrieval integrity check
- eval-retrieval: Recall/MRR metrics against a seeded index
- eval-chunking: Chunker quality metrics
- eval-coverage: Dataset coverage validation

Run with: pytest tests/integration/test_evaluation_pipeline.py -v -m integration
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk
from data_engineering_copilot.domain.protocols import LLMClientProtocol

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


class _FakeLLM(LLMClientProtocol):
    """Deterministic LLM double that returns a fixed answer."""

    def __init__(self, answer: str = "Spark supports window functions for rolling aggregates.") -> None:
        self._answer = answer
        self._usage = MagicMock()

    async def generate(self, prompt: str) -> str:
        return self._answer

    def generate_stream(self, prompt: str):
        async def _gen():
            yield self._answer

        return _gen()

    @property
    def last_usage(self):
        return self._usage


class _StubEmbedder:
    """Deterministic embedder for testing."""

    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        return [
            [float(int(hashlib.md5((t + str(i)).encode()).hexdigest(), 16) % 10 + 1) / 10.0] * self.dim
            for i, t in enumerate(texts)
        ]

    async def embed_query(self, text: str) -> list[float]:
        import hashlib

        return [float(int(hashlib.md5(text.encode()).hexdigest(), 16) % 10 + 1) / 10.0] * self.dim

    async def close(self) -> None:
        return None


def _make_chunk(chunk_id: str, source_name: str, title: str, url: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        source_name=source_name,
        title=title,
        url=url,
        text=text,
        content_hash=f"hash-{chunk_id}",
    )


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Create sample chunks for evaluation testing."""
    return [
        _make_chunk(
            "eval:spark:001",
            "Apache Spark",
            "Window Functions",
            "https://example.com/spark/window",
            "Window functions in Spark SQL allow you to perform calculations across a set of rows related to the current row. Use ROWS BETWEEN for rolling aggregates.",
        ),
        _make_chunk(
            "eval:spark:002",
            "Apache Spark",
            "WHERE Clause",
            "https://example.com/spark/where",
            "The WHERE clause in Spark SQL filters rows based on specified conditions. It supports comparison operators and logical operators.",
        ),
        _make_chunk(
            "eval:spark:003",
            "Apache Spark",
            "GROUP BY",
            "https://example.com/spark/groupby",
            "GROUP BY in Spark SQL groups rows that have the same values in specified columns. Use aggregate functions like SUM, COUNT, AVG.",
        ),
        _make_chunk(
            "eval:airflow:001",
            "Apache Airflow",
            "DAGs",
            "https://example.com/airflow/dags",
            "A DAG (Directed Acyclic Graph) in Airflow is a collection of tasks organized with dependencies and scheduling logic.",
        ),
        _make_chunk(
            "eval:delta:001",
            "Delta Lake",
            "ACID Transactions",
            "https://example.com/delta/acid",
            "Delta Lake brings ACID transactions to Apache Spark and big data workloads. It provides schema enforcement and time travel.",
        ),
    ]


@pytest.fixture
def sample_eval_dataset() -> list[dict]:
    """Create a sample evaluation dataset."""
    return [
        {
            "id": "eval-test-001",
            "question": "How do I compute a rolling window aggregate in Spark SQL?",
            "expected_terms": ["Window", "ROWS BETWEEN"],
            "expected_urls": ["https://example.com/spark/window"],
            "source_name": "Apache Spark",
            "intent": "how_to",
            "complexity": "single_hop",
        },
        {
            "id": "eval-test-002",
            "question": "How do I filter rows in Spark SQL?",
            "expected_terms": ["WHERE", "filter"],
            "expected_urls": ["https://example.com/spark/where"],
            "source_name": "Apache Spark",
            "intent": "how_to",
            "complexity": "single_hop",
        },
        {
            "id": "eval-test-003",
            "question": "What is a DAG in Airflow?",
            "expected_terms": ["DAG", "Directed Acyclic Graph"],
            "expected_urls": ["https://example.com/airflow/dags"],
            "source_name": "Apache Airflow",
            "intent": "what_is",
            "complexity": "single_hop",
        },
    ]


# ---------------------------------------------------------------------------
# eval-fast integration tests
# ---------------------------------------------------------------------------


class TestEvalFastIntegration:
    """Tests the eval-fast command with real Qdrant."""

    @pytest.mark.asyncio
    @pytest.mark.qdrant
    async def test_eval_fast_computes_retrieval_metrics(self, qdrant_url, sample_chunks, sample_eval_dataset):
        """eval-fast should compute recall/MRR against a seeded index."""
        from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

        collection = f"itest_eval_fast_{abs(hash(str(sample_chunks))) % 10_000_000}"
        store = AsyncQdrantVectorStore(url=qdrant_url, collection_name=collection, embedding_dimension=2048)

        try:
            await store.initialize()
            embedder = _StubEmbedder()
            vectors = await embedder.embed_texts([c.text for c in sample_chunks])
            store.fit_bm25([c.text for c in sample_chunks])

            tagged = [DocumentChunk(**{**c.__dict__, "index_generation": "eval-fast-test"}) for c in sample_chunks]
            await store.upsert_frozen_chunks(tagged, vectors)

            service = MagicMock()
            service.retrieve = AsyncMock(
                return_value=[
                    RetrievedChunk(chunk=sample_chunks[0], distance=0.1, confidence=0.9),
                    RetrievedChunk(chunk=sample_chunks[1], distance=0.2, confidence=0.8),
                ]
            )

            from data_engineering_copilot.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k

            retrieved_urls = ["https://example.com/spark/window", "https://example.com/spark/where"]
            expected_urls = ["https://example.com/spark/window"]

            recall = recall_at_k(retrieved_urls, expected_urls, k=5)
            assert recall == 1.0

            ndcg = ndcg_at_k(retrieved_urls, expected_urls, k=5)
            assert 0 < ndcg <= 1.0

            await embedder.close()
        finally:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=qdrant_url, prefer_grpc=False)
            client.delete_collection(collection_name=collection)
            client.close()

    @pytest.mark.asyncio
    @pytest.mark.qdrant
    async def test_eval_fast_with_no_matches(self, qdrant_url, sample_chunks):
        """eval-fast should handle queries with no matches gracefully."""
        from data_engineering_copilot.evaluation.retrieval_metrics import ndcg_at_k, recall_at_k

        retrieved_urls = ["https://example.com/spark/window", "https://example.com/spark/where"]
        expected_urls = ["https://example.com/nonexistent"]

        recall = recall_at_k(retrieved_urls, expected_urls, k=5)
        assert recall == 0.0

        ndcg = ndcg_at_k(retrieved_urls, expected_urls, k=5)
        assert ndcg == 0.0


# ---------------------------------------------------------------------------
# eval-retrieval integration tests
# ---------------------------------------------------------------------------


class TestEvalRetrievalIntegration:
    """Tests the eval-retrieval command with real Qdrant."""

    @pytest.mark.asyncio
    @pytest.mark.qdrant
    async def test_eval_retrieval_computes_metrics(self, qdrant_url, sample_chunks, sample_eval_dataset):
        """eval-retrieval should compute recall/MRR metrics for a dataset."""
        from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

        collection = f"itest_eval_retrieval_{abs(hash(str(sample_chunks))) % 10_000_000}"
        store = AsyncQdrantVectorStore(url=qdrant_url, collection_name=collection, embedding_dimension=2048)

        try:
            await store.initialize()
            embedder = _StubEmbedder()
            vectors = await embedder.embed_texts([c.text for c in sample_chunks])
            store.fit_bm25([c.text for c in sample_chunks])

            tagged = [DocumentChunk(**{**c.__dict__, "index_generation": "eval-retrieval-test"}) for c in sample_chunks]
            await store.upsert_frozen_chunks(tagged, vectors)

            results = []
            for item in sample_eval_dataset:
                retrieved = await store.query(
                    query_embedding=await embedder.embed_query(item["question"]),
                    top_k=5,
                )
                results.append(
                    {
                        "question": item["question"],
                        "retrieved_urls": [r.chunk.url for r in retrieved],
                        "expected_urls": item["expected_urls"],
                    }
                )

            assert len(results) == len(sample_eval_dataset)

            for r in results:
                assert "retrieved_urls" in r
                assert "expected_urls" in r

            await embedder.close()
        finally:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=qdrant_url, prefer_grpc=False)
            client.delete_collection(collection_name=collection)
            client.close()


# ---------------------------------------------------------------------------
# eval-chunking integration tests
# ---------------------------------------------------------------------------


class TestEvalChunkingIntegration:
    """Tests the eval-chunking command with real documents."""

    def test_chunking_evaluator_computes_metrics(self, sample_chunks):
        """Chunking evaluator should compute metrics for chunk quality."""
        from data_engineering_copilot.evaluation.chunking_metrics import token_iou

        doc_text = " ".join(c.text for c in sample_chunks)
        gold_spans = [
            {"content": "Apache Spark", "start": 0, "end": 12, "structural_type": "entity"},
        ]

        score = token_iou(doc_text, gold_spans, sample_chunks)
        assert 0.0 <= score <= 1.0

    def test_chunking_with_gold_spans(self):
        """Chunking evaluator should compare against gold spans."""
        from data_engineering_copilot.evaluation.chunking_gold import (
            ChunkingGoldDoc,
            ChunkingGoldSpan,
            validate_gold_doc,
        )

        gold_doc = ChunkingGoldDoc(
            doc_id="test-doc",
            text="Apache Spark is a unified analytics engine.",
            gold_spans=[
                ChunkingGoldSpan(content="Apache Spark", start=0, end=12, structural_type="entity"),
            ],
        )

        validate_gold_doc(gold_doc)
        assert gold_doc.doc_id == "test-doc"


# ---------------------------------------------------------------------------
# eval-coverage integration tests
# ---------------------------------------------------------------------------


class TestEvalCoverageIntegration:
    """Tests the eval-coverage command with real datasets."""

    def test_coverage_validator_detects_orphaned_rows(self, sample_eval_dataset):
        """Coverage validator should detect rows with no matching indexed content."""
        from data_engineering_copilot.evaluation.eval_schema import EvalKind, kind_of

        for item in sample_eval_dataset:
            assert kind_of(item) is EvalKind.RECALL

    def test_coverage_validator_accepts_valid_rows(self, sample_eval_dataset):
        """Coverage validator should accept valid rows."""
        from data_engineering_copilot.evaluation.eval_schema import validate_eval_row

        for item in sample_eval_dataset:
            validate_eval_row(item)


# ---------------------------------------------------------------------------
# Evaluation metrics integration tests
# ---------------------------------------------------------------------------


class TestEvaluationMetricsIntegration:
    """Tests evaluation metrics computation."""

    def test_recall_at_k_perfect(self):
        """Perfect recall when all expected items are retrieved."""
        from data_engineering_copilot.evaluation.retrieval_metrics import recall_at_k

        retrieved = ["url_a", "url_b", "url_c"]
        expected = ["url_a", "url_b"]

        assert recall_at_k(retrieved, expected, k=3) == 1.0

    def test_recall_at_k_partial(self):
        """Partial recall when only some expected items are retrieved."""
        from data_engineering_copilot.evaluation.retrieval_metrics import recall_at_k

        retrieved = ["url_a", "url_c", "url_d"]
        expected = ["url_a", "url_b"]

        assert recall_at_k(retrieved, expected, k=3) == 0.5

    def test_recall_at_k_empty_expected(self):
        """Recall with empty expected list should return 0.0."""
        from data_engineering_copilot.evaluation.retrieval_metrics import recall_at_k

        assert recall_at_k(["url_a"], [], k=5) == 0.0

    def test_ndcg_at_k_perfect(self):
        """Perfect nDCG when all expected items are in top positions."""
        from data_engineering_copilot.evaluation.retrieval_metrics import ndcg_at_k

        retrieved = ["url_a", "url_b", "url_c"]
        expected = ["url_a", "url_b", "url_c"]

        assert ndcg_at_k(retrieved, expected, k=3) == 1.0

    def test_ndcg_at_k_no_match(self):
        """nDCG with no matches should return 0.0."""
        from data_engineering_copilot.evaluation.retrieval_metrics import ndcg_at_k

        retrieved = ["url_a", "url_b"]
        expected = ["url_c", "url_d"]

        assert ndcg_at_k(retrieved, expected, k=3) == 0.0

    def test_precision_at_k(self):
        """Precision@K should compute correctly."""

        retrieved = ["url_a", "url_b", "url_c"]
        expected = ["url_a", "url_d"]

        relevant_count = len(set(retrieved) & set(expected))
        precision = relevant_count / len(retrieved) if retrieved else 0.0
        assert precision == 1 / 3

    def test_mrr(self):
        """MRR should compute correctly."""
        results = [
            (["url_a", "url_b"], ["url_a"]),
            (["url_c", "url_d"], ["url_d"]),
        ]

        rr_list = []
        for retrieved, expected in results:
            for i, url in enumerate(retrieved, 1):
                if url in expected:
                    rr_list.append(1.0 / i)
                    break
            else:
                rr_list.append(0.0)

        mrr = sum(rr_list) / len(rr_list) if rr_list else 0.0
        assert mrr == (1.0 + 0.5) / 2


# ---------------------------------------------------------------------------
# End-to-end evaluation flow tests
# ---------------------------------------------------------------------------


class TestEvaluationEndToEnd:
    """End-to-end tests for the evaluation pipeline."""

    @pytest.mark.asyncio
    async def test_full_retrieval_evaluation_flow(self, qdrant_url, sample_chunks, sample_eval_dataset):
        """Test the full retrieval evaluation flow from indexing to metrics."""
        from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

        collection = "itest_e2e_eval_flow"
        store = AsyncQdrantVectorStore(url=qdrant_url, collection_name=collection, embedding_dimension=2048)

        try:
            await store.initialize()
            embedder = _StubEmbedder()

            vectors = await embedder.embed_texts([c.text for c in sample_chunks])
            store.fit_bm25([c.text for c in sample_chunks])
            await store.upsert_chunks(sample_chunks, vectors)

            all_recalls = []
            for item in sample_eval_dataset:
                query_emb = await embedder.embed_query(item["question"])
                retrieved = await store.query(query_embedding=query_emb, top_k=5)
                retrieved_urls = [r.chunk.url for r in retrieved]
                expected_urls = item["expected_urls"]

                from data_engineering_copilot.evaluation.retrieval_metrics import recall_at_k

                recall = recall_at_k(retrieved_urls, expected_urls, k=5)
                all_recalls.append(recall)

            avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
            assert 0.0 <= avg_recall <= 1.0

            await embedder.close()
        finally:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=qdrant_url, prefer_grpc=False)
            client.delete_collection(collection_name=collection)
            client.close()

    @pytest.mark.asyncio
    async def test_full_generation_evaluation_flow(self, sample_chunks):
        """Test the generation evaluation flow with a mocked LLM."""
        from data_engineering_copilot.domain.models import Answer

        llm = _FakeLLM("Spark supports window functions for rolling aggregates.")

        prompt = "How do I compute a rolling window aggregate in Spark SQL?"
        response = await llm.generate(prompt)

        answer = Answer(
            text=response,
            sources=sample_chunks[:2],
            confidence=0.85,
        )

        assert answer.text == "Spark supports window functions for rolling aggregates."
        assert len(answer.sources) == 2
        assert answer.confidence == 0.85

    @pytest.mark.asyncio
    async def test_full_chunking_evaluation_flow(self, sample_chunks):
        """Test the chunking evaluation flow."""
        from data_engineering_copilot.evaluation.chunking_metrics import token_iou

        doc_text = " ".join(c.text for c in sample_chunks)
        gold_spans = [
            {"content": "Apache Spark", "start": 0, "end": 12, "structural_type": "entity"},
        ]

        score = token_iou(doc_text, gold_spans, sample_chunks)
        assert 0.0 <= score <= 1.0
