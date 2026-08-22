"""E2E RAG streaming tests: full pipeline with real Ollama + Qdrant.

Builds the RAG service with factory functions (custom settings, isolated collection).
Uses real Redis, Qdrant, and Ollama. Tests end-to-end answer generation.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import Answer, RagConfig, RawDocument
from data_engineering_copilot.infrastructure.html_to_markdown import MarkdownParser
from data_engineering_copilot.services.chunker import DocumentChunker
from tests.conftest import require_qdrant_and_ollama

SAMPLE_HTML = """<html><head><title>Delta Lake Guide</title></head><body>
<main>
<h1>Delta Lake Guide</h1>
<p>Delta Lake is an open-source storage framework that brings ACID transactions
to Apache Spark and big data workloads. It provides scalable metadata handling
and unifies streaming and batch data processing.</p>
<p>Delta Lake runs on top of your existing data lake and is fully compatible
with Apache Spark APIs. It simplifies building data pipelines by providing
transactions, schema enforcement, and time travel capabilities.</p>
</main></body></html>"""


@pytest.mark.e2e
class TestRagStreaming:
    """Journey 3: Ingest → RAG query → verify answer with real Ollama."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_infra(self, e2e_settings):
        require_qdrant_and_ollama(e2e_settings.qdrant_url)

    async def test_full_rag_pipeline_returns_answer(self, e2e_settings, e2e_embedder, e2e_llm):
        """Full RAG: parse → chunk → embed → upsert → retrieve → generate with real Ollama."""
        from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

        store = None
        try:
            store = AsyncQdrantVectorStore(
                url=e2e_settings.qdrant_url,
                collection_name=e2e_settings.collection_name,
                embedding_dimension=2048,
            )
            await store.initialize()

            raw = RawDocument(
                source_name="Delta Lake Documentation",
                url="https://delta.io/guide",
                html=SAMPLE_HTML,
            )
            parsed = MarkdownParser().parse(raw)
            assert parsed is not None

            chunker = DocumentChunker(chunk_size_chars=500, chunk_overlap_chars=100)
            chunks = await chunker.chunk(parsed)
            assert len(chunks) >= 1

            texts = [c.text for c in chunks]
            embeddings = await e2e_embedder.embed_texts(texts)
            await store.upsert_chunks(chunks, embeddings)

            from data_engineering_copilot.services.async_rag import AsyncRagService

            rag = AsyncRagService(
                config=RagConfig(),
                vector_store=store,
                llm_client=e2e_llm,
                embedder=e2e_embedder,
            )
            answer = await rag.answer("What is Delta Lake?")
            assert isinstance(answer, Answer)
            assert len(answer.text) > 20, f"Answer too short: {answer.text!r}"
            assert answer.confidence >= 0.0
            assert isinstance(answer.sources, tuple)
        finally:
            if store is not None:
                await store.close()

    async def test_rag_answer_cites_sources(self, e2e_settings, e2e_embedder, e2e_llm):
        """RAG answer should cite at least one source."""
        from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

        store = None
        try:
            store = AsyncQdrantVectorStore(
                url=e2e_settings.qdrant_url,
                collection_name=e2e_settings.collection_name,
                embedding_dimension=2048,
            )
            await store.initialize()

            raw = RawDocument(
                source_name="Delta Lake Documentation",
                url="https://delta.io/guide",
                html=SAMPLE_HTML,
            )
            parsed = MarkdownParser().parse(raw)
            assert parsed is not None
            chunker = DocumentChunker(chunk_size_chars=500, chunk_overlap_chars=100)
            chunks = await chunker.chunk(parsed)
            texts = [c.text for c in chunks]
            embeddings = await e2e_embedder.embed_texts(texts)
            await store.upsert_chunks(chunks, embeddings)

            from data_engineering_copilot.services.async_rag import AsyncRagService

            rag = AsyncRagService(
                config=RagConfig(),
                vector_store=store,
                llm_client=e2e_llm,
                embedder=e2e_embedder,
            )
            answer = await rag.answer("What is Delta Lake?")
            assert len(answer.sources) > 0, "Should cite at least one source"
        finally:
            if store is not None:
                await store.close()

    async def test_unrelated_question_acknowledges_gap(self, e2e_settings, e2e_embedder, e2e_llm):
        """An unrelated question should acknowledge the docs don't cover it."""
        from data_engineering_copilot.infrastructure.async_qdrant_store import AsyncQdrantVectorStore

        store = None
        try:
            store = AsyncQdrantVectorStore(
                url=e2e_settings.qdrant_url,
                collection_name=e2e_settings.collection_name,
                embedding_dimension=2048,
            )
            await store.initialize()

            raw = RawDocument(
                source_name="Delta Lake Documentation",
                url="https://delta.io/guide",
                html=SAMPLE_HTML,
            )
            parsed = MarkdownParser().parse(raw)
            assert parsed is not None
            chunker = DocumentChunker(chunk_size_chars=500, chunk_overlap_chars=100)
            chunks = await chunker.chunk(parsed)
            texts = [c.text for c in chunks]
            embeddings = await e2e_embedder.embed_texts(texts)
            await store.upsert_chunks(chunks, embeddings)

            from data_engineering_copilot.services.async_rag import AsyncRagService

            rag = AsyncRagService(
                config=RagConfig(),
                vector_store=store,
                llm_client=e2e_llm,
                embedder=e2e_embedder,
            )
            answer = await rag.answer("What is the capital of France?")
            text_lower = answer.text.lower()
            acknowledges = any(
                phrase in text_lower for phrase in ["cannot answer", "does not provide", "outside", "does not contain"]
            )
            assert acknowledges or answer.confidence < 0.5, (
                f"Expected gap acknowledgment. confidence={answer.confidence:.4f}, text={answer.text[:200]!r}"
            )
        finally:
            if store is not None:
                await store.close()
