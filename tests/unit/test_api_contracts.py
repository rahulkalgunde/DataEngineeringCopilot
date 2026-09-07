"""Contract tests for API signatures that caused repeated debugging time.

These pin down the exact signatures of constructors, methods, and properties
so that future test-writing doesn't have to guess. Each test here corresponds
to a real misunderstanding that cost 5-30 minutes of debugging in prior sessions.

Run with: pytest tests/unit/test_api_contracts.py -v
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from data_engineering_copilot.domain.exceptions import ProviderError, ProviderErrorCategory
from data_engineering_copilot.domain.models import CachedAnswer, DocumentChunk, RetrievedChunk

# ---------------------------------------------------------------------------
# Dataclass constructor contracts
# ---------------------------------------------------------------------------


class TestRetrievedChunkContract:
    """RetrievedChunk takes distance + confidence, NOT score."""

    def test_accepts_distance_and_confidence(self) -> None:
        chunk = DocumentChunk(
            chunk_id="c1",
            source_name="s",
            title="t",
            url="u",
            text="x",
        )
        rc = RetrievedChunk(chunk=chunk, distance=0.5, confidence=0.9)
        assert rc.distance == 0.5
        assert rc.confidence == 0.9

    def test_rejects_score_keyword(self) -> None:
        """Guarantees we never waste time trying score= again."""
        chunk = DocumentChunk(
            chunk_id="c1",
            source_name="s",
            title="t",
            url="u",
            text="x",
        )
        with pytest.raises(TypeError):
            RetrievedChunk(chunk=chunk, score=1.0)  # type: ignore[call-arg]


class TestCachedAnswerContract:
    """CachedAnswer uses sources (tuple) NOT citations (list)."""

    def test_accepts_sources_as_tuple(self) -> None:
        chunk = DocumentChunk(
            chunk_id="c1",
            source_name="s",
            title="t",
            url="u",
            text="x",
        )
        answer = CachedAnswer(text="test", sources=(chunk,), confidence=0.9)
        assert answer.sources == (chunk,)

    def test_rejects_citations_keyword(self) -> None:
        """Guarantees we never waste time trying citations= again."""
        with pytest.raises(TypeError):
            CachedAnswer(text="test", citations=[])  # type: ignore[call-arg]

    def test_sources_default_is_empty_tuple(self) -> None:
        answer = CachedAnswer(text="test")
        assert answer.sources == ()


class TestProviderErrorContract:
    """ProviderError takes (category, provider, model) positionally."""

    def test_positional_constructor(self) -> None:
        err = ProviderError(ProviderErrorCategory.RATE_LIMITED, "openrouter", "model-x")
        assert err.category == ProviderErrorCategory.RATE_LIMITED
        assert err.provider == "openrouter"
        assert err.model == "model-x"

    def test_category_is_enum_not_string(self) -> None:
        """Passing a string category would silently work but be wrong."""
        err = ProviderError(ProviderErrorCategory.PERMANENT_ERROR, "p", "m")
        assert isinstance(err.category, ProviderErrorCategory)


# ---------------------------------------------------------------------------
# Method name and type contracts
# ---------------------------------------------------------------------------


class TestQueryCacheContract:
    """QueryCache has aget/aset_exact, NOT get_or_compute. stats is a property."""

    def test_has_aget_exact_not_get_or_compute(self) -> None:
        from data_engineering_copilot.services.query_cache import QueryCache

        assert hasattr(QueryCache, "aget")
        assert hasattr(QueryCache, "aset_exact")
        assert not hasattr(QueryCache, "get_or_compute")

    def test_stats_is_property_not_method(self) -> None:
        from data_engineering_copilot.services.query_cache import QueryCache

        assert isinstance(inspect.getattr_static(QueryCache, "stats"), property)

    def test_is_cacheable_requires_sources(self) -> None:
        """An answer with empty sources is NOT cacheable — this silently
        caused cache misses until we pinned it down here."""
        from data_engineering_copilot.services.query_cache import QueryCache

        chunk = DocumentChunk(
            chunk_id="c1",
            source_name="s",
            title="t",
            url="u",
            text="x",
        )
        with_sources = CachedAnswer(text="ok", sources=(chunk,), confidence=0.9)
        without_sources = CachedAnswer(text="ok", sources=(), confidence=0.9)

        assert QueryCache.is_cacheable(with_sources) is True
        assert QueryCache.is_cacheable(without_sources) is False

    def test_is_cacheable_rejects_low_confidence(self) -> None:
        from data_engineering_copilot.services.query_cache import QueryCache

        chunk = DocumentChunk(
            chunk_id="c1",
            source_name="s",
            title="t",
            url="u",
            text="x",
        )
        low_conf = CachedAnswer(text="ok", sources=(chunk,), confidence=0.1)
        assert QueryCache.is_cacheable(low_conf, min_confidence=0.5) is False


class TestRelevanceGraderContract:
    """RelevanceGrader method is grade_chunks (not grades_relevance)."""

    def test_has_grade_chunks_not_grades_relevance(self) -> None:
        from data_engineering_copilot.services.relevance_grader import RelevanceGrader

        assert hasattr(RelevanceGrader, "grade_chunks")
        assert not hasattr(RelevanceGrader, "grades_relevance")

    def test_grade_chunks_is_async(self) -> None:
        from data_engineering_copilot.services.relevance_grader import RelevanceGrader

        assert inspect.iscoroutinefunction(RelevanceGrader.grade_chunks)


# ---------------------------------------------------------------------------
# Error constructor contracts
# ---------------------------------------------------------------------------


class TestLLMClientErrorContract:
    """LLMClientError uses ProviderErrorCategory enum, not raw strings."""

    def test_category_accepts_enum(self) -> None:
        from data_engineering_copilot.infrastructure.llm_client import LLMClientError

        err = LLMClientError(
            "fail",
            category=ProviderErrorCategory.RATE_LIMITED,
        )
        assert err.category == ProviderErrorCategory.RATE_LIMITED

    def test_has_response_body_field(self) -> None:
        from data_engineering_copilot.infrastructure.llm_client import LLMClientError

        err = LLMClientError("fail", response_body="raw body")
        assert err.response_body == "raw body"


# ---------------------------------------------------------------------------
# Factory / settings contracts
# ---------------------------------------------------------------------------


class TestFactoryEmbeddingContract:
    """Embedding chain is NVIDIA-only by decision (2026-09-07).

    Last 5 gen-builds: NVIDIA served 99.9% of embedding requests, openrouter 0%
    (always 503), huggingface 0.1% (transient NVIDIA blips). Abstraction of the
    chain into a single provider keeps failures fast and config simple.
    """

    def test_default_embedding_order_is_nvidia_only(self) -> None:
        from data_engineering_copilot.config.settings import AppSettings

        settings = AppSettings()
        assert settings.embedding_fallback_order == ["nvidia"]
        assert settings.offline_embedding_fallback_order == ["nvidia"]

    def test_build_rag_service_works_with_no_api_keys(self) -> None:
        """Hermetic tests pin local-hf in make_settings() so the no-api-key
        chain still builds (local-hf stays a standalone provider even though it
        is no longer a production chain fallback)."""
        from data_engineering_copilot.factory import build_rag_service
        from tests.conftest import make_settings

        settings = make_settings()
        service = build_rag_service(app_settings=settings)
        assert service is not None

    def test_single_provider_chain_is_wrapped_not_bare_client(self) -> None:
        """2026-09-07 decision: nvidia-only embedding order must NOT short-circuit
        to the bare client. The single provider still goes through
        ProviderFallbackChain so health cooldowns, the offline wait controller,
        and 503 exponential backoff apply (a bare client only had ~15s of
        tenacity retries — cannot ride a multi-minute provider outage)."""
        from data_engineering_copilot.factory import build_embedding_fallback_chain
        from data_engineering_copilot.infrastructure.async_openai_compatible_embeddings import (
            OpenAICompatibleEmbeddings,
        )
        from data_engineering_copilot.infrastructure.offline_embedding_wait import (
            OfflineEmbeddingWaitController,
        )
        from data_engineering_copilot.infrastructure.provider_fallback import ProviderFallbackChain
        from tests.conftest import make_settings

        s = make_settings(
            embedding_fallback_order=["nvidia"],
            offline_embedding_fallback_order=["nvidia"],
            nvidia_api_key="nvapi-placeholder",
            _test_allow_non_ollama=True,
        )
        assert s.embedding_fallback_order == ["nvidia"]

        chain = build_embedding_fallback_chain(purpose="global", app_settings=s)
        assert isinstance(chain, ProviderFallbackChain)
        assert not isinstance(chain, OpenAICompatibleEmbeddings)
        assert [p.name for p in chain._config.providers] == ["nvidia"]

        offline = build_embedding_fallback_chain(purpose="offline_batch", app_settings=s)
        assert isinstance(offline, OfflineEmbeddingWaitController)
        assert not isinstance(offline, OpenAICompatibleEmbeddings)
        assert isinstance(offline.inner, ProviderFallbackChain)
        assert hasattr(offline, "embed_texts")
        assert hasattr(offline, "embed_query")


# ---------------------------------------------------------------------------
# MagicMock contract
# ---------------------------------------------------------------------------


class TestMockingContract:
    """MagicMock without spec= makes hasattr always return True."""

    def test_unspec_mock_hasattr_always_true(self) -> None:
        """This is why test_fallback_embedder failed: hasattr(mock, 'execute')
        returned True even when we didn't want it to."""
        mock = MagicMock()
        assert hasattr(mock, "execute")
        assert hasattr(mock, "anything_at_all")

    def test_spec_mock_hasattr_respects_spec(self) -> None:
        mock = MagicMock(spec=["execute"])
        assert hasattr(mock, "execute")
        assert not hasattr(mock, "embed_texts")


# ---------------------------------------------------------------------------
# Numeric contract
# ---------------------------------------------------------------------------


class TestNdcgContract:
    """nDCG with binary relevance returns 1.0 when ALL expected items are present,
    regardless of position. We got the assertion wrong."""

    def test_ndcg_all_present_is_one(self) -> None:
        from data_engineering_copilot.evaluation.retrieval_metrics import ndcg_at_k

        result = ndcg_at_k(["a", "b", "c"], ["b", "a"], k=3)
        assert result == 1.0

    def test_ndcg_partial_is_less_than_one(self) -> None:
        from data_engineering_copilot.evaluation.retrieval_metrics import ndcg_at_k

        result = ndcg_at_k(["a", "b", "c"], ["a", "d"], k=3)
        assert 0 < result < 1.0

    def test_recall_all_present_is_one(self) -> None:
        from data_engineering_copilot.evaluation.retrieval_metrics import recall_at_k

        result = recall_at_k(["a", "b", "c"], ["b", "a"], k=3)
        assert result == 1.0
