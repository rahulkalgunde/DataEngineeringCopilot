"""Comprehensive tests to boost coverage to 100% for all non-UI modules."""

import json
import socket
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, Mock, patch, PropertyMock
from urllib.error import URLError

import pytest

from data_engineering_copilot.domain.models import (
    Answer,
    DocumentChunk,
    IngestionEvent,
    ParsedDocument,
    RawDocument,
    RetrievedChunk,
)


# =============================================================================
# services/reranker.py (37% → 100%)
# =============================================================================
class TestReranker:
    @patch("data_engineering_copilot.services.reranker.CrossEncoder", create=True)
    def test_cross_encoder_reranker_init_success(self, mock_ce_class):
        mock_model = MagicMock()
        mock_ce_class.return_value = mock_model

        from data_engineering_copilot.services.reranker import CrossEncoderReranker

        with patch.dict("sys.modules", {"sentence_transformers": MagicMock(CrossEncoder=mock_ce_class)}):
            reranker = CrossEncoderReranker(model_name="test-model")
            assert reranker.model is mock_model
            assert reranker.model_name == "test-model"
            assert reranker.is_available() is True

    def test_cross_encoder_reranker_init_import_error(self):
        from data_engineering_copilot.services.reranker import CrossEncoderReranker

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            reranker = CrossEncoderReranker(model_name="test-model")
            assert reranker.model is None
            assert reranker.is_available() is False

    def test_rerank_empty_chunks(self):
        from data_engineering_copilot.services.reranker import CrossEncoderReranker

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            reranker = CrossEncoderReranker(model_name="test-model")
            result = reranker.rerank("query", [], top_k=5)
            assert result == []

    def test_rerank_no_model_fallback(self):
        from data_engineering_copilot.services.reranker import CrossEncoderReranker

        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id=f"i", source_name="s", title="t", url="u", text=f"text {i}"),
                distance=0.1,
                confidence=0.9 - i * 0.1,
            )
            for i in range(5)
        ]

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            reranker = CrossEncoderReranker(model_name="test-model")
            result = reranker.rerank("query", chunks, top_k=3)
            assert len(result) == 3
            assert result == chunks[:3]

    def test_rerank_fewer_chunks_than_top_k(self):
        from data_engineering_copilot.services.reranker import CrossEncoderReranker

        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="1", source_name="s", title="t", url="u", text="text"),
                distance=0.1,
                confidence=0.9,
            )
        ]

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model = mock_model
        reranker.model_name = "test"

        result = reranker.rerank("query", chunks, top_k=5)
        assert len(result) == 1
        assert result == chunks

    def test_rerank_success_with_scoring(self):
        from data_engineering_copilot.services.reranker import CrossEncoderReranker

        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id=f"{i}", source_name="s", title="t", url="u", text=f"text {i}"),
                distance=0.1,
                confidence=0.9,
            )
            for i in range(5)
        ]

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5, 0.3, 0.7]

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model = mock_model
        reranker.model_name = "test"

        result = reranker.rerank("query", chunks, top_k=3)
        assert len(result) == 3
        # highest score (0.9) → chunk_id "1" (second chunk, index 1)
        assert result[0].chunk.chunk_id == "1"

    def test_rerank_exception_fallback(self):
        from data_engineering_copilot.services.reranker import CrossEncoderReranker

        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id=f"i", source_name="s", title="t", url="u", text=f"text {i}"),
                distance=0.1,
                confidence=0.9,
            )
            for i in range(5)
        ]

        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("model error")

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model = mock_model
        reranker.model_name = "test"

        result = reranker.rerank("query", chunks, top_k=3)
        assert len(result) == 3


# =============================================================================
# services/metrics.py (48% → 100%)
# =============================================================================
class TestMetrics:
    def test_retrieval_metrics_str(self):
        from data_engineering_copilot.services.metrics import RetrievalMetrics

        m = RetrievalMetrics(query="test query", retrieved_count=5, top_confidence=0.8)
        s = str(m)
        assert "RetrievalMetrics" in s
        assert "test query" in s

    def test_answer_metrics_str(self):
        from data_engineering_copilot.services.metrics import AnswerMetrics

        m = AnswerMetrics(
            question="test question",
            answer_length=10,
            answer_chars=100,
            has_key_sections=True,
            has_uncertainty_markers=False,
            source_count=3,
        )
        s = str(m)
        assert "AnswerMetrics" in s

    def test_query_metrics_str(self):
        from data_engineering_copilot.services.metrics import QueryMetrics

        m = QueryMetrics(query="test", query_difficulty="easy", was_answered=True, confidence_score=0.9)
        s = str(m)
        assert "QueryMetrics" in s
        assert "easy" in s

    def test_classify_query_difficulty_easy(self):
        from data_engineering_copilot.services.metrics import MetricsCollector

        mc = MetricsCollector()
        assert mc.classify_query_difficulty("What is Spark?") == "easy"

    def test_classify_query_difficulty_medium(self):
        from data_engineering_copilot.services.metrics import MetricsCollector

        mc = MetricsCollector()
        assert mc.classify_query_difficulty("How does Spark handle shuffle operations and data partitioning?") == "medium"

    def test_classify_query_difficulty_hard(self):
        from data_engineering_copilot.services.metrics import MetricsCollector

        mc = MetricsCollector()
        assert mc.classify_query_difficulty(
            "Compare the differences between Spark SQL and Hive, and explain how they handle partitioning and bucketing strategies"
        ) == "hard"

    def test_compute_retrieval_metrics_empty(self):
        from data_engineering_copilot.services.metrics import MetricsCollector

        mc = MetricsCollector()
        m = mc.compute_retrieval_metrics("query", [])
        assert m.retrieved_count == 0
        assert m.top_confidence == 0.0

    def test_compute_retrieval_metrics_success(self):
        from data_engineering_copilot.services.metrics import MetricsCollector

        mc = MetricsCollector()
        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="1", source_name="s", title="t", url="u", text="text"),
                distance=0.1,
                confidence=0.9,
            ),
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="2", source_name="s", title="t", url="u", text="text2"),
                distance=0.3,
                confidence=0.7,
            ),
        ]
        m = mc.compute_retrieval_metrics("query", chunks, top_k=5)
        assert m.retrieved_count == 2
        assert m.top_confidence == 0.9
        assert m.mrr == 1.0
        assert m.precision_at_3 >= 0

    def test_compute_answer_metrics(self):
        from data_engineering_copilot.services.metrics import MetricsCollector

        mc = MetricsCollector()
        answer = Answer(text="Answer: This is the answer.\nKey Points: point1, point2", sources=tuple(), confidence=0.8)
        m = mc.compute_answer_metrics("question", answer)
        assert m.answer_length > 0
        assert m.has_key_sections is True
        assert m.has_uncertainty_markers is False

    def test_compute_answer_metrics_with_uncertainty(self):
        from data_engineering_copilot.services.metrics import MetricsCollector

        mc = MetricsCollector()
        answer = Answer(
            text="The documentation does not clearly address this.",
            sources=tuple(),
            confidence=0.1,
        )
        m = mc.compute_answer_metrics("question", answer)
        assert m.has_uncertainty_markers is True

    def test_record_query_with_answer(self):
        from data_engineering_copilot.services.metrics import MetricsCollector

        mc = MetricsCollector()
        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="1", source_name="s", title="t", url="u", text="text"),
                distance=0.1,
                confidence=0.9,
            ),
        ]
        answer = Answer(text="Answer: test", sources=tuple(), confidence=0.8)
        qm = mc.record_query("test query", chunks, answer=answer, was_answered=True)
        assert qm.was_answered is True
        assert qm.answer_metrics is not None
        assert len(mc.queries) == 1

    def test_record_query_without_answer(self):
        from data_engineering_copilot.services.metrics import MetricsCollector

        mc = MetricsCollector()
        qm = mc.record_query("test query", [], was_answered=False)
        assert qm.was_answered is False
        assert qm.answer_metrics is None

    def test_get_session_summary_empty(self):
        from data_engineering_copilot.services.metrics import MetricsCollector

        mc = MetricsCollector()
        summary = mc.get_session_summary()
        assert summary["total_queries"] == 0
        assert summary["answer_rate"] == 0.0

    def test_get_session_summary_with_data(self):
        from data_engineering_copilot.services.metrics import MetricsCollector

        mc = MetricsCollector()
        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="1", source_name="s", title="t", url="u", text="text"),
                distance=0.1,
                confidence=0.9,
            ),
        ]
        mc.record_query("easy query", chunks, was_answered=True)
        mc.record_query("medium query", chunks, was_answered=False)
        summary = mc.get_session_summary()
        assert summary["total_queries"] == 2
        assert summary["answered_queries"] == 1


# =============================================================================
# services/context_assembler.py (55% → 100%)
# =============================================================================
class TestContextAssembler:
    def test_assemble_empty(self):
        from data_engineering_copilot.services.context_assembler import ContextAssembler

        assembler = ContextAssembler(max_context_chars=1000)
        ctx, sources = assembler.assemble([])
        assert ctx == ""
        assert sources == []

    def test_assemble_single_chunk(self):
        from data_engineering_copilot.services.context_assembler import ContextAssembler

        assembler = ContextAssembler(max_context_chars=1000)
        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="1", source_name="source1", title="t", url="u", text="hello world"),
                distance=0.1,
                confidence=0.9,
            ),
        ]
        ctx, sources = assembler.assemble(chunks)
        assert "source1" in ctx
        assert "hello world" in ctx
        assert sources == ["source1"]

    def test_assemble_truncation(self):
        from data_engineering_copilot.services.context_assembler import ContextAssembler

        assembler = ContextAssembler(max_context_chars=50)
        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id=str(i), source_name=f"src{i}", title="t", url="u", text="x " * 20),
                distance=0.1,
                confidence=0.9,
            )
            for i in range(5)
        ]
        ctx, sources = assembler.assemble(chunks)
        assert len(ctx) <= 60  # Allow some slack for formatting

    def test_deduplication_removes_overlapping(self):
        from data_engineering_copilot.services.context_assembler import ContextAssembler

        assembler = ContextAssembler(max_context_chars=10000)
        text = "the quick brown fox jumps over the lazy dog"
        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="1", source_name="s", title="t", url="u", text=text),
                distance=0.1,
                confidence=0.9,
            ),
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="2", source_name="s", title="t", url="u", text=text),
                distance=0.1,
                confidence=0.9,
            ),
        ]
        ctx, sources = assembler.assemble(chunks)
        assert sources.count("s") == 1

    def test_text_overlap_ratio_identical(self):
        from data_engineering_copilot.services.context_assembler import ContextAssembler

        assembler = ContextAssembler(max_context_chars=1000)
        ratio = assembler._text_overlap_ratio("hello world foo", "hello world foo")
        assert ratio == 1.0

    def test_text_overlap_ratio_no_overlap(self):
        from data_engineering_copilot.services.context_assembler import ContextAssembler

        assembler = ContextAssembler(max_context_chars=1000)
        ratio = assembler._text_overlap_ratio("apple banana cherry", "delta epsilon zeta")
        assert ratio == 0.0

    def test_text_overlap_ratio_empty(self):
        from data_engineering_copilot.services.context_assembler import ContextAssembler

        assembler = ContextAssembler(max_context_chars=1000)
        ratio = assembler._text_overlap_ratio("", "")
        assert ratio == 0.0


# =============================================================================
# observability/langfuse_client.py (56% → 100%)
# =============================================================================
class TestLangfuseClient:
    def test_candidate_hosts_empty(self):
        from data_engineering_copilot.observability.langfuse_client import _candidate_langfuse_hosts

        assert _candidate_langfuse_hosts("") == []
        assert _candidate_langfuse_hosts("  ") == []

    def test_candidate_hosts_localhost(self):
        from data_engineering_copilot.observability.langfuse_client import _candidate_langfuse_hosts

        result = _candidate_langfuse_hosts("http://localhost:3000")
        assert "http://localhost:3000" in result
        assert "http://127.0.0.1:3000" in result

    def test_candidate_hosts_127(self):
        from data_engineering_copilot.observability.langfuse_client import _candidate_langfuse_hosts

        result = _candidate_langfuse_hosts("http://127.0.0.1:3000")
        assert "http://127.0.0.1:3000" in result
        assert "http://localhost:3000" in result

    def test_candidate_hosts_docker_name(self):
        from data_engineering_copilot.observability.langfuse_client import _candidate_langfuse_hosts

        result = _candidate_langfuse_hosts("http://langfuse:3000")
        assert "http://langfuse:3000" in result
        assert "http://localhost:3000" in result
        assert "http://127.0.0.1:3000" in result

    def test_candidate_hosts_no_port(self):
        from data_engineering_copilot.observability.langfuse_client import _candidate_langfuse_hosts

        result = _candidate_langfuse_hosts("http://langfuse")
        assert "http://langfuse" in result

    def test_candidate_hosts_bare_host(self):
        from data_engineering_copilot.observability.langfuse_client import _candidate_langfuse_hosts

        result = _candidate_langfuse_hosts("langfuse:3000")
        assert len(result) > 0

    def test_check_langfuse_health_success(self):
        from data_engineering_copilot.observability.langfuse_client import _check_langfuse_health

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({"status": "OK"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("data_engineering_copilot.observability.langfuse_client.urllib.request.urlopen", return_value=mock_resp):
            assert _check_langfuse_health("http://localhost:3000") is True

    def test_check_langfuse_health_non_ok_status(self):
        from data_engineering_copilot.observability.langfuse_client import _check_langfuse_health

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({"status": "ERROR"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("data_engineering_copilot.observability.langfuse_client.urllib.request.urlopen", return_value=mock_resp):
            assert _check_langfuse_health("http://localhost:3000") is False

    def test_check_langfuse_health_http_error(self):
        from data_engineering_copilot.observability.langfuse_client import _check_langfuse_health

        mock_resp = MagicMock()
        mock_resp.status = 500
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("data_engineering_copilot.observability.langfuse_client.urllib.request.urlopen", return_value=mock_resp):
            assert _check_langfuse_health("http://localhost:3000") is False

    def test_check_langfuse_health_url_error(self):
        from data_engineering_copilot.observability.langfuse_client import _check_langfuse_health

        with patch("data_engineering_copilot.observability.langfuse_client.urllib.request.urlopen", side_effect=URLError("refused")):
            assert _check_langfuse_health("http://localhost:3000") is False

    def test_check_langfuse_health_unexpected_error(self):
        from data_engineering_copilot.observability.langfuse_client import _check_langfuse_health

        with patch("data_engineering_copilot.observability.langfuse_client.urllib.request.urlopen", side_effect=ValueError("weird")):
            assert _check_langfuse_health("http://localhost:3000") is False

    def test_get_langfuse_instance_all_fail(self):
        from data_engineering_copilot.observability.langfuse_client import get_langfuse_instance

        with patch("data_engineering_copilot.observability.langfuse_client._check_langfuse_health", return_value=False):
            result = get_langfuse_instance()
            assert result is None

    def test_get_langfuse_instance_import_error(self):
        from data_engineering_copilot.observability.langfuse_client import get_langfuse_instance

        with patch.dict("sys.modules", {"langfuse": None}):
            result = get_langfuse_instance()
            assert result is None

    def test_get_langfuse_instance_auth_fails(self):
        from data_engineering_copilot.observability.langfuse_client import get_langfuse_instance

        mock_lf_class = MagicMock()
        mock_lf = MagicMock()
        mock_lf.auth_check.return_value = False
        mock_lf_class.return_value = mock_lf

        mock_langfuse_mod = MagicMock(Langfuse=mock_lf_class)

        with patch("data_engineering_copilot.observability.langfuse_client._check_langfuse_health", return_value=True), \
             patch.dict("sys.modules", {"langfuse": mock_langfuse_mod}):
            result = get_langfuse_instance()
            assert result is None

    def test_observation_compat_update(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        mock_obs = MagicMock()
        mock_obs.update = MagicMock(return_value=mock_obs)
        compat = _ObservationCompat(MagicMock(), mock_obs, "span")
        result = compat.update(output="test")
        assert result is compat
        mock_obs.update.assert_called_once_with(output="test")

    def test_observation_compat_update_none(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        compat = _ObservationCompat(MagicMock(), None, "span")
        result = compat.update(output="test")
        assert result is compat

    def test_observation_compat_end(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        mock_obs = MagicMock()
        compat = _ObservationCompat(MagicMock(), mock_obs, "span")
        compat.end()
        mock_obs.end.assert_called_once()

    def test_observation_compat_end_none(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        compat = _ObservationCompat(MagicMock(), None, "span")
        result = compat.end()
        assert result is compat

    def test_observation_compat_log_event_with_dict(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        mock_obs = MagicMock()
        compat = _ObservationCompat(MagicMock(), mock_obs, "span")
        compat.log_event(name="test", key="value")
        mock_obs.log_event.assert_called_once()

    def test_observation_compat_log_event_with_payload(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        mock_obs = MagicMock()
        compat = _ObservationCompat(MagicMock(), mock_obs, "span")
        compat.log_event(name="test", payload="data")
        mock_obs.log_event.assert_called_once()

    def test_observation_compat_log_event_create_event_fallback(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        mock_obs = MagicMock(spec=[])  # no log_event
        mock_obs.create_event = MagicMock()
        compat = _ObservationCompat(MagicMock(), mock_obs, "span")
        compat.log_event(name="test")
        mock_obs.create_event.assert_called_once()

    def test_observation_compat_log_event_start_observation_fallback(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        mock_obs = MagicMock(spec=[])  # no log_event or create_event
        mock_client = MagicMock()
        mock_child = MagicMock()
        mock_client._client.start_observation.return_value = mock_child
        compat = _ObservationCompat(mock_client, mock_obs, "span", trace_id="tid")
        compat.log_event(name="test")
        mock_client._client.start_observation.assert_called_once()

    def test_observation_compat_log_event_none_obs(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        compat = _ObservationCompat(MagicMock(), None, "span")
        result = compat.log_event(name="test")
        assert result is compat

    def test_observation_compat_getattr(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        mock_obs = MagicMock()
        mock_obs.custom_attr = "value"
        compat = _ObservationCompat(MagicMock(), mock_obs, "span")
        assert compat.custom_attr == "value"

    def test_observation_compat_start_observation_v3(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        mock_client = MagicMock()
        mock_client._client.start_observation = MagicMock(return_value=MagicMock(id="child-id"))
        compat = _ObservationCompat(mock_client, MagicMock(id="parent-id"), "trace", trace_id="trace-id")
        child = compat.start_observation(name="child", as_type="span")
        assert child is not None

    def test_observation_compat_start_observation_v2_fallback(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        mock_client_inner = MagicMock(spec=["trace", "span", "generation"])
        mock_client = MagicMock()
        mock_client._client = mock_client_inner
        mock_client_inner.span.return_value = MagicMock(id="child-id")
        compat = _ObservationCompat(mock_client, MagicMock(id="parent-id"), "trace", trace_id="trace-id")
        child = compat.start_observation(name="child", as_type="span")
        assert child is not None

    def test_observation_compat_trace(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        mock_client = MagicMock()
        mock_client._client.start_observation = MagicMock(return_value=MagicMock(id="t"))
        compat = _ObservationCompat(mock_client, MagicMock(), "trace", trace_id="tid")
        child = compat.trace(name="child-trace")
        assert child is not None

    def test_observation_compat_span(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        mock_client = MagicMock()
        mock_client._client.start_observation = MagicMock(return_value=MagicMock(id="s"))
        compat = _ObservationCompat(mock_client, MagicMock(), "trace", trace_id="tid")
        child = compat.span(name="child-span")
        assert child is not None

    def test_observation_compat_generation(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        mock_client = MagicMock()
        mock_client._client.start_observation = MagicMock(return_value=MagicMock(id="g"))
        compat = _ObservationCompat(mock_client, MagicMock(), "trace", trace_id="tid")
        child = compat.generation(name="child-gen")
        assert child is not None

    def test_langfuse_compat_v3_start_observation(self):
        from data_engineering_copilot.observability.langfuse_client import LangfuseCompat

        mock_client = MagicMock()
        mock_client.start_observation = MagicMock(return_value=MagicMock(id="obs"))
        compat = LangfuseCompat(mock_client)
        obs = compat.start_observation(name="test", as_type="trace")
        assert obs is not None
        mock_client.start_observation.assert_called_once()

    def test_langfuse_compat_v2_fallback(self):
        from data_engineering_copilot.observability.langfuse_client import LangfuseCompat

        mock_client = MagicMock(spec=["trace", "span", "generation", "auth_check", "flush"])
        mock_client.trace.return_value = MagicMock(id="trace-id")
        compat = LangfuseCompat(mock_client)
        obs = compat.start_observation(name="test", as_type="trace")
        assert obs is not None
        mock_client.trace.assert_called_once()

    def test_langfuse_compat_trace(self):
        from data_engineering_copilot.observability.langfuse_client import LangfuseCompat

        mock_client = MagicMock(spec=["trace", "span", "generation", "auth_check", "flush"])
        mock_client.trace.return_value = MagicMock(id="t")
        compat = LangfuseCompat(mock_client)
        obs = compat.trace(name="test")
        assert obs is not None

    def test_langfuse_compat_span(self):
        from data_engineering_copilot.observability.langfuse_client import LangfuseCompat

        mock_client = MagicMock(spec=["trace", "span", "generation", "auth_check", "flush"])
        mock_client.span.return_value = MagicMock(id="s")
        compat = LangfuseCompat(mock_client)
        obs = compat.span(name="test")
        assert obs is not None

    def test_langfuse_compat_generation(self):
        from data_engineering_copilot.observability.langfuse_client import LangfuseCompat

        mock_client = MagicMock(spec=["trace", "span", "generation", "auth_check", "flush"])
        mock_client.generation.return_value = MagicMock(id="g")
        compat = LangfuseCompat(mock_client)
        obs = compat.generation(name="test")
        assert obs is not None

    def test_langfuse_compat_auth_check(self):
        from data_engineering_copilot.observability.langfuse_client import LangfuseCompat

        mock_client = MagicMock()
        mock_client.auth_check.return_value = True
        compat = LangfuseCompat(mock_client)
        assert compat.auth_check() is True

    def test_langfuse_compat_flush(self):
        from data_engineering_copilot.observability.langfuse_client import LangfuseCompat

        mock_client = MagicMock()
        compat = LangfuseCompat(mock_client)
        compat.flush()
        mock_client.flush.assert_called_once()

    def test_langfuse_compat_flush_no_method(self):
        from data_engineering_copilot.observability.langfuse_client import LangfuseCompat

        mock_client = MagicMock(spec=[])
        compat = LangfuseCompat(mock_client)
        compat.flush()  # should not raise

    def test_langfuse_compat_getattr(self):
        from data_engineering_copilot.observability.langfuse_client import LangfuseCompat

        mock_client = MagicMock()
        mock_client.custom_attr = "value"
        compat = LangfuseCompat(mock_client)
        assert compat.custom_attr == "value"


# =============================================================================
# infrastructure/qdrant_store.py (55% → 100%)
# =============================================================================
class TestQdrantStoreFull:
    @pytest.fixture
    def mock_store(self):
        with patch("data_engineering_copilot.infrastructure.qdrant_store.QdrantClient") as mock_cls:
            mock_client = Mock()
            mock_client.collection_exists.return_value = True
            mock_cls.return_value = mock_client
            yield mock_client

    def test_upsert_chunks_client_none(self):
        with patch("data_engineering_copilot.infrastructure.qdrant_store.QdrantClient") as mock_cls:
            mock_cls.side_effect = Exception("init fail")
            try:
                store = __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore(
                    url="http://localhost:6333", collection_name="test"
                )
            except Exception:
                pass
            # Test with None client
            store = __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore.__new__(
                __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore
            )
            store._client = None
            store._collection_name = "test"
            store._url = "http://localhost:6333"
            store.upsert_chunks([], [])  # should return without error

    def test_query_client_none(self):
        store = __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore.__new__(
            __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore
        )
        store._client = None
        store._collection_name = "test"
        store._url = "http://localhost:6333"
        result = store.query([], top_k=5)
        assert result == []

    def test_count_client_none(self):
        store = __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore.__new__(
            __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore
        )
        store._client = None
        store._collection_name = "test"
        store._url = "http://localhost:6333"
        assert store.count() == 0

    def test_get_content_hash_for_url_client_none(self):
        store = __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore.__new__(
            __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore
        )
        store._client = None
        store._collection_name = "test"
        store._url = "http://localhost:6333"
        assert store.get_content_hash_for_url("http://example.com") is None

    def test_delete_by_url_client_none(self):
        store = __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore.__new__(
            __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore
        )
        store._client = None
        store._collection_name = "test"
        store._url = "http://localhost:6333"
        store.delete_by_url("http://example.com")  # should not raise

    def test_upsert_chunks_exception(self, mock_store):
        mock_store.upsert.side_effect = Exception("upsert failed")
        store = __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore.__new__(
            __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore
        )
        store._client = mock_store
        store._collection_name = "test"
        store._url = "http://localhost:6333"
        chunk = DocumentChunk(chunk_id="1", source_name="s", title="t", url="u", text="text")
        with pytest.raises(Exception):
            store.upsert_chunks([chunk], [[0.1] * 768])

    def test_query_404_error(self, mock_store):
        mock_store.query_points.side_effect = Exception("404 Not Found")
        store = __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore.__new__(
            __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore
        )
        store._client = mock_store
        store._collection_name = "test"
        store._url = "http://localhost:6333"
        result = store.query([0.1] * 768, top_k=5)
        assert result == []

    def test_query_other_exception(self, mock_store):
        mock_store.query_points.side_effect = Exception("some other error")
        store = __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore.__new__(
            __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore
        )
        store._client = mock_store
        store._collection_name = "test"
        store._url = "http://localhost:6333"
        with pytest.raises(Exception):
            store.query([0.1] * 768, top_k=5)

    def test_get_content_hash_found(self, mock_store):
        mock_point = Mock()
        mock_point.payload = {"content_hash": "abc123"}
        mock_store.scroll.return_value = ([mock_point], None)
        store = __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore.__new__(
            __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore
        )
        store._client = mock_store
        store._collection_name = "test"
        store._url = "http://localhost:6333"
        result = store.get_content_hash_for_url("http://example.com")
        assert result == "abc123"

    def test_get_content_hash_not_found(self, mock_store):
        mock_store.scroll.return_value = ([], None)
        store = __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore.__new__(
            __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore
        )
        store._client = mock_store
        store._collection_name = "test"
        store._url = "http://localhost:6333"
        result = store.get_content_hash_for_url("http://example.com")
        assert result is None

    def test_get_content_hash_exception(self, mock_store):
        mock_store.scroll.side_effect = Exception("scroll error")
        store = __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore.__new__(
            __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore
        )
        store._client = mock_store
        store._collection_name = "test"
        store._url = "http://localhost:6333"
        result = store.get_content_hash_for_url("http://example.com")
        assert result is None

    def test_delete_by_url_success(self, mock_store):
        store = __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore.__new__(
            __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore
        )
        store._client = mock_store
        store._collection_name = "test"
        store._url = "http://localhost:6333"
        store.delete_by_url("http://example.com")
        mock_store.delete.assert_called_once()

    def test_delete_by_url_exception(self, mock_store):
        mock_store.delete.side_effect = Exception("delete error")
        store = __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore.__new__(
            __import__("data_engineering_copilot.infrastructure.qdrant_store", fromlist=["QdrantVectorStore"]).QdrantVectorStore
        )
        store._client = mock_store
        store._collection_name = "test"
        store._url = "http://localhost:6333"
        store.delete_by_url("http://example.com")  # should not raise


# =============================================================================
# infrastructure/vector_store.py (68% → 100%)
# =============================================================================
class TestVectorStoreAdapter:
    def test_adapter_upsert(self):
        from data_engineering_copilot.infrastructure.vector_store import QdrantVectorStore

        with patch("data_engineering_copilot.infrastructure.vector_store.QdrantStoreImpl") as mock_impl:
            store = QdrantVectorStore(persist_directory="/tmp", collection_name="test")
            store.upsert_chunks([], [])
            mock_impl.return_value.upsert_chunks.assert_called_once()

    def test_adapter_query(self):
        from data_engineering_copilot.infrastructure.vector_store import QdrantVectorStore

        with patch("data_engineering_copilot.infrastructure.vector_store.QdrantStoreImpl") as mock_impl:
            mock_impl.return_value.query.return_value = []
            store = QdrantVectorStore(persist_directory="/tmp", collection_name="test")
            result = store.query([0.1] * 768, top_k=5)
            assert result == []

    def test_adapter_count(self):
        from data_engineering_copilot.infrastructure.vector_store import QdrantVectorStore

        with patch("data_engineering_copilot.infrastructure.vector_store.QdrantStoreImpl") as mock_impl:
            mock_impl.return_value.count.return_value = 42
            store = QdrantVectorStore(persist_directory="/tmp", collection_name="test")
            assert store.count() == 42

    def test_adapter_hybrid_query(self):
        from data_engineering_copilot.infrastructure.vector_store import QdrantVectorStore

        with patch("data_engineering_copilot.infrastructure.vector_store.QdrantStoreImpl") as mock_impl:
            mock_impl.return_value.hybrid_query.return_value = []
            store = QdrantVectorStore(persist_directory="/tmp", collection_name="test")
            result = store.hybrid_query("test", top_k=5)
            assert result == []

    def test_adapter_get_content_hash(self):
        from data_engineering_copilot.infrastructure.vector_store import QdrantVectorStore

        with patch("data_engineering_copilot.infrastructure.vector_store.QdrantStoreImpl") as mock_impl:
            mock_impl.return_value.get_content_hash_for_url.return_value = "hash"
            store = QdrantVectorStore(persist_directory="/tmp", collection_name="test")
            result = store.get_content_hash_for_url("http://example.com")
            assert result == "hash"

    def test_adapter_delete_by_url(self):
        from data_engineering_copilot.infrastructure.vector_store import QdrantVectorStore

        with patch("data_engineering_copilot.infrastructure.vector_store.QdrantStoreImpl") as mock_impl:
            store = QdrantVectorStore(persist_directory="/tmp", collection_name="test")
            store.delete_by_url("http://example.com")
            mock_impl.return_value.delete_by_url.assert_called_once_with("http://example.com")


# =============================================================================
# infrastructure/crawler.py (80% → 100%)
# =============================================================================
class TestCrawlerFull:
    def test_extract_links_skips_mailto(self):
        from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler

        crawler = DocumentationCrawler(timeout_seconds=5, delay_seconds=0)
        html = '<a href="mailto:test@example.com">email</a><a href="http://example.com/page">link</a>'
        links = crawler._extract_links(html, "http://example.com/")
        assert len(links) == 1
        assert "example.com/page" in links[0]

    def test_extract_links_skips_tel(self):
        from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler

        crawler = DocumentationCrawler(timeout_seconds=5, delay_seconds=0)
        html = '<a href="tel:+1234567890">phone</a>'
        links = crawler._extract_links(html, "http://example.com/")
        assert len(links) == 0

    def test_extract_links_skips_javascript(self):
        from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler

        crawler = DocumentationCrawler(timeout_seconds=5, delay_seconds=0)
        html = '<a href="javascript:void(0)">click</a>'
        links = crawler._extract_links(html, "http://example.com/")
        assert len(links) == 0

    def test_dedupe_key_strips_index_html(self):
        from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler

        crawler = DocumentationCrawler(timeout_seconds=5, delay_seconds=0)
        key = crawler._dedupe_key("http://example.com/dir/index.html")
        assert key == "http://example.com/dir"

    def test_dedupe_key_root_path(self):
        from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler

        crawler = DocumentationCrawler(timeout_seconds=5, delay_seconds=0)
        key = crawler._dedupe_key("http://example.com/")
        assert key == "http://example.com/"

    def test_dedupe_key_regular_path(self):
        from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler

        crawler = DocumentationCrawler(timeout_seconds=5, delay_seconds=0)
        key = crawler._dedupe_key("http://example.com/page")
        assert key == "http://example.com/page"

    def test_is_allowed_valid(self):
        from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler
        from data_engineering_copilot.config.settings import DocumentationSource

        crawler = DocumentationCrawler(timeout_seconds=5, delay_seconds=0)
        source = DocumentationSource(
            name="test",
            start_urls=("http://example.com/",),
            allowed_domains=("example.com",),
            url_prefixes=("http://example.com/docs",),
        )
        assert crawler._is_allowed("http://example.com/docs/page", source) is True

    def test_is_allowed_wrong_domain(self):
        from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler
        from data_engineering_copilot.config.settings import DocumentationSource

        crawler = DocumentationCrawler(timeout_seconds=5, delay_seconds=0)
        source = DocumentationSource(
            name="test",
            start_urls=("http://example.com/",),
            allowed_domains=("example.com",),
        )
        assert crawler._is_allowed("http://other.com/page", source) is False

    def test_is_allowed_wrong_prefix(self):
        from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler
        from data_engineering_copilot.config.settings import DocumentationSource

        crawler = DocumentationCrawler(timeout_seconds=5, delay_seconds=0)
        source = DocumentationSource(
            name="test",
            start_urls=("http://example.com/",),
            allowed_domains=("example.com",),
            url_prefixes=("http://example.com/docs",),
        )
        assert crawler._is_allowed("http://example.com/other/page", source) is False

    def test_is_allowed_non_http(self):
        from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler
        from data_engineering_copilot.config.settings import DocumentationSource

        crawler = DocumentationCrawler(timeout_seconds=5, delay_seconds=0)
        source = DocumentationSource(
            name="test",
            start_urls=("http://example.com/",),
            allowed_domains=("example.com",),
        )
        assert crawler._is_allowed("ftp://example.com/page", source) is False

    def test_emit_calls_callback(self):
        from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler

        crawler = DocumentationCrawler(timeout_seconds=5, delay_seconds=0)
        callback = MagicMock()
        event = IngestionEvent(event_type="test", source_name="s", message="m")
        crawler._emit(callback, event)
        callback.assert_called_once_with(event)

    def test_emit_no_callback(self):
        from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler

        crawler = DocumentationCrawler(timeout_seconds=5, delay_seconds=0)
        crawler._emit(None, IngestionEvent(event_type="test", source_name="s", message="m"))

    def test_download_non_html(self):
        from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler

        crawler = DocumentationCrawler(timeout_seconds=5, delay_seconds=0)
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("data_engineering_copilot.infrastructure.crawler.urlopen", return_value=mock_resp):
            with pytest.raises(ValueError, match="unsupported content type"):
                crawler._download("http://example.com/test.json")


# =============================================================================
# infrastructure/ollama_client.py (87% → 100%)
# =============================================================================
class TestOllamaClientFull:
    def test_timeout_error(self):
        from data_engineering_copilot.infrastructure.ollama_client import OllamaClient, OllamaError

        client = OllamaClient("http://localhost:11434", "test", 10, 2048, 128)
        with patch("data_engineering_copilot.infrastructure.ollama_client.urlopen", side_effect=TimeoutError("timeout")):
            with pytest.raises(OllamaError, match="timed out"):
                client.generate("test prompt")

    def test_socket_timeout(self):
        from data_engineering_copilot.infrastructure.ollama_client import OllamaClient, OllamaError

        client = OllamaClient("http://localhost:11434", "test", 10, 2048, 128)
        with patch("data_engineering_copilot.infrastructure.ollama_client.urlopen", side_effect=socket.timeout("timeout")):
            with pytest.raises(OllamaError, match="timed out"):
                client.generate("test prompt")

    def test_url_error(self):
        from data_engineering_copilot.infrastructure.ollama_client import OllamaClient, OllamaError

        client = OllamaClient("http://localhost:11434", "test", 10, 2048, 128)
        with patch("data_engineering_copilot.infrastructure.ollama_client.urlopen", side_effect=URLError("refused")):
            with pytest.raises(OllamaError, match="Could not reach Ollama"):
                client.generate("test prompt")

    def test_empty_response(self):
        from data_engineering_copilot.infrastructure.ollama_client import OllamaClient, OllamaError

        client = OllamaClient("http://localhost:11434", "test", 10, 2048, 128)
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "", "done_reason": "stop"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("data_engineering_copilot.infrastructure.ollama_client.urlopen", return_value=mock_resp):
            with pytest.raises(OllamaError, match="no final answer"):
                client.generate("test prompt")

    def test_think_only_response(self):
        from data_engineering_copilot.infrastructure.ollama_client import OllamaClient, OllamaError

        client = OllamaClient("http://localhost:11434", "test", 10, 2048, 128)
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "<think>reasoning</think>", "done_reason": "stop"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("data_engineering_copilot.infrastructure.ollama_client.urlopen", return_value=mock_resp):
            with pytest.raises(OllamaError, match="no final answer"):
                client.generate("test prompt")

    def test_extract_final_response_with_think(self):
        from data_engineering_copilot.infrastructure.ollama_client import OllamaClient

        client = OllamaClient("http://localhost:11434", "test", 10, 2048, 128)
        result = client._extract_final_response("<think>reasoning</think>The answer")
        assert result == "The answer"

    def test_extract_final_response_empty(self):
        from data_engineering_copilot.infrastructure.ollama_client import OllamaClient

        client = OllamaClient("http://localhost:11434", "test", 10, 2048, 128)
        assert client._extract_final_response("") == ""
        assert client._extract_final_response("  ") == ""

    def test_extract_final_response_think_only(self):
        from data_engineering_copilot.infrastructure.ollama_client import OllamaClient

        client = OllamaClient("http://localhost:11434", "test", 10, 2048, 128)
        result = client._extract_final_response("<think>reasoning</think>")
        assert result == ""

    def test_format_raw_chat_prompt(self):
        from data_engineering_copilot.infrastructure.ollama_client import OllamaClient

        client = OllamaClient("http://localhost:11434", "test", 10, 2048, 128)
        prompt = client._format_raw_chat_prompt("What is Spark?")
        assert "SYSTEM" in prompt
        assert "What is Spark?" in prompt

    def test_generate_custom_params(self):
        from data_engineering_copilot.infrastructure.ollama_client import OllamaClient

        client = OllamaClient("http://localhost:11434", "test", 10, 2048, 128)
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "answer", "done_reason": "stop"}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("data_engineering_copilot.infrastructure.ollama_client.urlopen", return_value=mock_resp) as mock_open:
            result = client.generate("test", num_predict=64, num_ctx=1024)
            assert result == "answer"


# =============================================================================
# infrastructure/html_parser.py (81% → 100%)
# =============================================================================
class TestHtmlParserFull:
    def test_short_page_returns_none(self):
        from data_engineering_copilot.infrastructure.html_parser import DocumentationHtmlParser

        parser = DocumentationHtmlParser()
        raw = RawDocument(source_name="test", url="http://example.com", html="<html><body><p>short</p></body></html>")
        result = parser.parse(raw)
        assert result is None

    def test_title_from_h1(self):
        from data_engineering_copilot.infrastructure.html_parser import DocumentationHtmlParser

        parser = DocumentationHtmlParser()
        html = "<html><body><h1>My Title</h1>" + " ".join(["word"] * 50) + "</body></html>"
        raw = RawDocument(source_name="test", url="http://example.com", html=html)
        result = parser.parse(raw)
        assert result is not None
        assert result.title == "My Title"

    def test_title_from_tag(self):
        from data_engineering_copilot.infrastructure.html_parser import DocumentationHtmlParser

        parser = DocumentationHtmlParser()
        html = "<html><head><title>Page Title</title></head><body>" + " ".join(["word"] * 50) + "</body></html>"
        raw = RawDocument(source_name="test", url="http://example.com", html=html)
        result = parser.parse(raw)
        assert result is not None
        assert result.title == "Page Title"

    def test_title_fallback_to_url(self):
        from data_engineering_copilot.infrastructure.html_parser import DocumentationHtmlParser

        parser = DocumentationHtmlParser()
        html = "<html><body>" + " ".join(["word"] * 50) + "</body></html>"
        raw = RawDocument(source_name="test", url="http://example.com/page", html=html)
        result = parser.parse(raw)
        assert result is not None
        assert result.title == "http://example.com/page"

    def test_strips_nav_footer(self):
        from data_engineering_copilot.infrastructure.html_parser import DocumentationHtmlParser

        parser = DocumentationHtmlParser()
        html = "<html><body><nav>nav</nav><footer>footer</footer>" + " ".join(["word"] * 50) + "</body></html>"
        raw = RawDocument(source_name="test", url="http://example.com", html=html)
        result = parser.parse(raw)
        assert result is not None
        assert "nav" not in result.text.lower()


# =============================================================================
# services/rag.py (56% → 100%)
# =============================================================================
class TestRagFull:
    def _make_chunks(self, n=3):
        return [
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id=f"c{i}", source_name="src", title="t", url="u", text=f"text {i}"),
                distance=0.1,
                confidence=0.9 - i * 0.1,
            )
            for i in range(n)
        ]

    def test_answer_embed_failure(self):
        from data_engineering_copilot.services.rag import RagAnswerService

        mock_vs = MagicMock()
        mock_ollama = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_query.side_effect = Exception("embed error")

        service = RagAnswerService(mock_vs, mock_ollama, mock_embedder)
        answer = service.answer("test?")
        assert "embedding failed" in answer.text
        assert answer.confidence == 0.0

    def test_answer_vector_store_failure(self):
        from data_engineering_copilot.services.rag import RagAnswerService

        mock_vs = MagicMock()
        mock_vs.query.side_effect = Exception("qdrant error")
        mock_ollama = MagicMock()
        mock_embedder = MagicMock()

        service = RagAnswerService(mock_vs, mock_ollama, mock_embedder)
        answer = service.answer("test?")
        assert "vector store query failed" in answer.text
        assert answer.confidence == 0.0

    def test_answer_no_chunks(self):
        from data_engineering_copilot.services.rag import RagAnswerService

        mock_vs = MagicMock()
        mock_vs.query.return_value = []
        mock_ollama = MagicMock()
        mock_embedder = MagicMock()

        service = RagAnswerService(mock_vs, mock_ollama, mock_embedder)
        answer = service.answer("test?")
        assert "outside my knowledge repository" in answer.text

    def test_answer_low_confidence(self):
        from data_engineering_copilot.services.rag import RagAnswerService

        mock_vs = MagicMock()
        chunks = self._make_chunks(1)
        chunks[0] = RetrievedChunk(chunk=chunks[0].chunk, distance=0.9, confidence=0.05)
        mock_vs.query.return_value = chunks
        mock_ollama = MagicMock()
        mock_embedder = MagicMock()

        service = RagAnswerService(mock_vs, mock_ollama, mock_embedder)
        answer = service.answer("test?")
        assert "outside my knowledge repository" in answer.text

    def test_answer_with_langfuse_tracing(self):
        from data_engineering_copilot.services.rag import RagAnswerService

        mock_vs = MagicMock()
        mock_vs.query.return_value = self._make_chunks(3)
        mock_ollama = MagicMock()
        mock_ollama.generate.return_value = "The answer"
        mock_embedder = MagicMock()

        mock_langfuse = MagicMock()
        mock_trace = MagicMock()
        mock_trace.start_observation.return_value = MagicMock()
        mock_langfuse.start_observation.return_value = mock_trace

        service = RagAnswerService(mock_vs, mock_ollama, mock_embedder)
        service.langfuse = mock_langfuse

        answer = service.answer("test?")
        assert answer.text == "The answer"
        mock_langfuse.flush.assert_called_once()

    def test_answer_generation_failure(self):
        from data_engineering_copilot.services.rag import RagAnswerService

        mock_vs = MagicMock()
        mock_vs.query.return_value = self._make_chunks(3)
        mock_ollama = MagicMock()
        mock_ollama.generate.side_effect = Exception("ollama error")
        mock_embedder = MagicMock()

        service = RagAnswerService(mock_vs, mock_ollama, mock_embedder)
        answer = service.answer("test?")
        assert "error while generating" in answer.text.lower()

    def test_answer_reranker_applied(self):
        from data_engineering_copilot.services.rag import RagAnswerService

        mock_vs = MagicMock()
        mock_vs.query.return_value = self._make_chunks(10)
        mock_ollama = MagicMock()
        mock_ollama.generate.return_value = "Reranked answer"
        mock_embedder = MagicMock()

        with patch("data_engineering_copilot.services.rag.settings") as mock_settings:
            mock_settings.retrieval_top_k = 5
            mock_settings.reranker_enabled = True
            mock_settings.reranker_model = "test-model"
            mock_settings.reranker_top_k = 3
            mock_settings.max_context_chars = 4000
            mock_settings.confidence_threshold = 0.18

            with patch("data_engineering_copilot.services.rag.CrossEncoderReranker") as mock_reranker_cls:
                mock_reranker = MagicMock()
                mock_reranker.is_available.return_value = True
                mock_reranker.rerank.return_value = self._make_chunks(3)
                mock_reranker_cls.return_value = mock_reranker

                service = RagAnswerService(mock_vs, mock_ollama, mock_embedder)
                answer = service.answer("test?")
                assert answer.text == "Reranked answer"
                mock_reranker.rerank.assert_called_once()

    def test_answer_reranker_not_available(self):
        from data_engineering_copilot.services.rag import RagAnswerService

        mock_vs = MagicMock()
        mock_vs.query.return_value = self._make_chunks(3)
        mock_ollama = MagicMock()
        mock_ollama.generate.return_value = "Non-reranked answer"
        mock_embedder = MagicMock()

        with patch("data_engineering_copilot.services.rag.settings") as mock_settings:
            mock_settings.retrieval_top_k = 5
            mock_settings.reranker_enabled = True
            mock_settings.reranker_model = "test-model"
            mock_settings.reranker_top_k = 3
            mock_settings.max_context_chars = 4000
            mock_settings.confidence_threshold = 0.18

            with patch("data_engineering_copilot.services.rag.CrossEncoderReranker") as mock_reranker_cls:
                mock_reranker = MagicMock()
                mock_reranker.is_available.return_value = False
                mock_reranker_cls.return_value = mock_reranker

                service = RagAnswerService(mock_vs, mock_ollama, mock_embedder)
                answer = service.answer("test?")
                assert answer.text == "Non-reranked answer"

    def test_answer_reranker_disabled(self):
        from data_engineering_copilot.services.rag import RagAnswerService

        mock_vs = MagicMock()
        mock_vs.query.return_value = self._make_chunks(3)
        mock_ollama = MagicMock()
        mock_ollama.generate.return_value = "Direct answer"
        mock_embedder = MagicMock()

        with patch("data_engineering_copilot.services.rag.settings") as mock_settings:
            mock_settings.retrieval_top_k = 5
            mock_settings.reranker_enabled = False
            mock_settings.max_context_chars = 4000
            mock_settings.confidence_threshold = 0.18

            service = RagAnswerService(mock_vs, mock_ollama, mock_embedder)
            answer = service.answer("test?")
            assert answer.text == "Direct answer"

    def test_answer_with_langfuse_trace_error(self):
        from data_engineering_copilot.services.rag import RagAnswerService

        mock_vs = MagicMock()
        mock_vs.query.side_effect = Exception("qdrant error")
        mock_ollama = MagicMock()
        mock_embedder = MagicMock()

        mock_langfuse = MagicMock()
        mock_trace = MagicMock()
        mock_retrieval_span = MagicMock()
        mock_trace.start_observation.return_value = mock_retrieval_span
        mock_langfuse.start_observation.return_value = mock_trace

        service = RagAnswerService(mock_vs, mock_ollama, mock_embedder)
        service.langfuse = mock_langfuse
        answer = service.answer("test?")
        assert "vector store query failed" in answer.text
        mock_retrieval_span.update.assert_called()
        mock_retrieval_span.end.assert_called()
        mock_trace.update.assert_called()

    def test_answer_embed_failure_with_langfuse(self):
        from data_engineering_copilot.services.rag import RagAnswerService

        mock_vs = MagicMock()
        mock_ollama = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_query.side_effect = Exception("embed error")

        mock_langfuse = MagicMock()
        mock_trace = MagicMock()
        mock_retrieval_span = MagicMock()
        mock_trace.start_observation.return_value = mock_retrieval_span
        mock_langfuse.start_observation.return_value = mock_trace

        service = RagAnswerService(mock_vs, mock_ollama, mock_embedder)
        service.langfuse = mock_langfuse
        answer = service.answer("test?")
        assert "embedding failed" in answer.text
        mock_retrieval_span.update.assert_called()
        mock_retrieval_span.end.assert_called()
        mock_trace.update.assert_called()


# =============================================================================
# factory.py (94% → 100%)
# =============================================================================
class TestFactoryFull:
    def test_build_ingestion_service(self):
        from data_engineering_copilot.factory import build_ingestion_service
        from data_engineering_copilot.config.settings import AppSettings

        with patch("data_engineering_copilot.factory.QdrantVectorStore"), \
             patch("data_engineering_copilot.factory.SentenceTransformerEmbeddings"):
            settings = AppSettings()
            service = build_ingestion_service(settings)
            assert service is not None


# =============================================================================
# config/settings.py (93% → 100%)
# =============================================================================
class TestSettingsFull:
    def test_load_empty_list(self, tmp_path):
        from data_engineering_copilot.config.settings import load_documentation_sources

        config_file = tmp_path / "sources.json"
        config_file.write_text("[]")
        with pytest.raises(ValueError, match="non-empty list"):
            load_documentation_sources(config_file)

    def test_load_non_list(self, tmp_path):
        from data_engineering_copilot.config.settings import load_documentation_sources

        config_file = tmp_path / "sources.json"
        config_file.write_text('"not a list"')
        with pytest.raises(ValueError, match="non-empty list"):
            load_documentation_sources(config_file)

    def test_load_non_dict_item(self, tmp_path):
        from data_engineering_copilot.config.settings import load_documentation_sources

        config_file = tmp_path / "sources.json"
        config_file.write_text('["not a dict"]')
        with pytest.raises(ValueError, match="must be an object"):
            load_documentation_sources(config_file)

    def test_load_missing_name(self, tmp_path):
        from data_engineering_copilot.config.settings import load_documentation_sources

        config_file = tmp_path / "sources.json"
        config_file.write_text('[{"start_urls": ["http://example.com"], "allowed_domains": ["example.com"]}]')
        with pytest.raises(ValueError, match="non-empty `name`"):
            load_documentation_sources(config_file)

    def test_load_missing_start_urls(self, tmp_path):
        from data_engineering_copilot.config.settings import load_documentation_sources

        config_file = tmp_path / "sources.json"
        config_file.write_text('[{"name": "test", "allowed_domains": ["example.com"]}]')
        with pytest.raises(ValueError, match="at least one `start_urls`"):
            load_documentation_sources(config_file)

    def test_load_missing_allowed_domains(self, tmp_path):
        from data_engineering_copilot.config.settings import load_documentation_sources

        config_file = tmp_path / "sources.json"
        config_file.write_text('[{"name": "test", "start_urls": ["http://example.com"]}]')
        with pytest.raises(ValueError, match="at least one `allowed_domains`"):
            load_documentation_sources(config_file)

    def test_load_url_prefixes_not_list(self, tmp_path):
        from data_engineering_copilot.config.settings import load_documentation_sources

        config_file = tmp_path / "sources.json"
        config_file.write_text('[{"name": "test", "start_urls": ["http://example.com"], "allowed_domains": ["example.com"], "url_prefixes": "not a list"}]')
        with pytest.raises(ValueError, match="must be a list"):
            load_documentation_sources(config_file)

    def test_load_url_prefixes_non_string_items(self, tmp_path):
        from data_engineering_copilot.config.settings import load_documentation_sources

        config_file = tmp_path / "sources.json"
        config_file.write_text('[{"name": "test", "start_urls": ["http://example.com"], "allowed_domains": ["example.com"], "url_prefixes": [123]}]')
        with pytest.raises(ValueError, match="non-empty strings"):
            load_documentation_sources(config_file)

    def test_load_start_urls_not_list(self, tmp_path):
        from data_engineering_copilot.config.settings import load_documentation_sources

        config_file = tmp_path / "sources.json"
        config_file.write_text('[{"name": "test", "start_urls": "not a list", "allowed_domains": ["example.com"]}]')
        with pytest.raises(ValueError, match="must be a list"):
            load_documentation_sources(config_file)

    def test_load_allowed_domains_not_list(self, tmp_path):
        from data_engineering_copilot.config.settings import load_documentation_sources

        config_file = tmp_path / "sources.json"
        config_file.write_text('[{"name": "test", "start_urls": ["http://example.com"], "allowed_domains": "not a list"}]')
        with pytest.raises(ValueError, match="must be a list"):
            load_documentation_sources(config_file)


# =============================================================================
# services/ingestion.py (91% → 100%)
# =============================================================================
class TestIngestionFull:
    def test_selected_sources_none(self):
        from data_engineering_copilot.services.ingestion import IngestionService
        from data_engineering_copilot.config.settings import AppSettings

        settings = AppSettings()
        service = IngestionService(
            settings=settings,
            crawler=MagicMock(),
            parser=MagicMock(),
            chunker=MagicMock(),
            embeddings=MagicMock(),
            vector_store=MagicMock(),
        )
        result = service._selected_sources(None)
        assert result == settings.sources

    def test_selected_sources_empty_string(self):
        from data_engineering_copilot.services.ingestion import IngestionService
        from data_engineering_copilot.config.settings import AppSettings

        settings = AppSettings()
        service = IngestionService(
            settings=settings,
            crawler=MagicMock(),
            parser=MagicMock(),
            chunker=MagicMock(),
            embeddings=MagicMock(),
            vector_store=MagicMock(),
        )
        with pytest.raises(ValueError, match="At least one"):
            service._selected_sources(["  "])

    def test_selected_sources_unknown(self):
        from data_engineering_copilot.services.ingestion import IngestionService
        from data_engineering_copilot.config.settings import AppSettings

        settings = AppSettings()
        service = IngestionService(
            settings=settings,
            crawler=MagicMock(),
            parser=MagicMock(),
            chunker=MagicMock(),
            embeddings=MagicMock(),
            vector_store=MagicMock(),
        )
        with pytest.raises(ValueError, match="Unknown documentation source"):
            service._selected_sources(["Nonexistent Source"])

    def test_compute_content_hash(self):
        from data_engineering_copilot.services.ingestion import IngestionService

        h = IngestionService._compute_content_hash("hello world")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_get_stored_content_hash_none(self):
        from data_engineering_copilot.services.ingestion import IngestionService
        from data_engineering_copilot.config.settings import AppSettings

        settings = AppSettings()
        vs = MagicMock(spec=[])  # no get_content_hash_for_url
        service = IngestionService(
            settings=settings,
            crawler=MagicMock(),
            parser=MagicMock(),
            chunker=MagicMock(),
            embeddings=MagicMock(),
            vector_store=vs,
        )
        result = service._get_stored_content_hash("http://example.com")
        assert result is None

    def test_get_stored_content_hash_found(self):
        from data_engineering_copilot.services.ingestion import IngestionService
        from data_engineering_copilot.config.settings import AppSettings

        settings = AppSettings()
        vs = MagicMock()
        vs.get_content_hash_for_url.return_value = "abc123"
        service = IngestionService(
            settings=settings,
            crawler=MagicMock(),
            parser=MagicMock(),
            chunker=MagicMock(),
            embeddings=MagicMock(),
            vector_store=vs,
        )
        result = service._get_stored_content_hash("http://example.com")
        assert result == "abc123"

    def test_delete_chunks_for_url_no_method(self):
        from data_engineering_copilot.services.ingestion import IngestionService
        from data_engineering_copilot.config.settings import AppSettings

        settings = AppSettings()
        vs = MagicMock(spec=[])  # no delete_by_url
        service = IngestionService(
            settings=settings,
            crawler=MagicMock(),
            parser=MagicMock(),
            chunker=MagicMock(),
            embeddings=MagicMock(),
            vector_store=vs,
        )
        service._delete_chunks_for_url("http://example.com")  # should not raise

    def test_delete_chunks_for_url_with_method(self):
        from data_engineering_copilot.services.ingestion import IngestionService
        from data_engineering_copilot.config.settings import AppSettings

        settings = AppSettings()
        vs = MagicMock()
        service = IngestionService(
            settings=settings,
            crawler=MagicMock(),
            parser=MagicMock(),
            chunker=MagicMock(),
            embeddings=MagicMock(),
            vector_store=vs,
        )
        service._delete_chunks_for_url("http://example.com")
        vs.delete_by_url.assert_called_once_with("http://example.com")

    def test_emit_calls_callback(self):
        from data_engineering_copilot.services.ingestion import IngestionService
        from data_engineering_copilot.config.settings import AppSettings
        from data_engineering_copilot.domain.models import IngestionEvent

        settings = AppSettings()
        service = IngestionService(
            settings=settings,
            crawler=MagicMock(),
            parser=MagicMock(),
            chunker=MagicMock(),
            embeddings=MagicMock(),
            vector_store=MagicMock(),
        )
        callback = MagicMock()
        event = IngestionEvent(event_type="test", source_name="s", message="m")
        service._emit(callback, event)
        callback.assert_called_once_with(event)

    def test_emit_no_callback(self):
        from data_engineering_copilot.services.ingestion import IngestionService
        from data_engineering_copilot.config.settings import AppSettings
        from data_engineering_copilot.domain.models import IngestionEvent

        settings = AppSettings()
        service = IngestionService(
            settings=settings,
            crawler=MagicMock(),
            parser=MagicMock(),
            chunker=MagicMock(),
            embeddings=MagicMock(),
            vector_store=MagicMock(),
        )
        service._emit(None, IngestionEvent(event_type="test", source_name="s", message="m"))

    def test_ingest_flush_batch_runtime_error(self):
        from data_engineering_copilot.services.ingestion import IngestionService
        from data_engineering_copilot.config.settings import AppSettings, DocumentationSource
        from data_engineering_copilot.domain.models import ParsedDocument

        parsed = ParsedDocument(source_name="s", title="t", url="http://example.com", text="word " * 50)
        mock_parser = MagicMock()
        mock_parser.parse.return_value = parsed
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [
            DocumentChunk(chunk_id="c1", source_name="s", title="t", url="http://example.com", text="text")
        ]
        mock_embeddings = MagicMock()
        mock_embeddings.embed_texts.side_effect = RuntimeError("embed failed")
        mock_vector_store = MagicMock()

        source = DocumentationSource(
            name="test",
            start_urls=("http://example.com",),
            allowed_domains=("example.com",),
        )

        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = iter([
            RawDocument(source_name="test", url="http://example.com", html="<html><body>test</body></html>")
        ])

        settings_with_source = AppSettings.__new__(AppSettings)
        object.__setattr__(settings_with_source, 'sources', (source,))
        object.__setattr__(settings_with_source, 'max_pages_per_source', 1)
        object.__setattr__(settings_with_source, 'ingestion_batch_chunk_size', 1)

        service = IngestionService(
            settings=settings_with_source,
            crawler=mock_crawler,
            parser=mock_parser,
            chunker=mock_chunker,
            embeddings=mock_embeddings,
            vector_store=mock_vector_store,
        )
        with pytest.raises(RuntimeError, match="embed failed"):
            service.ingest(max_pages_per_source=1, source_names=["test"])

    def test_ingest_upsert_runtime_error(self):
        from data_engineering_copilot.services.ingestion import IngestionService
        from data_engineering_copilot.config.settings import AppSettings, DocumentationSource
        from data_engineering_copilot.domain.models import ParsedDocument

        parsed = ParsedDocument(source_name="s", title="t", url="http://example.com", text="word " * 50)
        mock_parser = MagicMock()
        mock_parser.parse.return_value = parsed
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [
            DocumentChunk(chunk_id="c1", source_name="s", title="t", url="http://example.com", text="text")
        ]
        mock_embeddings = MagicMock()
        mock_embeddings.embed_texts.return_value = [[0.1] * 768]
        mock_vector_store = MagicMock()
        mock_vector_store.upsert_chunks.side_effect = RuntimeError("upsert failed")

        source = DocumentationSource(
            name="test",
            start_urls=("http://example.com",),
            allowed_domains=("example.com",),
        )

        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = iter([
            RawDocument(source_name="test", url="http://example.com", html="<html><body>test</body></html>")
        ])

        settings_with_source = AppSettings.__new__(AppSettings)
        object.__setattr__(settings_with_source, 'sources', (source,))
        object.__setattr__(settings_with_source, 'max_pages_per_source', 1)
        object.__setattr__(settings_with_source, 'ingestion_batch_chunk_size', 1)

        service = IngestionService(
            settings=settings_with_source,
            crawler=mock_crawler,
            parser=mock_parser,
            chunker=mock_chunker,
            embeddings=mock_embeddings,
            vector_store=mock_vector_store,
        )
        with pytest.raises(RuntimeError, match="upsert failed"):
            service.ingest(max_pages_per_source=1, source_names=["test"])

    def test_ingest_content_hash_changed_deletes_old(self):
        from data_engineering_copilot.services.ingestion import IngestionService
        from data_engineering_copilot.config.settings import AppSettings, DocumentationSource
        from data_engineering_copilot.domain.models import ParsedDocument

        parsed = ParsedDocument(source_name="s", title="t", url="http://example.com", text="word " * 50)
        mock_parser = MagicMock()
        mock_parser.parse.return_value = parsed
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [
            DocumentChunk(chunk_id="c1", source_name="s", title="t", url="http://example.com", text="text")
        ]
        mock_embeddings = MagicMock()
        mock_embeddings.embed_texts.return_value = [[0.1] * 768]
        mock_vector_store = MagicMock()
        mock_vector_store.get_content_hash_for_url.return_value = "old_hash"

        source = DocumentationSource(
            name="test",
            start_urls=("http://example.com",),
            allowed_domains=("example.com",),
        )

        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = iter([
            RawDocument(source_name="test", url="http://example.com", html="<html><body>test</body></html>")
        ])

        settings_with_source = AppSettings.__new__(AppSettings)
        object.__setattr__(settings_with_source, 'sources', (source,))
        object.__setattr__(settings_with_source, 'max_pages_per_source', 1)
        object.__setattr__(settings_with_source, 'ingestion_batch_chunk_size', 100)

        service = IngestionService(
            settings=settings_with_source,
            crawler=mock_crawler,
            parser=mock_parser,
            chunker=mock_chunker,
            embeddings=mock_embeddings,
            vector_store=mock_vector_store,
        )
        service.ingest(max_pages_per_source=1, source_names=["test"])
        mock_vector_store.delete_by_url.assert_called_once_with("http://example.com")

    def test_ingest_page_skipped(self):
        from data_engineering_copilot.services.ingestion import IngestionService
        from data_engineering_copilot.config.settings import AppSettings, DocumentationSource

        mock_parser = MagicMock()
        mock_parser.parse.return_value = None  # skip page
        mock_chunker = MagicMock()
        mock_embeddings = MagicMock()
        mock_vector_store = MagicMock()

        source = DocumentationSource(
            name="test",
            start_urls=("http://example.com",),
            allowed_domains=("example.com",),
        )

        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = iter([
            RawDocument(source_name="test", url="http://example.com", html="<html><body>test</body></html>")
        ])

        settings_with_source = AppSettings.__new__(AppSettings)
        object.__setattr__(settings_with_source, 'sources', (source,))
        object.__setattr__(settings_with_source, 'max_pages_per_source', 1)
        object.__setattr__(settings_with_source, 'ingestion_batch_chunk_size', 100)

        service = IngestionService(
            settings=settings_with_source,
            crawler=mock_crawler,
            parser=mock_parser,
            chunker=mock_chunker,
            embeddings=mock_embeddings,
            vector_store=mock_vector_store,
        )
        events = []
        result = service.ingest(max_pages_per_source=1, source_names=["test"], on_event=events.append)
        assert result == 0
        assert any(e.event_type == "page_skipped" for e in events)

    def test_ingest_content_unchanged_skips(self):
        from data_engineering_copilot.services.ingestion import IngestionService
        from data_engineering_copilot.config.settings import AppSettings, DocumentationSource
        from data_engineering_copilot.domain.models import ParsedDocument
        import hashlib

        parsed = ParsedDocument(source_name="s", title="t", url="http://example.com", text="word " * 50)
        content_hash = hashlib.sha256(parsed.text.encode()).hexdigest()

        mock_parser = MagicMock()
        mock_parser.parse.return_value = parsed
        mock_chunker = MagicMock()
        mock_embeddings = MagicMock()
        mock_vector_store = MagicMock()
        mock_vector_store.get_content_hash_for_url.return_value = content_hash

        source = DocumentationSource(
            name="test",
            start_urls=("http://example.com",),
            allowed_domains=("example.com",),
        )

        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = iter([
            RawDocument(source_name="test", url="http://example.com", html="<html><body>test</body></html>")
        ])

        settings_with_source = AppSettings.__new__(AppSettings)
        object.__setattr__(settings_with_source, 'sources', (source,))
        object.__setattr__(settings_with_source, 'max_pages_per_source', 1)
        object.__setattr__(settings_with_source, 'ingestion_batch_chunk_size', 100)

        service = IngestionService(
            settings=settings_with_source,
            crawler=mock_crawler,
            parser=mock_parser,
            chunker=mock_chunker,
            embeddings=mock_embeddings,
            vector_store=mock_vector_store,
        )
        events = []
        result = service.ingest(max_pages_per_source=1, source_names=["test"], on_event=events.append)
        assert result == 0
        assert any(e.event_type == "page_skipped_duplicate" for e in events)


# =============================================================================
# services/semantic_chunker.py - additional coverage
# =============================================================================
class TestSemanticChunkerFull:
    def test_min_chunk_words_validation(self):
        from data_engineering_copilot.services.semantic_chunker import SemanticChunker

        with pytest.raises(ValueError, match="min_chunk_words must be non-negative"):
            SemanticChunker(
                chunk_size_words=100,
                overlap_words=10,
                embedding_model=MagicMock(),
                min_chunk_words=-1,
            )

    def test_tokenization_failure(self):
        from data_engineering_copilot.services.semantic_chunker import SemanticChunker

        mock_embedder = MagicMock()
        chunker = SemanticChunker(chunk_size_words=100, overlap_words=10, embedding_model=mock_embedder)

        doc = ParsedDocument(source_name="s", title="t", url="u", text="test text here")
        with patch("data_engineering_copilot.services.semantic_chunker.sent_tokenize", side_effect=Exception("tokenize error")):
            result = chunker.chunk(doc)
            assert result == []

    def test_embedding_failure(self):
        from data_engineering_copilot.services.semantic_chunker import SemanticChunker

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.side_effect = Exception("embed error")
        chunker = SemanticChunker(chunk_size_words=100, overlap_words=10, embedding_model=mock_embedder)

        doc = ParsedDocument(source_name="s", title="t", url="u", text="This is a test sentence. Another sentence here.")
        with patch("data_engineering_copilot.services.semantic_chunker.sent_tokenize", return_value=["This is a test sentence.", "Another sentence here."]):
            result = chunker.chunk(doc)
            assert result == []

    def test_embedding_count_mismatch(self):
        from data_engineering_copilot.services.semantic_chunker import SemanticChunker

        mock_embedder = MagicMock()
        mock_embedder.embed_texts.return_value = [[0.1] * 768]  # only 1 embedding for 2 sentences
        chunker = SemanticChunker(chunk_size_words=100, overlap_words=10, embedding_model=mock_embedder)

        doc = ParsedDocument(source_name="s", title="t", url="u", text="Sentence one. Sentence two.")
        with patch("data_engineering_copilot.services.semantic_chunker.sent_tokenize", return_value=["Sentence one.", "Sentence two."]):
            result = chunker.chunk(doc)
            assert result == []

    def test_empty_embeddings(self):
        from data_engineering_copilot.services.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(chunk_size_words=100, overlap_words=10, embedding_model=MagicMock())
        result = chunker._cluster_sentences([], [])
        assert result == []

    def test_valid_chunk_check(self):
        from data_engineering_copilot.services.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(chunk_size_words=100, overlap_words=10, embedding_model=MagicMock(), min_chunk_words=5)
        assert chunker._is_valid_chunk("") is False
        assert chunker._is_valid_chunk("short") is False
        assert chunker._is_valid_chunk("this is a valid chunk with enough words") is True
        assert chunker._is_valid_chunk("...,,,..,..,..,") is False

    def test_overlap_sentences_kept(self):
        from data_engineering_copilot.services.semantic_chunker import SemanticChunker

        mock_embedder = MagicMock()
        n_sentences = 20
        # Create embeddings that will be similar enough to cluster together
        embeddings = [[1.0, 0.0, 0.0]] * n_sentences
        mock_embedder.embed_texts.return_value = embeddings
        chunker = SemanticChunker(chunk_size_words=10, overlap_words=3, embedding_model=mock_embedder, min_semantic_similarity=0.5, min_chunk_words=1)

        text = ". ".join([f"This is sentence number {i} with some words to fill the space" for i in range(n_sentences)])
        doc = ParsedDocument(source_name="s", title="t", url="http://example.com/doc", text=text)

        chunks = chunker.chunk(doc)
        assert len(chunks) > 0


# =============================================================================
# services/chunker.py - additional coverage
# =============================================================================
class TestChunkerFull:
    def test_tokenization_fallback_to_fixed_size(self):
        from data_engineering_copilot.services.chunker import DocumentChunker, ChunkingStrategy

        chunker = DocumentChunker(chunk_size_words=50, overlap_words=10, strategy=ChunkingStrategy.SENTENCE_PRESERVING)
        doc = ParsedDocument(source_name="s", title="t", url="u", text="word " * 100)
        with patch("data_engineering_copilot.services.chunker.sent_tokenize", side_effect=Exception("token error")):
            result = chunker.chunk(doc)
            assert len(result) > 0  # falls back to fixed_size

    def test_unsupported_strategy(self):
        from data_engineering_copilot.services.chunker import DocumentChunker

        chunker = DocumentChunker.__new__(DocumentChunker)
        chunker.chunk_size_words = 100
        chunker.overlap_words = 10
        chunker.min_chunk_words = 10
        chunker.strategy = "unsupported"
        doc = ParsedDocument(source_name="s", title="t", url="u", text="word " * 100)
        with pytest.raises(ValueError, match="Unsupported chunking strategy"):
            chunker.chunk(doc)

    def test_no_sentences(self):
        from data_engineering_copilot.services.chunker import DocumentChunker, ChunkingStrategy

        chunker = DocumentChunker(chunk_size_words=50, overlap_words=10, strategy=ChunkingStrategy.SENTENCE_PRESERVING)
        doc = ParsedDocument(source_name="s", title="t", url="u", text="")
        with patch("data_engineering_copilot.services.chunker.sent_tokenize", return_value=[]):
            result = chunker.chunk(doc)
            assert result == []

    def test_is_valid_chunk_no_alphanumeric(self):
        from data_engineering_copilot.services.chunker import DocumentChunker

        chunker = DocumentChunker(chunk_size_words=100, overlap_words=10, min_chunk_words=1)
        assert chunker._is_valid_chunk("") is False
        assert chunker._is_valid_chunk("...,,,..") is False

    def test_fixed_size_with_overlap(self):
        from data_engineering_copilot.services.chunker import DocumentChunker, ChunkingStrategy

        chunker = DocumentChunker(chunk_size_words=10, overlap_words=3, strategy=ChunkingStrategy.FIXED_SIZE, min_chunk_words=2)
        doc = ParsedDocument(source_name="s", title="t", url="u", text="word " * 30)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0


# =============================================================================
# infrastructure/crawler.py - additional coverage
# =============================================================================
class TestCrawlerDownloadFailure:
    def test_download_failure_emits_event(self):
        from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler
        from data_engineering_copilot.config.settings import DocumentationSource

        crawler = DocumentationCrawler(timeout_seconds=5, delay_seconds=0)
        source = DocumentationSource(
            name="test",
            start_urls=("http://example.com",),
            allowed_domains=("example.com",),
        )
        events = []
        with patch.object(crawler, "_download", side_effect=Exception("network error")):
            docs = list(crawler.crawl(source, max_pages=5, on_event=events.append))
            assert len(docs) == 0
            assert any(e.event_type == "fetch_error" for e in events)

    def test_download_non_html_skipped(self):
        from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler
        from data_engineering_copilot.config.settings import DocumentationSource

        crawler = DocumentationCrawler(timeout_seconds=5, delay_seconds=0)
        source = DocumentationSource(
            name="test",
            start_urls=("http://example.com",),
            allowed_domains=("example.com",),
        )
        events = []
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("data_engineering_copilot.infrastructure.crawler.urlopen", return_value=mock_resp):
            docs = list(crawler.crawl(source, max_pages=5, on_event=events.append))
            assert len(docs) == 0


# =============================================================================
# services/rag.py - ProductionRagService coverage
# =============================================================================
class TestRagProduction:
    def test_production_rag_service_init(self):
        from data_engineering_copilot.services.rag import ProductionRagService

        with patch("data_engineering_copilot.services.rag.QdrantVectorStore"), \
             patch("data_engineering_copilot.services.rag.SentenceTransformerEmbeddings"), \
             patch("data_engineering_copilot.services.rag.OllamaClient"), \
             patch("data_engineering_copilot.services.rag.get_langfuse_instance", return_value=None):
            service = ProductionRagService()
            assert service.langfuse is None

    def test_production_rag_service_init_with_langfuse(self):
        from data_engineering_copilot.services.rag import ProductionRagService

        mock_langfuse = MagicMock()
        with patch("data_engineering_copilot.services.rag.QdrantVectorStore"), \
             patch("data_engineering_copilot.services.rag.SentenceTransformerEmbeddings"), \
             patch("data_engineering_copilot.services.rag.OllamaClient"), \
             patch("data_engineering_copilot.services.rag.get_langfuse_instance", return_value=mock_langfuse):
            service = ProductionRagService()
            assert service.langfuse is mock_langfuse


# =============================================================================
# infrastructure/qdrant_store.py - additional coverage
# =============================================================================
class TestQdrantStoreAdditional:
    def test_hybrid_query_success(self):
        from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore

        with patch("data_engineering_copilot.infrastructure.qdrant_store.QdrantClient") as mock_cls:
            mock_client = Mock()
            mock_client.collection_exists.return_value = True
            mock_client.query_points.return_value = []
            mock_cls.return_value = mock_client

            store = QdrantVectorStore(url="http://localhost:6333", collection_name="test")
            with patch.object(store, "query", return_value=[]) as mock_query:
                with patch("data_engineering_copilot.infrastructure.qdrant_store.SentenceTransformerEmbeddings") as mock_emb_cls:
                    mock_emb = MagicMock()
                    mock_emb.embed_query.return_value = [0.1] * 768
                    mock_emb_cls.return_value = mock_emb
                    result = store.hybrid_query("test query", top_k=5)
                    assert result == []
                    mock_emb.embed_query.assert_called_once_with("test query")

    def test_hybrid_query_exception(self):
        from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore

        with patch("data_engineering_copilot.infrastructure.qdrant_store.QdrantClient") as mock_cls:
            mock_client = Mock()
            mock_client.collection_exists.return_value = True
            mock_cls.return_value = mock_client

            store = QdrantVectorStore(url="http://localhost:6333", collection_name="test")
            with patch("data_engineering_copilot.infrastructure.qdrant_store.SentenceTransformerEmbeddings", side_effect=Exception("embed error")):
                with pytest.raises(Exception, match="embed error"):
                    store.hybrid_query("test query", top_k=5)

    def test_count_exception(self):
        from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore

        with patch("data_engineering_copilot.infrastructure.qdrant_store.QdrantClient") as mock_cls:
            mock_client = Mock()
            mock_client.collection_exists.return_value = True
            mock_client.get_collection.side_effect = Exception("count error")
            mock_cls.return_value = mock_client

            store = QdrantVectorStore(url="http://localhost:6333", collection_name="test")
            with pytest.raises(Exception, match="count error"):
                store.count()

    def test_upsert_no_client(self):
        from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore.__new__(QdrantVectorStore)
        store._client = None
        store._collection_name = "test"
        store._url = "http://localhost:6333"
        store.upsert_chunks([], [])
        # Should return without error


# =============================================================================
# infrastructure/embeddings.py - additional coverage
# =============================================================================
class TestEmbeddingsAdditional:
    def test_embed_query_empty_result(self):
        from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
        from pathlib import Path

        embedder = SentenceTransformerEmbeddings(
            model_name="nomic-embed-text",
            cache_dir=Path("/tmp/cache"),
            local_files_only=True,
        )
        with patch.object(embedder, "embed_texts", return_value=[]):
            with pytest.raises(RuntimeError, match="empty result"):
                embedder.embed_query("test query")

    def test_embed_query_none_result(self):
        from data_engineering_copilot.infrastructure.embeddings import SentenceTransformerEmbeddings
        from pathlib import Path

        embedder = SentenceTransformerEmbeddings(
            model_name="nomic-embed-text",
            cache_dir=Path("/tmp/cache"),
            local_files_only=True,
        )
        with patch.object(embedder, "embed_texts", return_value=[None]):
            with pytest.raises(RuntimeError, match="empty result"):
                embedder.embed_query("test query")


# =============================================================================
# observability/langfuse_client.py - additional coverage
# =============================================================================
class TestLangfuseAdditional:
    def test_get_langfuse_instance_success(self):
        from data_engineering_copilot.observability.langfuse_client import get_langfuse_instance

        mock_lf_class = MagicMock()
        mock_lf = MagicMock()
        mock_lf.auth_check.return_value = True
        mock_lf_class.return_value = mock_lf

        mock_langfuse_mod = MagicMock(Langfuse=mock_lf_class)

        with patch("data_engineering_copilot.observability.langfuse_client._check_langfuse_health", return_value=True), \
             patch.dict("sys.modules", {"langfuse": mock_langfuse_mod}):
            result = get_langfuse_instance()
            assert result is not None

    def test_observation_compat_log_event_with_dict_kwargs(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        mock_obs = MagicMock()
        mock_obs.log_event = MagicMock()
        compat = _ObservationCompat(MagicMock(), mock_obs, "span")
        compat.log_event(name="test", key1="value1", key2="value2")
        mock_obs.log_event.assert_called_once_with(name="test", key1="value1", key2="value2")

    def test_observation_compat_start_observation_trace_id(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        mock_client = MagicMock()
        mock_client._client.start_observation = MagicMock(return_value=MagicMock(id="child"))
        # Test with trace_id set
        compat = _ObservationCompat(mock_client, MagicMock(id="parent"), "span", trace_id="trace-123")
        child = compat.start_observation(name="child", as_type="span")
        assert child is not None


# =============================================================================
# services/rag.py - remaining Langfuse tracing paths
# =============================================================================
class TestRagLangfusePaths:
    def _make_chunks(self, n=3):
        return [
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id=f"c{i}", source_name="src", title="t", url="u", text=f"text {i}"),
                distance=0.1,
                confidence=0.9 - i * 0.1,
            )
            for i in range(n)
        ]

    def test_no_chunks_with_langfuse_trace(self):
        from data_engineering_copilot.services.rag import RagAnswerService

        mock_vs = MagicMock()
        mock_vs.query.return_value = []
        mock_ollama = MagicMock()
        mock_embedder = MagicMock()

        mock_langfuse = MagicMock()
        mock_trace = MagicMock()
        mock_retrieval_span = MagicMock()
        mock_trace.start_observation.return_value = mock_retrieval_span
        mock_langfuse.start_observation.return_value = mock_trace

        service = RagAnswerService(mock_vs, mock_ollama, mock_embedder)
        service.langfuse = mock_langfuse
        answer = service.answer("test?")
        assert "outside my knowledge repository" in answer.text
        mock_trace.update.assert_called()

    def test_low_confidence_with_langfuse_trace(self):
        from data_engineering_copilot.services.rag import RagAnswerService

        mock_vs = MagicMock()
        chunks = [RetrievedChunk(
            chunk=DocumentChunk(chunk_id="1", source_name="src", title="t", url="u", text="text"),
            distance=0.9,
            confidence=0.05,
        )]
        mock_vs.query.return_value = chunks
        mock_ollama = MagicMock()
        mock_embedder = MagicMock()

        mock_langfuse = MagicMock()
        mock_trace = MagicMock()
        mock_retrieval_span = MagicMock()
        mock_trace.start_observation.return_value = mock_retrieval_span
        mock_langfuse.start_observation.return_value = mock_trace

        service = RagAnswerService(mock_vs, mock_ollama, mock_embedder)
        service.langfuse = mock_langfuse
        answer = service.answer("test?")
        assert "outside my knowledge repository" in answer.text
        mock_trace.update.assert_called()

    def test_generation_failure_with_langfuse(self):
        from data_engineering_copilot.services.rag import RagAnswerService

        mock_vs = MagicMock()
        mock_vs.query.return_value = self._make_chunks(3)
        mock_ollama = MagicMock()
        mock_ollama.generate.side_effect = Exception("ollama error")
        mock_embedder = MagicMock()

        mock_langfuse = MagicMock()
        mock_trace = MagicMock()
        mock_gen_span = MagicMock()
        mock_retrieval_span = MagicMock()
        mock_trace.start_observation.side_effect = [mock_retrieval_span, mock_gen_span]
        mock_langfuse.start_observation.return_value = mock_trace

        service = RagAnswerService(mock_vs, mock_ollama, mock_embedder)
        service.langfuse = mock_langfuse
        answer = service.answer("test?")
        assert "error while generating" in answer.text.lower()
        mock_gen_span.update.assert_called()
        mock_gen_span.end.assert_called()
        mock_trace.update.assert_called()


# =============================================================================
# services/context_assembler.py - truncation logging
# =============================================================================
class TestContextAssemblerAdditional:
    def test_truncation_at_chunk_boundary(self):
        from data_engineering_copilot.services.context_assembler import ContextAssembler

        assembler = ContextAssembler(max_context_chars=60)
        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id=f"{i}", source_name=f"src{i}", title="t", url="u", text=f"This is chunk number {i} with some content"),
                distance=0.1,
                confidence=0.9 - i * 0.05,
            )
            for i in range(10)
        ]
        ctx, sources = assembler.assemble(chunks)
        assert len(ctx) <= 70  # Allow slight slack for formatting
        # Should have at least 1 but not all
        assert len(sources) >= 1


# =============================================================================
# services/reranker.py - score improvement logging
# =============================================================================
class TestRerankerScoreImprovement:
    def test_rerank_score_improvement_logging(self):
        from data_engineering_copilot.services.reranker import CrossEncoderReranker

        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="0", source_name="s", title="t", url="u", text="text 0"),
                distance=0.9,
                confidence=0.1,  # Low original confidence
            ),
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="1", source_name="s", title="t", url="u", text="text 1"),
                distance=0.9,
                confidence=0.1,
            ),
        ]

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.95, 0.1]  # Big improvement for first

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model = mock_model
        reranker.model_name = "test"

        result = reranker.rerank("query", chunks, top_k=2)
        assert len(result) == 2
        assert result[0].chunk.chunk_id == "0"


# =============================================================================
# infrastructure/crawler.py - line 129
# =============================================================================
class TestCrawlerContentCheck:
    def test_download_content_type_check(self):
        from data_engineering_copilot.infrastructure.crawler import DocumentationCrawler

        crawler = DocumentationCrawler(timeout_seconds=5, delay_seconds=0)
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_resp.read.return_value = b"<html><body>test content</body></html>"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("data_engineering_copilot.infrastructure.crawler.urlopen", return_value=mock_resp):
            result = crawler._download("http://example.com")
            assert "test content" in result


# =============================================================================
# services/semantic_chunker.py - overlap and validation paths
# =============================================================================
class TestSemanticChunkerOverlap:
    def test_merge_empty_clusters(self):
        from data_engineering_copilot.services.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(chunk_size_words=100, overlap_words=10, embedding_model=MagicMock())
        doc = ParsedDocument(source_name="s", title="t", url="u", text="text")
        result = chunker._merge_clusters_into_chunks(doc, [])
        assert result == []

    def test_merge_with_overlap(self):
        from data_engineering_copilot.services.semantic_chunker import SemanticChunker

        mock_embedder = MagicMock()
        chunker = SemanticChunker(chunk_size_words=15, overlap_words=5, embedding_model=mock_embedder, min_chunk_words=1)

        sentences = [f"This is sentence number {i} with some extra words to fill it up." for i in range(8)]
        text = " ".join(sentences)
        doc = ParsedDocument(source_name="s", title="t", url="http://example.com/doc", text=text)

        # Simulate clusters: each sentence in its own cluster
        sentence_groups = [[i] for i in range(len(sentences))]
        chunks = chunker._merge_clusters_into_chunks(doc, sentence_groups)
        assert len(chunks) >= 1

    def test_max_chunk_words_hard_limit(self):
        from data_engineering_copilot.services.semantic_chunker import SemanticChunker

        mock_embedder = MagicMock()
        chunker = SemanticChunker(
            chunk_size_words=10,
            overlap_words=2,
            embedding_model=mock_embedder,
            min_chunk_words=1,
            max_chunk_words=15,
        )

        sentences = [f"This is sentence {i} with extra words to pad." for i in range(10)]
        text = " ".join(sentences)
        doc = ParsedDocument(source_name="s", title="t", url="http://example.com/doc", text=text)

        sentence_groups = [[i] for i in range(len(sentences))]
        chunks = chunker._merge_clusters_into_chunks(doc, sentence_groups)
        assert len(chunks) >= 1

    def test_is_valid_chunk_no_alphanumeric(self):
        from data_engineering_copilot.services.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(chunk_size_words=100, overlap_words=10, embedding_model=MagicMock(), min_chunk_words=1)
        assert chunker._is_valid_chunk("") is False
        assert chunker._is_valid_chunk("...,,,..") is False
        assert chunker._is_valid_chunk("valid text with words") is True


# =============================================================================
# infrastructure/ollama_client.py - socket timeout
# =============================================================================
class TestOllamaSocketTimeout:
    def test_socket_timeout_error(self):
        from data_engineering_copilot.infrastructure.ollama_client import OllamaClient, OllamaError

        client = OllamaClient("http://localhost:11434", "test", 10, 2048, 128)
        with patch("data_engineering_copilot.infrastructure.ollama_client.urlopen", side_effect=socket.timeout("connection timed out")):
            with pytest.raises(OllamaError, match="timed out"):
                client.generate("test prompt")


# =============================================================================
# Remaining edge cases for final coverage push
# =============================================================================
class TestRemainingCoverage:
    def test_langfuse_log_event_dict_payload_no_kwargs(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        mock_obs = MagicMock()
        mock_obs.log_event = MagicMock()
        compat = _ObservationCompat(MagicMock(), mock_obs, "span")
        compat.log_event(name="test", payload={"key": "value"})
        mock_obs.log_event.assert_called_once()
        call_kwargs = mock_obs.log_event.call_args[1]
        assert call_kwargs["key"] == "value"

    def test_langfuse_log_event_dict_payload_with_kwargs(self):
        from data_engineering_copilot.observability.langfuse_client import _ObservationCompat

        mock_obs = MagicMock()
        mock_obs.log_event = MagicMock()
        compat = _ObservationCompat(MagicMock(), mock_obs, "span")
        compat.log_event(name="test", payload="data", extra="arg")
        mock_obs.log_event.assert_called_once()
        call_kwargs = mock_obs.log_event.call_args[1]
        assert call_kwargs["input"] == "data"

    def test_langfuse_instance_init_exception(self):
        from data_engineering_copilot.observability.langfuse_client import get_langfuse_instance

        mock_lf_class = MagicMock()
        mock_lf_class.side_effect = Exception("init error")

        mock_langfuse_mod = MagicMock(Langfuse=mock_lf_class)

        with patch("data_engineering_copilot.observability.langfuse_client._check_langfuse_health", return_value=True), \
             patch.dict("sys.modules", {"langfuse": mock_langfuse_mod}):
            result = get_langfuse_instance()
            assert result is None

    def test_context_assembler_truncation_log(self):
        from data_engineering_copilot.services.context_assembler import ContextAssembler

        assembler = ContextAssembler(max_context_chars=50)
        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="0", source_name="src0", title="t", url="u", text="A" * 30),
                distance=0.1,
                confidence=0.9,
            ),
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="1", source_name="src1", title="t", url="u", text="B" * 30),
                distance=0.2,
                confidence=0.8,
            ),
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="2", source_name="src2", title="t", url="u", text="C" * 30),
                distance=0.3,
                confidence=0.7,
            ),
        ]
        ctx, sources = assembler.assemble(chunks)
        # Should have truncated
        assert len(ctx) <= 60
        assert len(sources) >= 1

    def test_reranker_score_improvement_log(self):
        from data_engineering_copilot.services.reranker import CrossEncoderReranker

        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="0", source_name="s", title="t", url="u", text="text zero"),
                distance=0.95,
                confidence=0.05,  # Very low original confidence
            ),
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="1", source_name="s", title="t", url="u", text="text one"),
                distance=0.95,
                confidence=0.05,
            ),
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="2", source_name="s", title="t", url="u", text="text two"),
                distance=0.95,
                confidence=0.05,
            ),
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="3", source_name="s", title="t", url="u", text="text three"),
                distance=0.95,
                confidence=0.05,
            ),
            RetrievedChunk(
                chunk=DocumentChunk(chunk_id="4", source_name="s", title="t", url="u", text="text four"),
                distance=0.95,
                confidence=0.05,
            ),
        ]

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.95, 0.1, 0.1, 0.1, 0.1]

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model = mock_model
        reranker.model_name = "test"

        result = reranker.rerank("query", chunks, top_k=3)
        assert len(result) == 3
        assert result[0].chunk.chunk_id == "0"
