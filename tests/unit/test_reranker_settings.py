"""Tests for reranker configuration fields in AppSettings and RagConfig."""

from __future__ import annotations

import pytest

from data_engineering_copilot.domain.models import RagConfig, RerankerType
from tests.conftest import make_settings

pytestmark = pytest.mark.unit


class TestRerankerType:
    def test_cross_encoder_value(self) -> None:
        assert RerankerType.CROSS_ENCODER == "cross_encoder"

    def test_colbert_value(self) -> None:
        assert RerankerType.COLBERT == "colbert"

    def test_is_str_subclass(self) -> None:
        assert isinstance(RerankerType.CROSS_ENCODER, str)


class TestAppSettingsRerankerDefaults:
    def test_reranker_type_default(self) -> None:
        s = make_settings()
        assert s.reranker_type == "cross_encoder"

    def test_reranker_pool_size_default(self) -> None:
        s = make_settings()
        assert s.reranker_pool_size == 80

    def test_reranker_doc_truncation_chars_default(self) -> None:
        s = make_settings()
        assert s.reranker_doc_truncation_chars == 1200

    def test_reranker_selective_threshold_default(self) -> None:
        s = make_settings()
        assert s.reranker_selective_threshold == 0.70

    def test_colbert_rerank_model_default(self) -> None:
        s = make_settings()
        assert s.colbert_rerank_model == "colbert-ir/colbertv2.0"

    def test_colbert_max_query_tokens_default(self) -> None:
        s = make_settings()
        assert s.colbert_max_query_tokens == 32

    def test_colbert_max_doc_tokens_default(self) -> None:
        s = make_settings()
        assert s.colbert_max_doc_tokens == 256


class TestRagConfigRerankerDefaults:
    def test_reranker_type_default(self) -> None:
        c = RagConfig()
        assert c.reranker_type == "cross_encoder"

    def test_reranker_pool_size_default(self) -> None:
        c = RagConfig()
        assert c.reranker_pool_size == 0

    def test_reranker_doc_truncation_chars_default(self) -> None:
        c = RagConfig()
        assert c.reranker_doc_truncation_chars == 2000

    def test_reranker_selective_threshold_default(self) -> None:
        c = RagConfig()
        assert c.reranker_selective_threshold == 1.0

    def test_colbert_rerank_model_default(self) -> None:
        c = RagConfig()
        assert c.colbert_rerank_model == "colbert-ir/colbertv2.0"

    def test_colbert_max_query_tokens_default(self) -> None:
        c = RagConfig()
        assert c.colbert_max_query_tokens == 32

    def test_colbert_max_doc_tokens_default(self) -> None:
        c = RagConfig()
        assert c.colbert_max_doc_tokens == 256


class TestRagConfigRerankerCustom:
    def test_custom_values(self) -> None:
        c = RagConfig(
            reranker_type="colbert",
            reranker_pool_size=100,
            reranker_doc_truncation_chars=1500,
            reranker_selective_threshold=0.8,
            colbert_rerank_model="custom/model",
            colbert_max_query_tokens=64,
            colbert_max_doc_tokens=512,
        )
        assert c.reranker_type == "colbert"
        assert c.reranker_pool_size == 100
        assert c.reranker_doc_truncation_chars == 1500
        assert c.reranker_selective_threshold == 0.8
        assert c.colbert_rerank_model == "custom/model"
        assert c.colbert_max_query_tokens == 64
        assert c.colbert_max_doc_tokens == 512
