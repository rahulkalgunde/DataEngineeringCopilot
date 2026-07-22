"""Tests for reranker singleton and lazy load behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestRerankerSingleton:
    def test_singleton_returns_same_instance(self):
        from data_engineering_copilot.services.reranker import get_reranker

        r1 = get_reranker()
        r2 = get_reranker()
        assert r1 is r2

    def test_singleton_with_same_model(self):
        from data_engineering_copilot.services.reranker import get_reranker

        r1 = get_reranker(model_name="test-model")
        r2 = get_reranker(model_name="test-model")
        assert r1 is r2

    def test_different_models_return_different_instances(self):
        from data_engineering_copilot.services.reranker import get_reranker

        r1 = get_reranker(model_name="model-a")
        r2 = get_reranker(model_name="model-b")
        assert r1 is not r2


class TestRerankerLazyLoad:
    def test_model_not_loaded_when_import_fails(self):
        from data_engineering_copilot.services.reranker import CrossEncoderReranker

        with patch.dict("sys.modules", {"sentence_transformers": None}):
            r = CrossEncoderReranker(model_name="test-model")
            assert r.model is None
            assert r.is_available() is False

    def test_model_loaded_on_init(self):
        from data_engineering_copilot.services.reranker import CrossEncoderReranker

        mock_model = MagicMock()
        mock_module = MagicMock()
        mock_module.CrossEncoder.return_value = mock_model
        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            r = CrossEncoderReranker(model_name="test-model")
            assert r.model is not None
            assert r.is_available() is True

    def test_singleton_clear_resets(self):
        from data_engineering_copilot.services.reranker import clear_reranker_cache, get_reranker

        r1 = get_reranker()
        clear_reranker_cache()
        r2 = get_reranker()
        assert r1 is not r2
