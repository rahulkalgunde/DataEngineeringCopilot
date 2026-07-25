"""Tests for RagasEvaluator — RAGAS evaluation wrapper (ragas_evaluation.py:24)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from data_engineering_copilot.services.ragas_evaluation import RagasEvaluator


class MockDatasets:
    class Dataset:
        @staticmethod
        def from_dict(data):
            return MagicMock()


class TestRagasEvaluator:
    def test_ragas_not_installed_returns_none(self):
        ev = RagasEvaluator()
        result = ev.evaluate(questions=["q"], answers=["a"], contexts=[["c"]])
        assert result is None

    def test_ragas_installed_returns_result(self):
        mock_result = MagicMock()
        mock_result.get.side_effect = lambda key, default=0: {
            "context_recall": 0.8,
            "context_precision": 0.9,
            "faithfulness": 0.95,
            "answer_relevancy": 0.85,
        }.get(key, default)

        ev = RagasEvaluator()
        with (
            patch.object(ev, "_lazy_init", return_value=True),
            patch.dict("sys.modules", {"datasets": MockDatasets}, clear=False),
            patch.object(ev, "_evaluate", return_value=mock_result),
        ):
            result = ev.evaluate(
                questions=["What is Spark?"],
                answers=["Spark is a engine."],
                contexts=[["Spark documentation"]],
            )

        assert result is not None
        assert result.context_recall == 0.8
        assert result.context_precision == 0.9
        assert result.faithfulness == 0.95
        assert result.answer_relevancy == 0.85
        expected_overall = round(0.8 * 0.3 + 0.95 * 0.4 + 0.85 * 0.3, 4)
        assert result.overall == expected_overall

    def test_with_ground_truth(self):
        mock_result = MagicMock()
        mock_result.get.side_effect = lambda key, default=0: {
            "context_recall": 1.0,
            "context_precision": 1.0,
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
        }.get(key, default)

        ev = RagasEvaluator()
        with (
            patch.object(ev, "_lazy_init", return_value=True),
            patch.dict("sys.modules", {"datasets": MockDatasets}, clear=False),
            patch.object(ev, "_evaluate", return_value=mock_result),
        ):
            ev.evaluate(
                questions=["q"],
                answers=["a"],
                contexts=[["c"]],
                ground_truth=["gt"],
            )

    def test_missing_keys_default_to_zero(self):
        mock_result = MagicMock()
        mock_result.get.return_value = 0

        ev = RagasEvaluator()
        with (
            patch.object(ev, "_lazy_init", return_value=True),
            patch.dict("sys.modules", {"datasets": MockDatasets}, clear=False),
            patch.object(ev, "_evaluate", return_value=mock_result),
        ):
            result = ev.evaluate(
                questions=["q"],
                answers=["a"],
                contexts=[["c"]],
            )

        assert result is not None
        assert result.context_recall == 0.0
        assert result.faithfulness == 0.0
        assert result.answer_relevancy == 0.0

    def test_lazy_init_caches_success(self):
        ev = RagasEvaluator()
        with patch("builtins.__import__") as mock_import:
            mock_ragas = MagicMock()
            mock_import.return_value = mock_ragas
            result = ev._lazy_init()
        assert result is True
        assert ev._evaluate is not None

    def test_lazy_init_failure_returns_false(self):
        ev = RagasEvaluator()
        with patch("builtins.__import__", side_effect=ImportError):
            result = ev._lazy_init()
        assert result is False
