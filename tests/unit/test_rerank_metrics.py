"""Tests for reranker evaluation metrics."""

from __future__ import annotations

import pytest

from data_engineering_copilot.evaluation.rerank_metrics import (
    evaluate_reranker,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

pytestmark = pytest.mark.unit


class TestNDCG:
    def test_perfect_ordering(self):
        assert ndcg_at_k([1, 1, 1], 3) == 1.0

    def test_worst_ordering(self):
        score = ndcg_at_k([0, 0, 1], 3)
        assert 0.0 < score < 1.0

    def test_empty(self):
        assert ndcg_at_k([], 5) == 0.0

    def test_all_irrelevant(self):
        assert ndcg_at_k([0, 0, 0], 3) == 0.0

    def test_single_relevant(self):
        score = ndcg_at_k([0, 0, 1], 3)
        assert 0.0 < score < 1.0


class TestMRR:
    def test_first_relevant(self):
        assert mrr([1, 0, 0], 3) == 1.0

    def test_second_relevant(self):
        assert mrr([0, 1, 0], 3) == 0.5

    def test_no_relevant(self):
        assert mrr([0, 0, 0], 3) == 0.0

    def test_k_limits_search(self):
        assert mrr([0, 0, 1], 2) == 0.0

    def test_empty(self):
        assert mrr([], 5) == 0.0


class TestPrecision:
    def test_all_relevant(self):
        assert precision_at_k([1, 1, 1], 3) == 1.0

    def test_half_relevant(self):
        assert precision_at_k([1, 0, 1], 3) == pytest.approx(2 / 3)

    def test_k_zero(self):
        assert precision_at_k([1, 1], 0) == 0.0

    def test_empty(self):
        assert precision_at_k([], 5) == 0.0


class TestRecall:
    def test_full_recall(self):
        assert recall_at_k([1, 1, 0], 2, 3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k([1, 0, 0], 3, 3) == pytest.approx(1 / 3)

    def test_no_relevant_in_corpus(self):
        assert recall_at_k([0, 0], 0, 3) == 0.0

    def test_empty(self):
        assert recall_at_k([], 5, 3) == 0.0


class TestEvaluateReranker:
    def test_positive_gain(self):
        post = [1, 1, 0]
        pre = [0, 1, 1]
        result = evaluate_reranker(post, pre, k=3)
        assert result["ndcg_gain"] > 0
        assert result["mrr_gain"] > 0

    def test_no_gain(self):
        result = evaluate_reranker([1, 0, 0], [1, 0, 0], k=3)
        assert result["ndcg_gain"] == 0.0
        assert result["mrr_gain"] == 0.0
        assert result["precision_gain"] == 0.0

    def test_negative_gain(self):
        post = [0, 0, 1]
        pre = [1, 0, 0]
        result = evaluate_reranker(post, pre, k=3)
        assert result["ndcg_gain"] < 0
        assert result["mrr_gain"] < 0
