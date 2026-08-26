"""Tests for evaluation/retrieval_metrics.py."""

from __future__ import annotations

from data_engineering_copilot.evaluation.retrieval_metrics import (
    ndcg_at_k,
    percentile,
    recall_at_k,
)


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["a", "b"], 3) == 1.0

    def test_partial_recall(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["a", "d"], 3) == 0.5

    def test_no_recall(self) -> None:
        assert recall_at_k(["a", "b"], ["c", "d"], 2) == 0.0

    def test_k_limits(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["c"], 1) == 0.0

    def test_empty_expected(self) -> None:
        assert recall_at_k(["a", "b"], [], 2) == 0.0


class TestNdcgAtK:
    def test_perfect_ndcg(self) -> None:
        assert ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == 1.0

    def test_partial_ndcg(self) -> None:
        result = ndcg_at_k(["a", "b", "c"], ["b", "a"], 3)
        assert 0 < result <= 1.0

    def test_no_match(self) -> None:
        assert ndcg_at_k(["a", "b"], ["c", "d"], 2) == 0.0

    def test_deduped(self) -> None:
        # Duplicate URLs should not earn credit twice
        result = ndcg_at_k(["a", "a", "b"], ["a", "b"], 3)
        assert result <= 1.0

    def test_empty_expected(self) -> None:
        assert ndcg_at_k(["a", "b"], [], 2) == 0.0


class TestPercentile:
    def test_median(self) -> None:
        assert percentile([1, 2, 3, 4, 5], 0.5) == 3.0

    def test_empty(self) -> None:
        assert percentile([], 0.5) == 0.0

    def test_single(self) -> None:
        assert percentile([42], 0.5) == 42

    def test_quartiles(self) -> None:
        vals = [1, 2, 3, 4]
        assert percentile(vals, 0.25) == 1.75
        assert percentile(vals, 0.75) == 3.25
