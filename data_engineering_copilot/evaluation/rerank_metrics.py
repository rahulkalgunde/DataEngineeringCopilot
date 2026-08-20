"""Reranker evaluation metrics: nDCG@K, MRR, Precision@K, Recall@K."""

from __future__ import annotations

import math


def dcg_at_k(relevance: list[int], k: int) -> float:
    """Discounted Cumulative Gain at position k."""
    score = 0.0
    for i, rel in enumerate(relevance[:k]):
        score += (2**rel - 1) / math.log2(i + 2)
    return score


def ndcg_at_k(relevance: list[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at position k.

    Args:
        relevance: Binary relevance labels (1=relevant, 0=not).
        k: Number of top positions to consider.
    """
    if not relevance:
        return 0.0
    actual = dcg_at_k(relevance, k)
    ideal = dcg_at_k(sorted(relevance, reverse=True), k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def mrr(labels: list[int], k: int) -> float:
    """Mean Reciprocal Rank at position k.

    Args:
        labels: Binary labels (1=relevant, 0=not) in ranked order.
        k: Number of top positions to consider.
    """
    for i, label in enumerate(labels[:k]):
        if label == 1:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevance: list[int], k: int) -> float:
    """Precision at position k.

    Args:
        relevance: Binary relevance labels (1=relevant, 0=not).
        k: Number of top positions to consider.
    """
    if k == 0:
        return 0.0
    top_k = relevance[:k]
    return sum(top_k) / k


def recall_at_k(relevance: list[int], total_relevant: int, k: int) -> float:
    """Recall at position k.

    Args:
        relevance: Binary relevance labels (1=relevant, 0=not).
        total_relevant: Total number of relevant documents in the corpus.
        k: Number of top positions to consider.
    """
    if total_relevant == 0:
        return 0.0
    top_k = relevance[:k]
    return sum(top_k) / total_relevant


def evaluate_reranker(
    post_rerank: list[int],
    pre_rerank: list[int],
    k: int,
) -> dict[str, float]:
    """Compute all metrics for pre- and post-rerank relevance lists.

    Args:
        post_rerank: Binary relevance labels after reranking.
        pre_rerank: Binary relevance labels before reranking.
        k: Cutoff position for metrics.

    Returns:
        Dict with ndcg_gain, mrr_gain, precision_gain, recall_gain.
    """
    post_metrics = {
        "ndcg": ndcg_at_k(post_rerank, k),
        "mrr": mrr(post_rerank, k),
        "precision": precision_at_k(post_rerank, k),
        "recall": recall_at_k(post_rerank, sum(post_rerank), k),
    }
    pre_metrics = {
        "ndcg": ndcg_at_k(pre_rerank, k),
        "mrr": mrr(pre_rerank, k),
        "precision": precision_at_k(pre_rerank, k),
        "recall": recall_at_k(pre_rerank, sum(pre_rerank), k),
    }
    return {
        "ndcg_gain": post_metrics["ndcg"] - pre_metrics["ndcg"],
        "mrr_gain": post_metrics["mrr"] - pre_metrics["mrr"],
        "precision_gain": post_metrics["precision"] - pre_metrics["precision"],
        "recall_gain": post_metrics["recall"] - pre_metrics["recall"],
    }
