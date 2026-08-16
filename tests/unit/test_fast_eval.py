"""Hermetic tests for the free (zero-LLM) layered integrity evaluator.

Covers the pure functions in ``data_engineering_copilot/evaluation/fast_eval``:
chunk statistics, boundary heuristics, embedding validation, consistency and
semantic ordering. No infra, no LLM.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.evaluation.fast_eval import (
    _ends_mid_sentence,
    _looks_like_continuation,
    _splits_code_fence,
    _splits_markdown_table,
    chunk_boundary_issues,
    chunk_size_stats,
    cosine_similarity,
    embedding_consistency,
    embedding_semantic_sanity,
    validate_embedding,
)


def test_chunk_size_stats_empty() -> None:
    stats = chunk_size_stats([])
    assert stats["count"] == 0
    assert stats["min_chars"] is None


def test_chunk_size_stats_distribution() -> None:
    chunks = [{"text": "a" * 100}, {"text": "b" * 200}, {"text": "c" * 300}]
    stats = chunk_size_stats(chunks)
    assert stats["count"] == 3
    assert stats["min_chars"] == 100
    assert stats["max_chars"] == 300
    assert stats["mean_chars"] == 200
    assert stats["median_chars"] == 200
    assert stats["empty"] == 0


def test_chunk_size_stats_flags_oversized_and_empty() -> None:
    chunks = [{"text": ""}, {"text": "x" * 7000}]
    stats = chunk_size_stats(chunks)
    assert stats["empty"] == 1
    assert stats["oversized"] == 1


def test_looks_like_continuation() -> None:
    assert _looks_like_continuation("configuration parameter spark.sql.shuffle")
    assert not _looks_like_continuation("Configuration parameter")
    assert not _looks_like_continuation("  Starting with capital")
    assert not _looks_like_continuation("```python")


def test_ends_mid_sentence() -> None:
    assert _ends_mid_sentence("the number of partitions used")
    assert not _ends_mid_sentence("the number of partitions.")
    assert not _ends_mid_sentence("the number of partitions.")
    assert not _ends_mid_sentence("the number of partitions.")


def test_splits_code_fence() -> None:
    assert _splits_code_fence("text\n```python\nprint(1)")
    assert not _splits_code_fence("text\n```python\nprint(1)\n```")


def test_splits_markdown_table() -> None:
    assert _splits_markdown_table("| name | value |")
    assert not _splits_markdown_table("plain text")


def test_chunk_boundary_issues_flags() -> None:
    chunks = [
        {"chunk_id": "c1", "text": "the configuration parameter controls"},
        {"chunk_id": "c2", "text": "```python\nprint(1)"},
        {"chunk_id": "c3", "text": "| name | value |"},
        {"chunk_id": "c4", "text": "A complete sentence."},
    ]
    issues = chunk_boundary_issues(chunks)
    by_id = {i["chunk_id"]: i["issues"] for i in issues}
    assert "ends_mid_sentence" in by_id["c1"]
    assert "unbalanced_code_fence" in by_id["c2"]
    assert "starts_markdown_table" in by_id["c3"]
    assert "c4" not in by_id


def test_cosine_similarity_mismatch_returns_zero() -> None:
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0
    assert cosine_similarity([], []) == 0.0


def test_cosine_similarity_identical_vectors() -> None:
    assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_validate_embedding_dimension() -> None:
    problems = validate_embedding([0.1, 0.2], 3)
    assert any("dimension_mismatch" in p for p in problems)


def test_validate_embedding_nan() -> None:
    problems = validate_embedding([0.1, float("nan")], 2)
    assert any("nonfinite" in p for p in problems)


def test_validate_embedding_inf() -> None:
    problems = validate_embedding([0.1, float("inf")], 2)
    assert any("nonfinite" in p for p in problems)


def test_validate_embedding_zero_norm() -> None:
    problems = validate_embedding([0.0, 0.0], 2)
    assert "zero_norm" in problems


def test_validate_embedding_clean() -> None:
    assert validate_embedding([0.1, 0.2], 2) == []


def test_embedding_consistency_deterministic_embedder() -> None:
    embed = lambda _text: [0.5, 0.5]  # noqa: E731
    result = embedding_consistency(embed, "How do I configure Spark executor memory?")
    assert result["similarity"] == pytest.approx(1.0)


def test_embedding_semantic_sanity_sync() -> None:
    def embed(text: str) -> list[float]:
        # Deterministic fake: dim-2 vector keyed by a couple of tokens.
        vec = [0.0, 0.0]
        if "spark" in text:
            vec[0] += 1.0
        if "executor" in text or "memory" in text:
            vec[1] += 1.0
        if "dataframe" in text:
            vec[0] += 0.5
        if "column" in text:
            vec[1] += 0.5
        norm = (vec[0] ** 2 + vec[1] ** 2) ** 0.5
        return [v / norm if norm else 0.0 for v in vec]

    pairs = [
        {"query": "Spark executor memory", "relevant": "Spark executor memory config", "irrelevant": "Docker volume"},
    ]
    result = embedding_semantic_sanity(embed, pairs)
    assert result["passed"] == 1
    assert result["pairs"] == 1
    assert result["results"][0]["sim_relevant"] > result["results"][0]["sim_irrelevant"]
