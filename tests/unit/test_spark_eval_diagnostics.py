"""Hermetic tests for the Spark eval retrieval diagnostics helpers.

These cover the pure functions in ``cli.py`` that turn a query result plus its
retrieval provenance into candidate-vs-final recall metrics. No infra, no LLM.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.cli import (
    _compute_spark_eval_metrics,
    _compute_spark_eval_result,
    _percentile,
)
from data_engineering_copilot.domain.models import Answer, DocumentChunk


def _chunk(cid: str, url: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=cid,
        source_name="Test",
        title="t",
        url=url,
        text="content",
    )


def test_percentile_empty_returns_none() -> None:
    assert _percentile([], 0.5) is None


def test_percentile_single_value() -> None:
    assert _percentile([5.0], 0.5) == 5.0


def test_percentile_interpolated() -> None:
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)


def test_compute_result_metrics_and_provenance() -> None:
    fused = [
        {"rank": 0, "chunk_id": "c1", "url": "https://docs/expected.html", "distance": 0.1, "confidence": 0.9},
        {"rank": 1, "chunk_id": "c2", "url": "https://docs/other.html", "distance": 0.2, "confidence": 0.8},
        {"rank": 2, "chunk_id": "c3", "url": "https://docs/expected2.html", "distance": 0.3, "confidence": 0.7},
    ]
    final = [
        {"rank": 0, "chunk_id": "c1", "url": "https://docs/expected.html", "distance": 0.1, "confidence": 0.9},
        {"rank": 1, "chunk_id": "c2", "url": "https://docs/other.html", "distance": 0.2, "confidence": 0.8},
    ]
    prov = {
        "schema_version": "1",
        "question": "q",
        "effective_query": "q",
        "cache_hit": False,
        "query_variants": [],
        "fused": fused,
        "rerank": {"enabled": True, "pool_size": 3, "top_k": 3, "final_top_k": 2},
        "final_context": final,
        "candidate_pool_size": 3,
        "stage_times": {"retrieval": 12.3, "rerank": 45.6, "total": 120.0},
    }
    answer = Answer(
        text="An answer.\n\nMissing information: transform docs",
        sources=(_chunk("c1", "https://docs/expected.html"), _chunk("c2", "https://docs/other.html")),
        confidence=0.8,
    )
    item = {
        "id": "q1",
        "expected_terms": ["filter", "transform"],
        "expected_urls": ["https://docs/expected.html", "https://docs/expected2.html"],
    }
    context = "filter documentation is available here"

    result = _compute_spark_eval_result(item, "What is filter?", answer, context, prov)

    # Term recall from assembled context: only "filter" appears.
    assert result["term_recall"] == 0.5
    # Source recall from final context: expected.html present, expected2.html absent.
    assert result["source_recall"] == 0.5
    # Candidate recall from fused: both expected URLs present.
    assert result["candidate_source_recall"] == 1.0
    # expected2.html was retrieved (fused rank 2) but dropped from final context.
    assert result["expected_fused_ranks"] == {
        "https://docs/expected.html": 0,
        "https://docs/expected2.html": 2,
    }
    assert result["dropped_expected_urls"] == ["https://docs/expected2.html"]
    assert result["candidate_pool_size"] == 3
    assert result["rerank_enabled"] is True
    assert result["rerank_pool_size"] == 3
    assert result["insufficient_context"] is True
    assert result["retrieval_ms"] == 12.3
    assert result["total_ms"] == 120.0


def test_compute_result_empty_provenance() -> None:
    answer = Answer(
        text="I cannot answer this question because it is outside my knowledge repository.",
        sources=(),
        confidence=0.0,
    )
    result = _compute_spark_eval_result({"id": "q2"}, "nope", answer, "", {})
    assert result["source_recall"] == 0.0
    assert result["candidate_source_recall"] == 0.0
    assert result["candidate_pool_size"] == 0
    assert result["insufficient_context"] is True
    assert result["dropped_expected_urls"] == []


def test_compute_metrics_aggregates() -> None:
    results = [
        {
            "id": "a",
            "term_recall": 1.0,
            "source_recall": 1.0,
            "candidate_source_recall": 1.0,
            "insufficient_context": False,
            "dropped_expected_urls": [],
            "cache_hit": False,
            "retrieval_ms": 10.0,
        },
        {
            "id": "b",
            "term_recall": 0.0,
            "source_recall": 0.0,
            "candidate_source_recall": 1.0,
            "insufficient_context": True,
            "dropped_expected_urls": ["https://x"],
            "cache_hit": False,
            "retrieval_ms": 30.0,
        },
    ]
    metrics = _compute_spark_eval_metrics(results)
    assert metrics["query_count"] == 2
    assert metrics["avg_term_recall"] == 0.5
    assert metrics["avg_source_recall"] == 0.5
    assert metrics["avg_candidate_source_recall"] == 1.0
    assert metrics["insufficient_context_rate"] == 0.5
    assert metrics["queries_dropping_expected_sources"] == 1
    assert metrics["median_retrieval_ms"] == 20.0
    assert metrics["p95_retrieval_ms"] == 29.0


def test_spark_eval_initializes_reranker(monkeypatch, tmp_path) -> None:
    """The spark-eval harness must initialize the reranker (mirrors the API
    singleton path) so baselines measure the real reranking pipeline."""
    import json

    from data_engineering_copilot.cli import evaluate_spark_dataset

    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "q1",
                "question": "What is filter?",
                "expected_terms": ["filter"],
                "expected_urls": ["https://docs/filter.html"],
            }
        ),
        encoding="utf-8",
    )

    initialized = {"called": False}

    class _Reranker:
        async def initialize(self) -> None:
            initialized["called"] = True

    class _Service:
        reranker = _Reranker()

        async def answer(self, query, provenance=None, bypass_cache=False):
            assert provenance is not None
            provenance.append(
                {
                    "fused": [],
                    "final_context": [],
                    "rerank": {"enabled": True, "pool_size": 0, "top_k": 0, "final_top_k": 0},
                    "stage_times": {"retrieval": 1.0, "rerank": 1.0, "total": 2.0},
                }
            )
            return _Answer()

    class _Answer:
        text = "filter docs"
        sources = ()
        stage_times = {"retrieval": 1.0, "rerank": 1.0, "total": 2.0}

    monkeypatch.setattr("data_engineering_copilot.factory.build_rag_service", lambda: _Service())

    # Empty sources ⇒ recall below threshold ⇒ exit 1; the important assertion is
    # that the harness initialized the reranker before running queries.
    assert evaluate_spark_dataset(dataset) == 1
    assert initialized["called"] is True


def test_spark_eval_dataset_sql_function_rows_target_scala_source() -> None:
    """Q3–5 of the Spark eval must point at the higherOrderFunctions.scala
    source (the Jekyll hub page only contains ``{% include_api_gen %}`` tags)."""
    import json
    from pathlib import Path

    dataset_path = Path(__file__).resolve().parents[1] / "evaluation" / "eval_dataset_spark.jsonl"
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    targets = {
        "spark-nested-array-filter-003",
        "spark-array-transform-004",
        "spark-array-aggregate-005",
    }
    scala_src = "sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/higherOrderFunctions.scala"
    for row in rows:
        if row["id"] not in targets:
            continue
        assert any(scala_src in url for url in row["expected_urls"]), row["id"]
        assert "sql-ref-functions-builtin.md" not in " ".join(row["expected_urls"]), row["id"]
        assert "sql_function_ref" in row["expected_doc_types"], row["id"]


def _spark_dataset_rows() -> list[dict]:
    import json
    from pathlib import Path

    dataset_path = Path(__file__).resolve().parents[1] / "evaluation" / "eval_dataset_spark.jsonl"
    return [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_spark_eval_dataset_every_row_is_spark_or_out_of_scope() -> None:
    """Gate: every evaluation row is Spark-only or explicitly out of scope."""
    rows = _spark_dataset_rows()
    assert rows, "dataset must not be empty"
    for row in rows:
        if row.get("out_of_scope"):
            # Out-of-scope rows expect a refusal and carry no Spark evidence.
            assert row["expected_urls"] == [], row["id"]
            assert row["expected_terms"], row["id"]
            continue
        assert row["expected_urls"], f"in-scope row {row['id']} must specify expected URLs"
        assert row["expected_terms"], f"row {row['id']} must specify expected terms"
        assert row["expected_doc_types"], f"row {row['id']} must specify expected doc types"
        assert "forbidden_terms" in row, f"row {row['id']} must specify forbidden terms"
        for url in row["expected_urls"]:
            assert "spark.apache.org" in url or "raw.githubusercontent.com/apache/spark/" in url, row["id"]
            assert "airflow" not in url.lower(), row["id"]


def test_spark_eval_dataset_has_no_nonspark_rows() -> None:
    """The original Airflow row must be gone; Airflow/Delta are out-of-scope only."""
    rows = _spark_dataset_rows()
    for row in rows:
        if "airflow" in (row["question"] or "").lower() or "delta" in (row["question"] or "").lower():
            assert row.get("out_of_scope") is True, row["id"]


def test_spark_eval_dataset_covers_required_topics() -> None:
    """JDBC, data-source, streaming, and deployment rows must be present."""
    rows = _spark_dataset_rows()
    ids = {row["id"] for row in rows}
    assert any("jdbc" in row_id for row_id in ids)
    assert any("csv" in row_id or "parquet" in row_id for row_id in ids)
    assert any("streaming" in row_id for row_id in ids)
    assert any("deployment" in row_id for row_id in ids)
    # Window, array, struct, SQL-function, API rows retained.
    assert "spark-window-rolling-001" in ids
    assert "spark-window-dense-rank-002" in ids
    assert "spark-nested-array-filter-003" in ids
    assert "spark-array-transform-004" in ids
    assert "spark-array-aggregate-005" in ids
    assert "spark-struct-access-006" in ids
    assert "spark-api-filter-007" in ids
    assert "spark-api-transform-008" in ids
    assert "spark-api-aggregate-009" in ids
    assert "nonspark-source-control-010" not in ids


def test_spark_eval_dataset_out_of_scope_rows_exist() -> None:
    """Kubernetes and React rows are present but marked out of scope.

    The original Delta Lake/Airflow out-of-scope rows were replaced during the
    T3 reconciliation because the combined pinned generation legitimately
    answers those topics (they are in the corpus); Kubernetes and React are
    genuinely outside the Spark knowledge base.
    """
    rows = _spark_dataset_rows()
    oos = [row for row in rows if row.get("out_of_scope")]
    assert len(oos) == 2
    topics = " ".join((row["question"] or "").lower() for row in oos)
    assert "kubernetes" in topics
    assert "react" in topics


def test_compute_result_reports_forbidden_term_hits() -> None:
    answer = Answer(
        text="Use the Delta Lake connector to read the table.",
        sources=(_chunk("c1", "https://docs/spark.html"),),
        confidence=0.9,
    )
    item = {
        "id": "q1",
        "expected_terms": ["read"],
        "expected_urls": ["https://docs/spark.html"],
        "forbidden_terms": ["delta"],
    }
    result = _compute_spark_eval_result(item, "read a table", answer, "read with delta", {})
    assert result["forbidden_term_hits"] == ["delta"]


def test_compute_result_out_of_scope_refusal() -> None:
    answer = Answer(
        text="I cannot answer this question because it is outside my knowledge repository.",
        sources=(),
        confidence=0.0,
    )
    item = {"id": "oos", "out_of_scope": True, "expected_terms": ["delta"]}
    result = _compute_spark_eval_result(item, "Delta time travel", answer, "", {})
    assert result["out_of_scope"] is True
    assert result["insufficient_context"] is True


def test_compute_metrics_excludes_out_of_scope_from_thresholds() -> None:
    results = [
        {
            "id": "a",
            "out_of_scope": False,
            "term_recall": 1.0,
            "source_recall": 1.0,
            "candidate_source_recall": 1.0,
            "insufficient_context": False,
            "dropped_expected_urls": [],
            "forbidden_term_hits": [],
            "cache_hit": False,
            "retrieval_ms": 10.0,
        },
        {
            "id": "b",
            "out_of_scope": False,
            "term_recall": 1.0,
            "source_recall": 1.0,
            "candidate_source_recall": 1.0,
            "insufficient_context": False,
            "dropped_expected_urls": [],
            "forbidden_term_hits": [],
            "cache_hit": False,
            "retrieval_ms": 20.0,
        },
        {
            "id": "oos",
            "out_of_scope": True,
            "term_recall": 0.0,
            "source_recall": 0.0,
            "candidate_source_recall": 0.0,
            "insufficient_context": True,
            "dropped_expected_urls": [],
            "forbidden_term_hits": [],
            "cache_hit": False,
            "retrieval_ms": 30.0,
        },
    ]
    metrics = _compute_spark_eval_metrics(results)
    assert metrics["in_scope_query_count"] == 2
    assert metrics["out_of_scope_query_count"] == 1
    assert metrics["avg_term_recall"] == 1.0
    assert metrics["avg_source_recall"] == 1.0
    assert metrics["out_of_scope_refusal_rate"] == 1.0


def test_compute_result_detects_insufficient_context_json() -> None:
    """A structured INSUFFICIENT_CONTEXT refusal (JSON status) is detected."""
    answer = Answer(
        text='{\n  "status": "INSUFFICIENT_CONTEXT",\n  "answer": null,\n'
        '  "missing_info": "The documentation does not cover Delta Lake."\n}',
        sources=(_chunk("c1", "https://docs/spark.html"),),
        confidence=0.6,
    )
    item = {"id": "oos", "out_of_scope": True, "expected_terms": ["delta"]}
    result = _compute_spark_eval_result(item, "Delta time travel", answer, "", {})
    assert result["insufficient_context"] is True
