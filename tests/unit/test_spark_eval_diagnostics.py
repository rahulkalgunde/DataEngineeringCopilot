"""Hermetic tests for the Spark eval retrieval diagnostics helpers.

These cover the pure functions in ``cli.py`` that turn a query result plus its
retrieval provenance into candidate-vs-final recall metrics. No infra, no LLM.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.cli import (
    _compute_spark_eval_metrics,
    _compute_spark_eval_result,
    _compute_stage_recalls,
    _percentile,
    gate_oos_refusal_rate,
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

        async def answer(self, query, provenance=None, bypass_cache=False, retrieval_only=False):
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


def test_spark_eval_splits_retrieval_only_and_full_generation(monkeypatch, tmp_path) -> None:
    """In-scope rows without forbidden terms are scored retrieval-only (no
    generation), while out-of-scope and forbidden-term rows run full generation."""
    import json

    from data_engineering_copilot.cli import evaluate_spark_dataset

    dataset = tmp_path / "eval.jsonl"
    rows = [
        {
            "id": "in_scope",
            "question": "What is filter?",
            "expected_terms": ["filter"],
            "expected_urls": ["https://docs/filter.html"],
        },
        {
            "id": "oos",
            "question": "How does Kubernetes autoscale?",
            "expected_terms": ["kubernetes"],
            "expected_urls": [],
            "out_of_scope": True,
        },
        {
            "id": "forbidden",
            "question": "Delta time travel",
            "expected_terms": ["delta"],
            "expected_urls": ["https://docs/delta.html"],
            "forbidden_terms": ["delta"],
        },
    ]
    dataset.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    calls: list[tuple[str, bool]] = []

    class _Reranker:
        async def initialize(self) -> None:
            return None

    class _Service:
        reranker = _Reranker()

        async def answer(self, query, provenance=None, bypass_cache=False, expected_urls=None, retrieval_only=False):
            calls.append((query, retrieval_only))
            assert provenance is not None
            provenance.append(
                {
                    "fused": [{"rank": 0, "chunk_id": "c1", "url": "https://docs/filter.html", "confidence": 0.9}],
                    "final_context": [
                        {"rank": 0, "chunk_id": "c1", "url": "https://docs/filter.html", "confidence": 0.9}
                    ],
                    "rerank": {"enabled": True, "pool_size": 1, "top_k": 1, "final_top_k": 1},
                    "stage_times": {"retrieval": 1.0, "rerank": 1.0, "total": 2.0},
                }
            )
            return _Answer()

    class _Answer:
        text = "filter docs"
        sources = (_chunk("c1", "https://docs/filter.html"),)
        stage_times = {"retrieval": 1.0, "rerank": 1.0, "total": 2.0}

    monkeypatch.setattr("data_engineering_copilot.factory.build_rag_service", lambda: _Service())

    evaluate_spark_dataset(dataset)

    by_id = {row["id"]: row["question"] for row in rows}
    call_map = {question: retrieval_only for question, retrieval_only in calls}
    assert call_map[by_id["in_scope"]] is True
    assert call_map[by_id["oos"]] is False
    assert call_map[by_id["forbidden"]] is False


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


def test_spark_eval_dataset_every_row_is_corpus_or_out_of_scope() -> None:
    """Gate: every evaluation row targets the pinned corpus or is out of scope."""
    rows = _spark_dataset_rows()
    assert rows, "dataset must not be empty"
    for row in rows:
        if row.get("out_of_scope"):
            # Out-of-scope rows expect a refusal and carry no evidence.
            assert row["expected_urls"] == [], row["id"]
            assert row["expected_terms"], row["id"]
            continue
        assert row["expected_urls"], f"in-scope row {row['id']} must specify expected URLs"
        assert row["expected_terms"], f"row {row['id']} must specify expected terms"
        assert row["expected_doc_types"], f"row {row['id']} must specify expected doc types"
        assert "forbidden_terms" in row, f"row {row['id']} must specify forbidden terms"
        for url in row["expected_urls"]:
            assert any(
                host in url
                for host in (
                    "spark.apache.org",
                    "raw.githubusercontent.com/apache/spark/",
                    "raw.githubusercontent.com/apache/airflow/",
                    "raw.githubusercontent.com/delta-io/delta/",
                    "platform.claude.com",
                    "code.claude.com",
                    "docs.databricks.com",
                )
            ), row["id"]


def test_spark_eval_dataset_covers_corpus_sources() -> None:
    """Airflow, Delta, Claude, and Databricks rows are in-scope now that the
    combined pinned generation legitimately answers those topics."""
    rows = _spark_dataset_rows()
    in_scope = [row for row in rows if not row.get("out_of_scope")]
    topics = " ".join((row["question"] or "").lower() for row in in_scope)
    assert "airflow" in topics, "Airflow rows missing"
    assert "delta" in topics, "Delta rows missing"
    assert "claude" in topics, "Claude rows missing"


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


def test_compute_stage_recalls_basic() -> None:
    """Stage recall/survival is computed from provenance snapshots."""
    prov = {
        "stage_snapshots": [
            {
                "stage": "dense_retrieval",
                "chunk_ids": ["c1", "c2"],
                "urls": ["https://docs/a.html", "https://docs/b.html"],
                "count": 2,
            },
            {
                "stage": "sibling_rejoin",
                "chunk_ids": ["c1", "c2", "c3"],
                "urls": ["https://docs/a.html", "https://docs/b.html", "https://docs/c.html"],
                "count": 3,
            },
            {"stage": "rerank", "chunk_ids": ["c1"], "urls": ["https://docs/a.html"], "count": 1},
        ]
    }
    expected = ["https://docs/a.html", "https://docs/b.html"]
    stage_recalls = _compute_stage_recalls(prov, expected, k=2)
    assert "dense_retrieval" in stage_recalls
    assert "sibling_rejoin" in stage_recalls
    assert "rerank" in stage_recalls
    # dense: both expected URLs in top-2 -> recall=1.0
    assert stage_recalls["dense_retrieval"]["recall_at_k"] == 1.0
    # sibling: added c3, both expected still in top-2 -> recall=1.0
    assert stage_recalls["sibling_rejoin"]["recall_at_k"] == 1.0
    # rerank: only c1 remains -> recall=0.5
    assert stage_recalls["rerank"]["recall_at_k"] == 0.5
    # survival rates
    assert stage_recalls["dense_retrieval"]["survival_rate"] == 1.0
    assert stage_recalls["sibling_rejoin"]["survival_rate"] == 1.0
    assert stage_recalls["rerank"]["survival_rate"] == pytest.approx(1.0 / 3)


def test_compute_stage_recalls_empty() -> None:
    """Empty snapshots yield empty stage metrics."""
    assert _compute_stage_recalls({}, []) == {}
    assert _compute_stage_recalls({"stage_snapshots": []}, ["u1"]) == {}


def test_compute_spark_eval_result_includes_stage_recalls() -> None:
    """Per-query result includes stage_recalls when provenance has snapshots."""
    prov = {
        "schema_version": "1",
        "question": "q",
        "effective_query": "q",
        "cache_hit": False,
        "query_variants": [],
        "fused": [],
        "rerank": {"enabled": False},
        "final_context": [],
        "dropped": [],
        "expected_urls": ["https://docs/a.html"],
        "candidate_pool_size": 1,
        "stage_snapshots": [
            {"stage": "dense_retrieval", "chunk_ids": ["c1"], "urls": ["https://docs/a.html"], "count": 1},
            {"stage": "rerank", "chunk_ids": ["c1"], "urls": ["https://docs/a.html"], "count": 1},
        ],
        "stage_times": {"retrieval": 1.0},
    }
    answer = Answer(
        text="answer",
        sources=(_chunk("c1", "https://docs/a.html"),),
        confidence=0.9,
    )
    item = {"id": "q1", "expected_urls": ["https://docs/a.html"]}
    result = _compute_spark_eval_result(item, "q", answer, "context", prov)
    assert "stage_recalls" in result
    assert result["stage_recalls"]["dense_retrieval"]["recall_at_k"] == 1.0
    assert result["stage_recalls"]["rerank"]["recall_at_k"] == 1.0


def test_compute_metrics_aggregates_stage_recalls() -> None:
    """Aggregated metrics include per-stage avg recall and survival."""
    results = [
        {
            "id": "a",
            "term_recall": 1.0,
            "source_recall": 1.0,
            "candidate_source_recall": 1.0,
            "insufficient_context": False,
            "dropped_expected_urls": [],
            "forbidden_term_hits": [],
            "cache_hit": False,
            "retrieval_ms": 10.0,
            "stage_recalls": {
                "dense_retrieval": {"recall_at_k": 1.0, "survival_rate": 1.0},
                "rerank": {"recall_at_k": 0.8, "survival_rate": 0.5},
            },
        },
        {
            "id": "b",
            "term_recall": 0.0,
            "source_recall": 0.0,
            "candidate_source_recall": 0.0,
            "insufficient_context": True,
            "dropped_expected_urls": [],
            "forbidden_term_hits": [],
            "cache_hit": False,
            "retrieval_ms": 30.0,
            "stage_recalls": {
                "dense_retrieval": {"recall_at_k": 0.0, "survival_rate": 1.0},
                "rerank": {"recall_at_k": 0.0, "survival_rate": 0.0},
            },
        },
    ]
    metrics = _compute_spark_eval_metrics(results)
    assert "stage_recalls" in metrics
    assert metrics["stage_recalls"]["dense_retrieval"]["avg_recall_at_k"] == 0.5
    assert metrics["stage_recalls"]["rerank"]["avg_recall_at_k"] == 0.4
    assert metrics["stage_recalls"]["rerank"]["avg_survival_rate"] == 0.25


def test_gate_oos_refusal_rate_passes() -> None:
    results = [
        {"id": "a", "out_of_scope": False, "insufficient_context": False},
        {"id": "oos1", "out_of_scope": True, "insufficient_context": True},
        {"id": "oos2", "out_of_scope": True, "insufficient_context": True},
    ]
    verdict = gate_oos_refusal_rate(results, threshold=0.95)
    assert verdict["passed"] is True
    assert verdict["rate"] == 1.0
    assert verdict["oos_count"] == 2


def test_gate_oos_refusal_rate_fails() -> None:
    results = [
        {"id": "a", "out_of_scope": False, "insufficient_context": False},
        {"id": "oos1", "out_of_scope": True, "insufficient_context": True},
        {"id": "oos2", "out_of_scope": True, "insufficient_context": False},
    ]
    verdict = gate_oos_refusal_rate(results, threshold=0.95)
    assert verdict["passed"] is False
    assert verdict["rate"] == 0.5
    assert verdict["reason"] == "below_threshold"


def test_gate_oos_refusal_rate_no_oos() -> None:
    results = [
        {"id": "a", "out_of_scope": False, "insufficient_context": False},
    ]
    verdict = gate_oos_refusal_rate(results, threshold=0.95)
    assert verdict["passed"] is True
    assert verdict["rate"] == 1.0
    assert verdict["reason"] == "no_oos_rows"
