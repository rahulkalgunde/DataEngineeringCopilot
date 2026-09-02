"""Hermetic tests for the worst-row scanner.

No live LLM/provider calls — pure offline analysis of in-memory data.
"""

from __future__ import annotations

import pytest

from data_engineering_copilot.evaluation.worst_row_scanner import (
    DEFAULT_WORST_N,
    EvalRow,
    PlatformMetrics,
    compute_platform_metrics,
    enrich_with_source,
    find_worst_rows,
    format_platform_summary,
    format_worst_row,
    load_evaluation_results,
    load_golden_qa,
    scan_worst_rows,
)


class TestEvalRow:
    def test_composite_score_avg(self):
        row = EvalRow(id="x", faithfulness=0.9, relevance=0.8, rubric=4.0)
        assert row.composite_score == pytest.approx((0.9 + 0.8 + 0.8) / 3.0)

    def test_composite_score_edge_cases(self):
        assert EvalRow(id="x", faithfulness=0.0, relevance=0.0, rubric=0.0).composite_score == 0.0
        assert EvalRow(id="x", faithfulness=1.0, relevance=1.0, rubric=5.0).composite_score == pytest.approx(1.0)

    def test_below_threshold_all_pass(self):
        row = EvalRow(
            id="x",
            faithfulness=0.90,
            relevance=0.85,
            rubric=4.5,
        )
        assert row.below_threshold() == []

    def test_below_threshold_faithfulness(self):
        row = EvalRow(
            id="x",
            faithfulness=0.70,
            relevance=0.90,
            rubric=4.5,
        )
        assert row.below_threshold() == ["faithfulness"]

    def test_below_threshold_multiple(self):
        row = EvalRow(
            id="x",
            faithfulness=0.70,
            relevance=0.70,
            rubric=3.0,
        )
        flags = row.below_threshold()
        assert "faithfulness" in flags
        assert "relevance" in flags
        assert "rubric" in flags


class TestLoadGoldenQa:
    def test_loads_source_names(self, tmp_path):
        golden = tmp_path / "qa_test.jsonl"
        golden.write_text(
            '{"id":"q-1","question":"a","ground_truth":"b","source_name":"Apache Spark 4.0.0"}\n'
            '{"id":"q-2","question":"c","ground_truth":"d","source_name":"Delta Lake"}\n'
            '{"id":"q-3","question":"e","ground_truth":"f"}\n',
            encoding="utf-8",
        )
        result = load_golden_qa(golden)
        assert result == {
            "q-1": "Apache Spark 4.0.0",
            "q-2": "Delta Lake",
        }

    def test_skips_comments_and_blanks(self, tmp_path):
        golden = tmp_path / "qa_test.jsonl"
        golden.write_text(
            "# version: 2026-09-01\n\n"
            '{"id":"q-1","question":"a","ground_truth":"b","source_name":"Spark"}\n'
            "\n"
            '{"id":"q-2","question":"c","ground_truth":"d","source_name":"Delta"}\n',
            encoding="utf-8",
        )
        result = load_golden_qa(golden)
        assert len(result) == 2


class TestLoadEvaluationResults:
    def test_loads_scores(self, tmp_path):
        results = tmp_path / "results.jsonl"
        results.write_text(
            '{"id":"r-1","faithfulness":0.85,"relevance":0.80,"rubric":4.0,"question":"q1","answer":"a1"}\n'
            '{"id":"r-2","faithfulness":0.60,"relevance":0.55,"rubric":3.0,"question":"q2","answer":"a2"}\n',
            encoding="utf-8",
        )
        rows = load_evaluation_results(results)
        assert len(rows) == 2
        assert rows[0].id == "r-1"
        assert rows[0].faithfulness == 0.85
        assert rows[1].rubric == 3.0

    def test_loads_judge_votes(self, tmp_path):
        results = tmp_path / "results.jsonl"
        results.write_text(
            '{"id":"r-1","faithfulness":0.9,"relevance":0.9,"rubric":4.5,"question":"q","answer":"a",'
            '"judge_votes":[{"faithfulness":0.9,"relevance":0.9,"rubric":4.5},{"faithfulness":0.8,"relevance":0.8,"rubric":4.0}]}\n',
            encoding="utf-8",
        )
        rows = load_evaluation_results(results)
        assert len(rows[0].judge_votes) == 2

    def test_skips_comments(self, tmp_path):
        results = tmp_path / "results.jsonl"
        results.write_text(
            '# header\n\n{"id":"r-1","faithfulness":0.9,"relevance":0.9,"rubric":4.0,"question":"q","answer":"a"}\n',
            encoding="utf-8",
        )
        rows = load_evaluation_results(results)
        assert len(rows) == 1


class TestEnrichWithSource:
    def test_enriches_by_id(self):
        rows = [
            EvalRow(id="q-1", faithfulness=0.9, relevance=0.9, rubric=4.0),
            EvalRow(id="q-2", faithfulness=0.8, relevance=0.8, rubric=4.0),
            EvalRow(id="q-3", faithfulness=0.7, relevance=0.7, rubric=4.0),
        ]
        source_map = {"q-1": "Spark", "q-2": "Delta Lake"}
        enrich_with_source(rows, source_map)
        assert rows[0].source_name == "Spark"
        assert rows[1].source_name == "Delta Lake"
        assert rows[2].source_name == "unknown"

    def test_preserves_existing_source_name(self):
        rows = [
            EvalRow(id="q-1", faithfulness=0.9, relevance=0.9, rubric=4.0, source_name="Airflow"),
        ]
        enrich_with_source(rows, {"q-1": "Spark"})
        assert rows[0].source_name == "Airflow"


class TestComputePlatformMetrics:
    def test_aggregates_correctly(self):
        rows = [
            EvalRow(id="q-1", faithfulness=0.9, relevance=0.9, rubric=4.0, source_name="Spark"),
            EvalRow(id="q-2", faithfulness=0.7, relevance=0.7, rubric=3.0, source_name="Spark"),
            EvalRow(id="q-3", faithfulness=0.8, relevance=0.6, rubric=3.5, source_name="Delta"),
        ]
        metrics = compute_platform_metrics(rows)
        assert "Spark" in metrics
        assert "Delta" in metrics
        assert metrics["Spark"].total == 2
        assert metrics["Spark"].faithfulness_mean == pytest.approx(0.8)
        assert metrics["Spark"].relevance_mean == pytest.approx(0.8)
        assert metrics["Spark"].rubric_mean == pytest.approx(3.5)
        assert metrics["Delta"].total == 1

    def test_unknown_platform(self):
        rows = [EvalRow(id="q-1", faithfulness=0.9, relevance=0.9, rubric=4.0, source_name=None)]
        metrics = compute_platform_metrics(rows)
        assert "unknown" in metrics


class TestFindWorstRows:
    def test_returns_bottom_n_per_platform(self):
        rows = [
            EvalRow(id="q-1", faithfulness=0.9, relevance=0.9, rubric=5.0, source_name="Spark"),
            EvalRow(id="q-2", faithfulness=0.7, relevance=0.7, rubric=3.0, source_name="Spark"),
            EvalRow(id="q-3", faithfulness=0.5, relevance=0.5, rubric=2.0, source_name="Spark"),
            EvalRow(id="q-4", faithfulness=0.8, relevance=0.8, rubric=4.0, source_name="Delta"),
        ]
        metrics = compute_platform_metrics(rows)
        worst = find_worst_rows(metrics, n=2)
        assert len(worst["Spark"]) == 2
        assert worst["Spark"][0].id == "q-3"
        assert worst["Spark"][1].id == "q-2"
        assert worst["Delta"][0].id == "q-4"

    def test_defaults_to_3(self):
        rows = [
            EvalRow(id="q-1", faithfulness=0.9, relevance=0.9, rubric=5.0, source_name="Spark"),
            EvalRow(id="q-2", faithfulness=0.8, relevance=0.8, rubric=4.0, source_name="Spark"),
            EvalRow(id="q-3", faithfulness=0.7, relevance=0.7, rubric=3.0, source_name="Spark"),
            EvalRow(id="q-4", faithfulness=0.6, relevance=0.6, rubric=2.0, source_name="Spark"),
            EvalRow(id="q-5", faithfulness=0.5, relevance=0.5, rubric=1.0, source_name="Spark"),
        ]
        metrics = compute_platform_metrics(rows)
        worst = find_worst_rows(metrics)
        assert len(worst["Spark"]) == DEFAULT_WORST_N


class TestFormatPlatformSummary:
    def test_shows_gate_status(self):
        pm = PlatformMetrics(
            source_name="Spark",
            rows=[],
            faithfulness_mean=0.90,
            relevance_mean=0.85,
            rubric_mean=4.2,
            composite_mean=0.86,
            total=10,
        )
        summary = format_platform_summary(pm)
        assert "Spark" in summary
        assert "n=10" in summary
        assert "✓" in summary

    def test_shows_failure_markers(self):
        pm = PlatformMetrics(
            source_name="Delta",
            rows=[],
            faithfulness_mean=0.70,
            relevance_mean=0.75,
            rubric_mean=3.5,
            composite_mean=0.70,
            total=5,
        )
        summary = format_platform_summary(pm)
        assert "✗" in summary


class TestFormatWorstRow:
    def test_truncates_long_question(self):
        row = EvalRow(
            id="x",
            faithfulness=0.5,
            relevance=0.5,
            rubric=2.0,
            question="A" * 120,
            source_name="Spark",
        )
        formatted = format_worst_row(row)
        assert "..." in formatted
        assert len(formatted) > 0

    def test_shows_threshold_flags(self):
        row = EvalRow(
            id="y",
            faithfulness=0.5,
            relevance=0.6,
            rubric=2.0,
            question="Why is Delta Lake better?",
            source_name="Delta",
        )
        formatted = format_worst_row(row)
        assert "y" in formatted
        assert "faithfulness" in formatted
        assert "rubric" in formatted


class TestScanWorstRows:
    def test_full_pipeline(self, tmp_path):
        golden = tmp_path / "golden.jsonl"
        golden.write_text(
            '{"id":"r-1","question":"a","ground_truth":"b","source_name":"Spark"}\n'
            '{"id":"r-2","question":"c","ground_truth":"d","source_name":"Delta"}\n'
            '{"id":"r-3","question":"e","ground_truth":"f","source_name":"Spark"}\n',
            encoding="utf-8",
        )
        results = tmp_path / "results.jsonl"
        results.write_text(
            '{"id":"r-1","faithfulness":0.9,"relevance":0.9,"rubric":5.0,"question":"a","answer":"x"}\n'
            '{"id":"r-2","faithfulness":0.5,"relevance":0.5,"rubric":2.0,"question":"c","answer":"y"}\n'
            '{"id":"r-3","faithfulness":0.7,"relevance":0.7,"rubric":3.5,"question":"e","answer":"z"}\n',
            encoding="utf-8",
        )

        report = scan_worst_rows(results, [golden], n=2)

        assert report["total_rows"] == 3
        assert report["total_platforms"] == 2
        assert "Spark" in report["platforms"]
        assert "Delta" in report["platforms"]
        assert report["platforms"]["Delta"].faithfulness_mean == pytest.approx(0.5)
        worst = report["worst_rows"]
        assert len(worst["Delta"]) == 1
        assert worst["Delta"][0].id == "r-2"
        assert len(worst["Spark"]) == 2

    def test_merges_multiple_golden_files(self, tmp_path):
        g1 = tmp_path / "g1.jsonl"
        g1.write_text('{"id":"r-1","question":"a","ground_truth":"b","source_name":"Spark"}\n', encoding="utf-8")
        g2 = tmp_path / "g2.jsonl"
        g2.write_text('{"id":"r-2","question":"c","ground_truth":"d","source_name":"Delta"}\n', encoding="utf-8")
        results = tmp_path / "results.jsonl"
        results.write_text(
            '{"id":"r-1","faithfulness":0.9,"relevance":0.9,"rubric":4.0,"question":"a","answer":"x"}\n'
            '{"id":"r-2","faithfulness":0.5,"relevance":0.5,"rubric":2.0,"question":"c","answer":"y"}\n',
            encoding="utf-8",
        )

        report = scan_worst_rows(results, [g1, g2])
        assert report["total_platforms"] == 2
