"""Tests for the offline candidate-diversity benchmark."""

from __future__ import annotations

from data_engineering_copilot.evaluation.context_diversity_benchmark import (
    DiversityScenario,
    _chunk,
    _jaccard,
    _measure,
    _select_current,
    _select_mmr,
    _tokenize,
    default_scenarios,
    run_context_diversity_benchmark,
)


class TestTokenHelpers:
    def test_tokenize_normalizes_case(self):
        assert "spark" in _tokenize("Apache Spark")

    def test_jaccard_identical(self):
        assert _jaccard(_tokenize("a b c"), _tokenize("a b c")) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard(_tokenize("a b"), _tokenize("c d")) == 0.0


class TestSelectionStrategies:
    def test_current_collapses_siblings(self):
        text = "shared parent context repeated across sibling children"
        chunks = [
            _chunk(f"sib-{i}", text, source_name="spark", confidence=0.95 - i * 0.01, parent_chunk_id="parent")
            for i in range(3)
        ]
        selected = _select_current(chunks, top_k=3)
        assert len(selected) == 1

    def test_current_keeps_distinct_facts(self):
        chunks = [
            _chunk(
                "a", "filter a DataFrame with isNotNull to keep rows with a value", source_name="spark", confidence=0.93
            ),
            _chunk(
                "b", "filter a DataFrame with isNotNull to drop null rows", source_name="sql-guide", confidence=0.91
            ),
        ]
        selected = _select_current(chunks, top_k=2)
        assert len(selected) == 2

    def test_mmr_returns_at_most_top_k(self):
        chunks = [
            _chunk(
                f"c{i}",
                f"distinct technical term number {i} about spark",
                source_name="spark",
                confidence=0.9 - i * 0.05,
            )
            for i in range(8)
        ]
        selected = _select_mmr(chunks, top_k=4, lambda_param=0.5)
        assert len(selected) == 4


class TestMetrics:
    def test_duplicate_rate_counts_near_identical_second_copy(self):
        scenario = DiversityScenario(
            id="s",
            query="q",
            chunks=(
                _chunk(
                    "dup-1", "spark.sql.shuffle.partitions controls shuffle partitions", source_name="a", confidence=0.9
                ),
                _chunk(
                    "dup-2",
                    "spark.sql.shuffle.partitions controls shuffle partitions",
                    source_name="b",
                    confidence=0.88,
                ),
            ),
            required_facts=("spark.sql.shuffle.partitions",),
            top_k=2,
        )
        metrics = _measure("current", "s", list(scenario.chunks), scenario)
        assert metrics.duplicate_rate == 0.5

    def test_source_coverage_and_groundedness(self):
        scenario = DiversityScenario(
            id="s",
            query="q",
            chunks=(
                _chunk(
                    "a", "window functions support rowsBetween frame boundaries", source_name="spark", confidence=0.9
                ),
                _chunk(
                    "b", "partitionBy groups rows before window evaluation", source_name="sql-guide", confidence=0.8
                ),
            ),
            required_facts=("rowsBetween",),
            top_k=2,
        )
        metrics = _measure("current", "s", list(scenario.chunks), scenario)
        assert metrics.source_coverage == 1.0
        assert metrics.groundedness == 1.0
        assert metrics.selected_items == 2


class TestBenchmarkRun:
    def test_default_scenarios_run_deterministically(self):
        report1 = run_context_diversity_benchmark()
        report2 = run_context_diversity_benchmark()
        assert report1.as_dict() == report2.as_dict()
        assert len(report1.scenarios) == 3

    def test_scenario_ids_unique_and_schema(self):
        for scenario in default_scenarios():
            assert scenario.id
            assert scenario.expected_sources
            assert scenario.top_k > 0

    def test_gate_reports_decision_in_notes(self):
        report = run_context_diversity_benchmark()
        assert report.notes
        assert report.decision in ("keep_current", "enable_mmr")
        # A rejection must explain why in the notes.
        if not report.gate_passed:
            assert "keep current" in report.notes[0].lower() or "no strategy" in report.notes[0].lower()

    def test_passes_matches_gate_decision(self):
        report = run_context_diversity_benchmark()
        assert report.passes() is report.gate_passed

    def test_empty_scenarios_are_safe(self):
        report = run_context_diversity_benchmark(scenarios=[])
        assert report.scenarios == ()
        assert report.gate_passed is False
        assert report.decision == "keep_current"


class TestCli:
    def test_main_runs(self, capsys):
        from data_engineering_copilot.evaluation.context_diversity_benchmark import main

        code = main([])
        out = capsys.readouterr().out
        assert "decision=" in out
        assert isinstance(code, int)

    def test_main_accepts_mmr_lambda(self):
        from data_engineering_copilot.evaluation.context_diversity_benchmark import main

        assert isinstance(main(["--mmr-lambda", "0.7"]), int)
