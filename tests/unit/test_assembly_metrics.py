"""Tests for assembly evaluation metrics."""

from __future__ import annotations

import pytest

from data_engineering_copilot.evaluation.assembly_metrics import (
    context_compression_ratio,
    duplicate_candidate_rate,
    needle_loss_rate,
    source_coverage_rate,
)

pytestmark = pytest.mark.unit


class TestDuplicateRate:
    def test_no_duplicates(self):
        assert duplicate_candidate_rate("hello world foo bar") == 0.0

    def test_all_duplicates(self):
        assert duplicate_candidate_rate("hello hello hello") == pytest.approx(2 / 3)

    def test_empty(self):
        assert duplicate_candidate_rate("") == 0.0


class TestSourceCoverage:
    def test_full_coverage(self):
        assert source_coverage_rate(["a", "b", "c"], 3) == 1.0

    def test_partial_coverage(self):
        assert source_coverage_rate(["a", "b"], 4) == 0.5

    def test_zero_sources(self):
        assert source_coverage_rate([], 5) == 0.0


class TestCompressionRatio:
    def test_no_compression(self):
        assert context_compression_ratio(1000, 1000) == 1.0

    def test_half_compression(self):
        assert context_compression_ratio(500, 1000) == 0.5

    def test_zero_candidates(self):
        assert context_compression_ratio(100, 0) == 0.0


class TestNeedleLoss:
    def test_all_needles_present(self):
        text = "Spark SQL supports window functions for ranking rows within partitions"
        facts = ["window functions for ranking", "sql supports window functions"]
        assert needle_loss_rate(text, facts) == 0.0

    def test_needle_missing(self):
        text = "Airflow DAG scheduling configuration"
        facts = ["window functions for ranking"]
        assert needle_loss_rate(text, facts) == 1.0

    def test_empty_facts(self):
        assert needle_loss_rate("anything", []) == 0.0

    def test_partial_needles(self):
        text = "Spark SQL window functions for data processing"
        facts = ["window functions for data processing", "Airflow DAG scheduling for retries"]
        assert needle_loss_rate(text, facts) == 0.5
