"""Tests for QA eval run metrics (answer correctness vs ground truth)."""

from __future__ import annotations

from data_engineering_copilot.services.rag_evaluation import answer_token_f1


class TestAnswerTokenF1:
    def test_exact(self):
        assert answer_token_f1("Spark is an engine", "Spark is an engine") == 1.0

    def test_no_overlap(self):
        assert answer_token_f1("Spark engine", "Rust compiler") == 0.0

    def test_partial(self):
        f1 = answer_token_f1("unified analytics engine for large scale data", "unified engine")
        assert 0.0 < f1 < 1.0

    def test_empty_reference(self):
        assert answer_token_f1("", "anything") == 0.0

    def test_empty_hypothesis(self):
        assert answer_token_f1("reference here", "") == 0.0

    def test_case_and_punctuation_insensitive(self):
        assert answer_token_f1("Spark SQL, ORDER BY", "spark sql order by") == 1.0
