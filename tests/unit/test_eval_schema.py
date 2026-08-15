"""Tests for the unified evaluation dataset schema."""

from __future__ import annotations

from data_engineering_copilot.evaluation.eval_schema import (
    EvalKind,
    kind_of,
    parse_eval_rows,
    validate_eval_row,
    write_eval_rows,
)


class TestKindOf:
    def test_recall_row(self):
        row = {"id": "a-1", "question": "q", "expected_terms": ["sum"], "expected_urls": ["u"]}
        assert kind_of(row) is EvalKind.RECALL

    def test_qa_row(self):
        row = {"id": "a-1", "question": "q", "ground_truth": "g", "contexts": ["c"]}
        assert kind_of(row) is EvalKind.QA

    def test_empty_is_qa(self):
        assert kind_of({"question": "q"}) is EvalKind.QA


class TestValidateEvalRow:
    def test_valid_recall_row(self):
        row = {
            "id": "spark-window-1",
            "question": "How do I window?",
            "expected_terms": ["Window", "orderBy"],
            "expected_urls": ["https://raw.githubusercontent.com/apache/spark/x/docs/window.md"],
            "expected_doc_types": ["guide"],
            "forbidden_terms": [],
        }
        assert validate_eval_row(row) == []

    def test_valid_qa_row(self):
        row = {"id": "airflow-dag-1", "question": "What is a DAG?", "ground_truth": "A DAG is...", "contexts": ["..."]}
        assert validate_eval_row(row) == []

    def test_missing_question(self):
        assert validate_eval_row({"id": "x", "expected_terms": ["t"], "expected_urls": ["u"]})

    def test_missing_id(self):
        row = {"question": "q", "expected_terms": ["t"], "expected_urls": ["u"]}
        assert any("id is required" in e for e in validate_eval_row(row))

    def test_bad_slug(self):
        row = {"id": "Bad Slug!", "question": "q", "expected_terms": ["t"], "expected_urls": ["u"]}
        assert any("must match" in e for e in validate_eval_row(row))

    def test_recall_without_terms(self):
        row = {"id": "x", "question": "q", "expected_urls": ["u"]}
        assert any("expected_terms" in e for e in validate_eval_row(row))

    def test_oos_cannot_have_urls(self):
        row = {"id": "oos-1", "question": "q", "out_of_scope": True, "expected_terms": ["t"], "expected_urls": ["u"]}
        assert any("out-of-scope row must not carry" in e for e in validate_eval_row(row))

    def test_oos_must_have_terms(self):
        row = {"id": "oos-1", "question": "q", "out_of_scope": True}
        assert any("expected_terms" in e for e in validate_eval_row(row))

    def test_in_scope_recall_requires_urls(self):
        row = {"id": "x", "question": "q", "expected_terms": ["t"]}
        assert any("expected_urls" in e for e in validate_eval_row(row))

    def test_qa_without_ground_truth(self):
        row = {"id": "x", "question": "q", "contexts": ["c"]}
        assert any("ground_truth" in e for e in validate_eval_row(row))


class TestParseWrite:
    def test_roundtrip(self, tmp_path):
        rows = [{"id": "a", "question": "q1", "expected_terms": ["x"], "expected_urls": ["u"]}]
        path = tmp_path / "eval.jsonl"
        write_eval_rows(path, rows)
        assert parse_eval_rows(path) == rows

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "eval.jsonl"
        path.write_text('{"id":"a","question":"q"}\n\n{"id":"b","question":"q2"}\n', encoding="utf-8")
        assert len(parse_eval_rows(path)) == 2
