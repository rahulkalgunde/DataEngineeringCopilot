"""Tests for the corpus-coverage validator (eval datasets vs active generation)."""

from __future__ import annotations

import json

import pytest

from data_engineering_copilot.services.eval_coverage import CoverageValidator, resolve_generation_root


def _build_generation(tmp_path):
    """Create a minimal generation corpus directory (chunks.jsonl)."""
    gen = tmp_path / "pinned_gen_test"
    gen.mkdir(parents=True)
    rows = [
        {
            "url": "https://raw.githubusercontent.com/apache/spark/abc/docs/window.md",
            "source_name": "Apache Spark 4.0.0",
            "text": "Window functions use orderBy and rangeBetween to compute rolling sums.",
        },
        {
            "url": "https://platform.claude.com/docs/api/messages.md",
            "source_name": "Claude Platform Docs",
            "text": "The Messages API accepts a model and system prompt and returns text.",
        },
    ]
    with open(gen / "chunks.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return gen


class TestResolveGenerationRoot:
    def test_pinned_corpus(self, tmp_path):
        (tmp_path / "pinned_corpus" / "g1").mkdir(parents=True)
        (tmp_path / "pinned_corpus" / "g1" / "chunks.jsonl").touch()
        assert resolve_generation_root("g1", tmp_path) == tmp_path / "pinned_corpus" / "g1"

    def test_spark_corpus(self, tmp_path):
        (tmp_path / "spark_corpus" / "g1").mkdir(parents=True)
        (tmp_path / "spark_corpus" / "g1" / "chunks.jsonl").touch()
        assert resolve_generation_root("g1", tmp_path) == tmp_path / "spark_corpus" / "g1"

    def test_pinned_corpus_naming_contract_layout(self, tmp_path):
        # gen-build writes artifact dirs named after the COLLECTION
        # (data_engineering_docs__<gen>) per config/naming.py — the resolver
        # must find them or eval-fast/eval-coverage fail for every
        # contract-compliant generation.
        gen = tmp_path / "pinned_corpus" / "data_engineering_docs__ci-repro"
        gen.mkdir(parents=True)
        (gen / "chunks.jsonl").touch()
        assert (
            resolve_generation_root("ci-repro", tmp_path)
            == tmp_path / "pinned_corpus" / "data_engineering_docs__ci-repro"
        )

    def test_missing(self, tmp_path):
        assert resolve_generation_root("nope", tmp_path) is None


class TestCoverageValidator:
    def test_url_covered_and_terms(self, tmp_path):
        v = CoverageValidator(_build_generation(tmp_path))
        assert v.indexed_url_count == 2
        assert v.url_covered("https://raw.githubusercontent.com/apache/spark/abc/docs/window.md")
        assert not v.url_covered("https://example.com/nope.md")

    def test_validate_row_pass(self, tmp_path):
        v = CoverageValidator(_build_generation(tmp_path))
        row = {
            "id": "spark-window-1",
            "question": "rolling sum?",
            "expected_terms": ["orderBy", "rangeBetween"],
            "expected_urls": ["https://raw.githubusercontent.com/apache/spark/abc/docs/window.md"],
            "source_name": "Apache Spark 4.0.0",
        }
        verdict = v.validate_row(row)
        assert verdict["status"] == "pass"
        assert verdict["missing_urls"] == []
        assert verdict["missing_terms"] == []

    def test_validate_row_missing_url(self, tmp_path):
        v = CoverageValidator(_build_generation(tmp_path))
        row = {
            "id": "spark-window-1",
            "question": "rolling sum?",
            "expected_terms": ["orderBy"],
            "expected_urls": ["https://raw.githubusercontent.com/apache/spark/abc/docs/not_indexed.md"],
        }
        assert v.validate_row(row)["status"] == "fail"

    def test_validate_row_term_absent_from_corpus(self, tmp_path):
        v = CoverageValidator(_build_generation(tmp_path))
        row = {
            "id": "spark-window-1",
            "question": "rolling sum?",
            "expected_terms": ["zzznozzz"],
            "expected_urls": ["https://raw.githubusercontent.com/apache/spark/abc/docs/window.md"],
        }
        verdict = v.validate_row(row)
        assert verdict["status"] == "fail"
        assert verdict["missing_terms"] == ["zzznozzz"]

    def test_out_of_scope_always_passes(self, tmp_path):
        v = CoverageValidator(_build_generation(tmp_path))
        row = {"id": "oos-1", "question": "react hooks?", "out_of_scope": True, "expected_terms": ["react"]}
        assert v.validate_row(row)["status"] == "pass"

    def test_report_aggregates(self, tmp_path):
        v = CoverageValidator(_build_generation(tmp_path))
        rows = [
            {
                "id": "a",
                "question": "q",
                "expected_terms": ["orderBy"],
                "expected_urls": ["https://raw.githubusercontent.com/apache/spark/abc/docs/window.md"],
                "source_name": "Apache Spark 4.0.0",
            },
            {"id": "b", "question": "q2", "out_of_scope": True, "expected_terms": ["react"]},
        ]
        report = v.report(rows)
        assert report["rows"] == 2
        assert report["pass"] == 2
        assert report["fail"] == 0

    def test_empty_corpus_fails_terms(self, tmp_path):
        gen = tmp_path / "gen"
        gen.mkdir()
        (gen / "chunks.jsonl").write_text("", encoding="utf-8")
        v = CoverageValidator(gen)
        row = {
            "id": "x",
            "question": "q",
            "expected_terms": ["anything"],
            "expected_urls": ["https://example.com/x.md"],
        }
        assert v.validate_row(row)["status"] == "fail"

    @pytest.mark.parametrize("term", ["Window", "orderBy"])
    def test_term_present_case_insensitive(self, tmp_path, term):
        v = CoverageValidator(_build_generation(tmp_path))
        assert v.term_present(term, source="Apache Spark 4.0.0")
