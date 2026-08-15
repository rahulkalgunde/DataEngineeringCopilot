"""Tests for the synthetic recall-eval generator (offline deterministic path)."""

from __future__ import annotations

import json

from data_engineering_copilot.evaluation.eval_schema import validate_eval_row
from data_engineering_copilot.evaluation.synthetic_generator import (
    chunks_for_source,
    deterministic_candidates,
    gate_and_write,
    generate,
)
from data_engineering_copilot.services.eval_coverage import CoverageValidator


def _build_corpus(tmp_path):
    gen = tmp_path / "gen"
    gen.mkdir()
    rows = [
        {
            "url": "https://raw.githubusercontent.com/apache/spark/abc/docs/window.md",
            "source_name": "Apache Spark 4.0.0",
            "title": "Window Functions",
            "heading_path": ["Window Functions", "OVER Clause"],
            "doc_type": "guide",
            "text": "Window functions use orderBy and rangeBetween to compute rolling sums.",
        },
        {
            "url": "https://platform.claude.com/docs/en/api/messages.md",
            "source_name": "Claude Platform Docs",
            "title": "Messages API",
            "heading_path": ["Messages API"],
            "doc_type": "guide",
            "text": "The Messages API accepts a model and system prompt.",
        },
    ]
    with open(gen / "chunks.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return gen


def test_chunks_for_source(tmp_path):
    gen = _build_corpus(tmp_path)
    spark = chunks_for_source(gen, "Apache Spark 4.0.0")
    assert len(spark) == 1
    assert spark[0]["title"] == "Window Functions"


def test_deterministic_candidates_are_valid_and_grounded(tmp_path):
    gen = _build_corpus(tmp_path)
    chunks = chunks_for_source(gen, "Apache Spark 4.0.0")
    rows = deterministic_candidates(chunks, source="Apache Spark 4.0.0", limit=10)
    assert rows
    validator = CoverageValidator(gen)
    for row in rows:
        assert validate_eval_row(row) == []
        assert validator.validate_row(row)["status"] == "pass"


def test_gate_and_write(tmp_path):
    gen = _build_corpus(tmp_path)
    chunks = chunks_for_source(gen, "Claude Platform Docs")
    rows = deterministic_candidates(chunks, source="Claude Platform Docs", limit=10)
    out = tmp_path / "recall_synthetic_claude.jsonl"
    written = gate_and_write(rows, out, CoverageValidator(gen))
    assert written == len(rows)
    assert out.exists()
    assert len([json.loads(line) for line in out.read_text().splitlines() if line.strip()]) == written


def test_generate_without_ragas_falls_back_to_deterministic(tmp_path):
    gen = _build_corpus(tmp_path)
    out = tmp_path / "recall_synthetic_spark.jsonl"
    written = generate(gen, "Apache Spark 4.0.0", out, limit=5)
    assert written == 1  # only one spark chunk
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert all(r["source_name"] == "Apache Spark 4.0.0" for r in rows)
